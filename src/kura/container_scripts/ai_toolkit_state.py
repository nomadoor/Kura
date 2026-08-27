# This script runs inside the pinned AI Toolkit container.
# Do not import kura here; it is delivered as `python -c` source text.

import hashlib
import json
import os
import pathlib
import runpy
import shutil
import sys
import tempfile


def fail(message):
    raise RuntimeError("[kura] AI Toolkit training-state failure: " + message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path):
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def metadata_step(path):
    from toolkit.metadata import load_metadata_from_safetensors

    metadata = load_metadata_from_safetensors(str(path))
    training = metadata.get("training_info") if isinstance(metadata, dict) else None
    step = training.get("step") if isinstance(training, dict) else None
    try:
        return int(step)
    except (TypeError, ValueError):
        fail(f"saved weight has no valid training_info.step: {path}")


def optimizer_completed_step(optimizer_state):
    states = optimizer_state.get("state") if isinstance(optimizer_state, dict) else None
    if not isinstance(states, dict) or not states:
        fail("saved optimizer state has no parameter states")
    steps = set()
    for value in states.values():
        step = value.get("step") if isinstance(value, dict) else None
        if hasattr(step, "item"):
            step = step.item()
        try:
            step = int(step)
        except (TypeError, ValueError):
            fail("saved optimizer state has no consistent completed-update counter")
        steps.add(step)
    if len(steps) != 1 or next(iter(steps)) <= 0:
        fail("saved optimizer state has no consistent completed-update counter")
    return next(iter(steps))


def weight_tensor_digest(path):
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) == 8:
            header_size = int.from_bytes(prefix, "little", signed=False)
            if 0 < header_size < path.stat().st_size - 8:
                handle.seek(8 + header_size)
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                return digest.hexdigest()
    return sha256_file(path)


def state_values_equal(left, right, torch):
    if hasattr(torch, "is_tensor") and torch.is_tensor(left):
        return torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            state_values_equal(left[key], right[key], torch) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            state_values_equal(a, b, torch) for a, b in zip(left, right)
        )
    return left == right


def saved_states_equivalent(existing, staged):
    import torch

    existing_weight = existing / "model.safetensors"
    staged_weight = staged / "model.safetensors"
    existing_optimizer = existing / "optimizer.pt"
    staged_optimizer = staged / "optimizer.pt"
    if not all(path.is_file() for path in (existing_weight, staged_weight, existing_optimizer, staged_optimizer)):
        return False
    if weight_tensor_digest(existing_weight) != weight_tensor_digest(staged_weight):
        return False
    try:
        left = torch.load(existing_optimizer, map_location="cpu", weights_only=True)
        right = torch.load(staged_optimizer, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return state_values_equal(left, right, torch)


def copy_weight_with_logical_step(source, target, logical_step):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    try:
        training = json.loads(metadata.get("training_info", "{}"))
    except (TypeError, json.JSONDecodeError):
        training = {}
    if not isinstance(training, dict):
        training = {}
    training["step"] = int(logical_step)
    metadata["training_info"] = json.dumps(training)
    tensors = load_file(str(source), device="cpu")
    save_file(tensors, str(target), metadata=metadata)


def publish_generation(process, step, spec):
    if not process.accelerator.is_main_process:
        return
    import torch

    suffix = f"_{int(step):09d}" if step is not None else ""
    weight = pathlib.Path(process.save_root) / f"{process.job.name}{suffix}.safetensors"
    if not weight.is_file():
        fail(f"paired weight was not saved: {weight}")
    try:
        optimizer_state = process.optimizer.state_dict()
        logical_step = optimizer_completed_step(optimizer_state)
    except Exception as exc:
        fail(f"cannot inspect paired optimizer state: {exc}")
    state_root = pathlib.Path(spec["state_root"])
    state_root.mkdir(parents=True, exist_ok=True)
    destination = state_root / f"{spec['run_id']}-step{logical_step:08d}-state"
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=state_root))
    try:
        model_target = staging / "model.safetensors"
        copy_weight_with_logical_step(weight, model_target, logical_step)
        if metadata_step(model_target) != logical_step:
            fail(f"paired weight step does not match completed optimizer updates {logical_step}: {weight}")
        optimizer_tmp = staging / "optimizer.pt.tmp"
        try:
            torch.save(optimizer_state, optimizer_tmp)
            loaded = torch.load(optimizer_tmp, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                fail("saved optimizer state is not a mapping")
            del loaded, optimizer_state
        except Exception as exc:
            fail(f"cannot save paired optimizer state: {exc}")
        optimizer_target = staging / "optimizer.pt"
        os.replace(optimizer_tmp, optimizer_target)
        info = {
            "schema_version": 1,
            "backend": "ai-toolkit",
            "logical_step": logical_step,
            "weight_sha256": sha256_file(model_target),
            "optimizer_sha256": sha256_file(optimizer_target),
        }
        (staging / "state-info.json").write_text(json.dumps(info, sort_keys=True) + "\n", encoding="utf-8")
        for path in staging.iterdir():
            fsync_file(path)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists():
            if not saved_states_equivalent(destination, staging):
                fail(f"conflicting training-state generation already exists: {destination}")
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        generations = []
        prefix = f"{spec['run_id']}-step"
        for candidate in state_root.glob(f"{prefix}*-state"):
            middle = candidate.name[len(prefix):-len("-state")]
            if candidate.is_dir() and middle.isdigit():
                generations.append((int(middle), candidate))
        generations.sort(reverse=True)
        for _, old in generations[int(spec.get("keep_generations", 2)):]:
            shutil.rmtree(old)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def materialize_resume_source(spec):
    resume = spec.get("resume")
    if not isinstance(resume, dict):
        return None
    source = pathlib.Path(resume["payload"])
    weight = source / "model.safetensors"
    optimizer = source / "optimizer.pt"
    if not weight.is_file() or not optimizer.is_file():
        fail("Resume payload is missing model.safetensors or optimizer.pt")
    save_root = pathlib.Path(spec["state_root"]) / spec["run_id"]
    save_root.mkdir(parents=True, exist_ok=True)
    staged_weight = save_root / f"{spec['run_id']}_{int(resume['source_step']):09d}.safetensors"
    shutil.copy2(weight, staged_weight)
    shutil.copy2(optimizer, save_root / "optimizer.pt")
    return staged_weight.resolve()


def install_hooks(spec, expected_weight):
    import torch
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    original_load_weights = BaseSDTrainProcess.load_weights
    original_save = BaseSDTrainProcess.save
    original_before_loop = SDTrainer.hook_before_train_loop

    def load_weights(process, path):
        result = original_load_weights(process, path)
        process._kura_loaded_weight = pathlib.Path(path).resolve()
        return result

    def save(process, step=None):
        result = original_save(process, step)
        publish_generation(process, step, spec)
        return result

    def before_loop(process):
        resume = spec.get("resume")
        if isinstance(resume, dict):
            source_step = int(resume["source_step"])
            if getattr(process, "_kura_loaded_weight", None) != expected_weight:
                fail("trainer did not load the compiled Resume weight")
            if int(process.step_num) != source_step or int(process.start_step) != source_step:
                fail("trainer did not restore the compiled Resume step")
            if process.network is None or process.network.did_change_weights:
                fail("trainer changed the Resume network weights")
            optimizer_path = pathlib.Path(process.save_root) / "optimizer.pt"
            previous = [(group.get("lr"), group.get("initial_lr")) for group in process.optimizer.param_groups]
            try:
                state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
                process.optimizer.load_state_dict(state)
            except Exception as exc:
                fail(f"optimizer Resume load failed before the first update: {exc}")
            finally:
                if "state" in locals():
                    del state
            for group, (lr, initial_lr) in zip(process.optimizer.param_groups, previous):
                group["lr"] = lr
                group["initial_lr"] = lr if initial_lr is None else initial_lr
            print(f"[kura] AI Toolkit optimizer Resume verified at step {source_step}", flush=True)
        return original_before_loop(process)

    BaseSDTrainProcess.load_weights = load_weights
    BaseSDTrainProcess.save = save
    SDTrainer.hook_before_train_loop = before_loop


def main():
    if len(sys.argv) != 2:
        fail("usage: ai_toolkit_state.py SPEC_JSON")
    spec = json.loads(sys.argv[1])
    expected_weight = materialize_resume_source(spec)
    install_hooks(spec, expected_weight)
    sys.argv = ["run.py", spec["config_path"]]
    runpy.run_path("run.py", run_name="__main__")


if __name__ == "__main__":
    main()
