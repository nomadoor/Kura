# This script runs inside the pinned sd-scripts and Musubi Tuner containers, which save
# training state with Accelerate. Kura writes it into a run's resolved directory at compile.
# Do not import kura here.

import hashlib
import json
import os
import pathlib
import runpy
import sys


# The backends this runner serves. sd-scripts also writes train_state.json, which its Resume
# reads for the application step, so Kura normalizes it too.
BACKENDS = ("sd-scripts", "musubi-tuner")


def fail(message):
    raise RuntimeError("[kura] Accelerate training-state failure: " + message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_completed_step(state):
    entries = state.get("state") if isinstance(state, dict) else None
    if not isinstance(entries, dict) or not entries:
        return None
    steps = set()
    for value in entries.values():
        step = value.get("step") if isinstance(value, dict) else None
        if hasattr(step, "item"):
            step = step.item()
        if step is None:
            continue
        try:
            steps.add(int(step))
        except (TypeError, ValueError):
            fail("optimizer state has an invalid completed-update counter")
    if not steps:
        return None
    if len(steps) != 1 or next(iter(steps)) <= 0:
        fail("optimizer state has no consistent completed-update counter")
    return next(iter(steps))


def fsync_replace(temporary, target):
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def normalize_saved_state(output_dir, backend, num_processes):
    """Record the step a complete Accelerate save holds, from its scheduler checked against its
    optimizer, in kura-state-info.json, written last; for sd-scripts also set train_state.json.

    Accelerate steps a prepared scheduler once per process for each optimizer update (both
    trainers scale its length by the process count), so the step is last_epoch // num_processes,
    as the patched Musubi trainer reads it on Resume."""
    import torch

    output = pathlib.Path(output_dir)
    scheduler_path = output / "scheduler.bin"
    optimizer_path = output / "optimizer.bin"
    try:
        scheduler = torch.load(scheduler_path, map_location="cpu", weights_only=True)
        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        fail(f"cannot inspect persisted state: {exc}")
    scheduler_step = scheduler.get("last_epoch") if isinstance(scheduler, dict) else None
    if isinstance(scheduler_step, bool) or not isinstance(scheduler_step, int) or scheduler_step <= 0:
        fail("scheduler state has no cumulative last_epoch")
    scheduler_calls = scheduler.get("_step_count") if isinstance(scheduler, dict) else None
    if isinstance(scheduler_calls, int) and not isinstance(scheduler_calls, bool) and scheduler_calls - 1 != scheduler_step:
        fail(f"scheduler _step_count {scheduler_calls} does not match last_epoch {scheduler_step}")
    if scheduler_step % num_processes:
        fail(f"scheduler last_epoch {scheduler_step} is not a whole number of updates on {num_processes} processes")
    scheduler_step //= num_processes
    optimizer_step = optimizer_completed_step(optimizer)
    if optimizer_step is not None and optimizer_step != scheduler_step:
        fail(f"optimizer step {optimizer_step} does not match scheduler step {scheduler_step}")
    info = {"schema_version": 1, "backend": backend, "logical_step": scheduler_step}
    if backend == "sd-scripts":
        train_state_path = output / "train_state.json"
        try:
            train_state = json.loads(train_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"cannot inspect persisted state: {exc}")
        if not isinstance(train_state, dict):
            fail("train_state.json is not an object")
        train_state["current_step"] = scheduler_step
        train_state_tmp = output / "train_state.json.tmp"
        train_state_tmp.write_text(json.dumps(train_state, sort_keys=True) + "\n", encoding="utf-8")
        fsync_replace(train_state_tmp, train_state_path)
        info["train_state_sha256"] = sha256_file(train_state_path)
    info["optimizer_sha256"] = sha256_file(optimizer_path)
    info["scheduler_sha256"] = sha256_file(scheduler_path)
    info_tmp = output / "kura-state-info.json.tmp"
    info_tmp.write_text(json.dumps(info, sort_keys=True) + "\n", encoding="utf-8")
    fsync_replace(info_tmp, output / "kura-state-info.json")
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def install_hooks(backend):
    if backend not in BACKENDS:
        fail(f"unsupported backend {backend!r}")
    from accelerate import Accelerator

    original_save_state = Accelerator.save_state

    def save_state(accelerator, *args, **kwargs):
        output_dir = args[0] if args else kwargs.get("output_dir")
        if output_dir is None:
            fail("Accelerate save_state did not declare its output directory")
        output = pathlib.Path(output_dir)
        marker = output / "kura-state-info.json"
        if marker.exists():
            marker.unlink()
            directory_fd = os.open(output, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        result = original_save_state(accelerator, *args, **kwargs)
        normalize_saved_state(output_dir, backend, accelerator.num_processes)
        return result

    Accelerator.save_state = save_state


def main():
    # state-runner.py BACKEND ENTRYPOINT [ARGS...]: run the trainer as `ENTRYPOINT ARGS...`.
    if len(sys.argv) < 3:
        fail("usage: state-runner.py BACKEND ENTRYPOINT [ARGS...]")
    backend, entrypoint, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
    install_hooks(backend)
    sys.argv = [entrypoint, *argv]
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
