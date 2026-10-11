"""Resume step arithmetic has one owner, and every backend, executor, and reader agrees with it."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

import kura.monitor as monitor
import kura.run_commands.experiment as experiment
import kura.run_commands.plan as plan
from kura.backends.ai_toolkit import compile_ai_toolkit
from kura.backends.musubi_command import command_musubi_tuner
from kura.backends.sd_scripts import command_sd_scripts
from kura.dataset_transfer import _resume_entries
from kura.executors.common import _configured_final_step, _materialize_stdout_progress
from kura.executors.runpod import stage_runpod
from kura.run_commands.plan import _resume_plan_payload
from kura.run_commands.runpod_ssh import _pull_remote_training_state_items
from kura.run_envelope import final_step, resume_target_step
from kura.training_artifacts import (
    compile_resume_lock,
    logical_step,
    publish_completed_training_states,
    publish_training_state,
    recipe_fingerprint,
    resume_artifact_directory,
    resume_steps,
    training_state_payload,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only  # noqa: E402
from tests.test_training_resume import _safetensors_bytes, _torch_archive_bytes, _write_state_marker  # noqa: E402


RESTORATION = {
    "level": "best_effort_resume",
    "restored": ["model", "optimizer", "scheduler", "rng"],
    "not_restored": ["exact_dataloader_position"],
}
STATE_FILES = ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl")


def _state_bytes(name: str) -> bytes:
    return _safetensors_bytes(name.encode()) if name.endswith(".safetensors") else _torch_archive_bytes(name.encode())


def _write_state(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in STATE_FILES:
        (directory / name).write_bytes(_state_bytes(name))
    return directory


def musubi_run() -> dict[str, Any]:
    return {
        "id": "derived",
        "type": "train",
        "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
        "datasets": [{"id": "tiny", "digest": "sha256:abc"}],
        "recipe": {"steps": 1000, "seed": 42},
        "backend": {"name": "musubi-tuner", "config": {
            "architecture": "flux2",
            "model_version": "klein-base-4b",
            "network_dim": 4,
            "network_alpha": 4,
            "learning_rate": 1.0e-4,
            "batch_size": 1,
            "resolution": [512, 512],
            "model_paths": {
                "dit": "/models/flux2-klein-base-4b.safetensors",
                "vae": "/models/flux2-vae.safetensors",
                "text_encoder": "/models/qwen_3_4b.safetensors",
            },
        }},
    }


def sd_scripts_run() -> dict[str, Any]:
    return {
        "id": "derived",
        "type": "train",
        "recipe": {"steps": 1000, "seed": 42},
        "datasets": [{"id": "sample", "digest": "sha256:test"}],
        "backend": {"name": "sd-scripts", "config": {
            "architecture": "sd15",
            "mode": "lora",
            "model_paths": {"base": "/models/sd15.safetensors"},
            "dataset_config": {
                "general": {"resolution": [512, 512], "caption_extension": ".txt"},
                "datasets": [{"batch_size": 1, "subsets": [{"dataset_id": "sample", "num_repeats": 1}]}],
            },
            "network_dim": 8,
            "learning_rate": 0.0001,
            "optimizer_type": "AdamW8bit",
            "mixed_precision": "bf16",
        }},
    }


def ai_toolkit_run() -> dict[str, Any]:
    return {
        "id": "derived",
        "type": "train",
        "backend": {"name": "ai-toolkit", "adapter_version": 1, "config": {
            "model_arch": "flux2_klein_4b", "network_dim": 4, "network_alpha": 4, "learning_rate": 1.0e-4,
            "batch_size": 1, "gradient_checkpointing": False, "optimizer_type": "adamw8bit",
            "quantize": False, "quantize_te": False, "low_vram": False,
        }},
        "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
        "datasets": [{"id": "tiny", "digest": "sha256:abc"}],
        "recipe": {"steps": 1000, "seed": 42},
    }


def as_resume(run: dict[str, Any], *, artifact_id: str = "state-1", manifest_sha256: str = "a" * 64, source_step: int = 1000, additional: int = 200) -> dict[str, Any]:
    derived = deepcopy(run)
    derived["parent_run"] = "source"
    derived["continuation"] = {
        "mode": "resume",
        "source": {
            "artifact_id": artifact_id,
            "manifest_sha256": manifest_sha256,
            "observed_step": source_step,
            "recipe_sha256": recipe_fingerprint(run),
        },
        "additional_steps": additional,
        "target_step": source_step + additional,
        "restoration_contract": deepcopy(RESTORATION),
    }
    return derived


def publish_source(root: Path, run: dict[str, Any], *, step: int = 1000) -> dict[str, Any]:
    return publish_training_state(
        root,
        source_run="source",
        source_realization=None,
        backend=run["backend"]["name"],
        observed_step=step,
        candidate=_write_state(root / "candidate" / run["backend"]["name"]),
        native_format="accelerate-state-directory",
        restoration_contract=deepcopy(RESTORATION),
        compatibility={"recipe_sha256": recipe_fingerprint(run)},
    )


def compiled_native_target(run: dict[str, Any], scratch: Path) -> int:
    """The step count each trainer is told to stop at, read from its compiled command or config."""
    name = run["backend"]["name"]
    if name == "musubi-tuner":
        return int(re.search(r"--max_train_steps (\d+)", command_musubi_tuner(run)["argv"][2]).group(1))
    if name == "sd-scripts":
        return int(re.search(r"--max_train_steps (\d+)", command_sd_scripts(run)["argv"][2]).group(1))
    destination = scratch / "ai-toolkit"
    destination.parent.mkdir(parents=True, exist_ok=True)
    freeze_fixture(run, destination.parent)
    compile_ai_toolkit(run, destination)
    config = yaml.safe_load(destination.with_suffix(".yaml").read_text(encoding="utf-8"))
    return config["config"]["process"][0]["train"]["steps"]


class ResumeTargetTests(unittest.TestCase):
    def test_resume_target_step_is_the_one_rule_for_the_requested_target(self) -> None:
        self.assertEqual(resume_target_step(1000, 200, None), 1200)
        self.assertEqual(resume_target_step(1000, None, 1500), 1500)
        for observed, additional, to_step, message in (
            (1000, 200, 1500, "exactly one"),
            (1000, None, None, "exactly one"),
            (1000, 0, None, "positive integer"),
            (1000, True, None, "positive integer"),
            (1000, None, 1000, "greater than"),
            (1000, 0, None, "--additional-steps"),
            (1000, None, 1000, "--to-step"),
        ):
            with self.subTest(additional=additional, to_step=to_step), self.assertRaisesRegex(ValueError, message):
                resume_target_step(observed, additional, to_step)

    def test_final_step_is_the_resume_target_else_the_recipe_steps(self) -> None:
        self.assertEqual(final_step(musubi_run()), 1000)
        self.assertEqual(final_step(as_resume(musubi_run())), 1200)
        broken = as_resume(musubi_run())
        del broken["continuation"]["target_step"]
        with self.assertRaisesRegex(ValueError, "target_step"):
            final_step(broken)

    def test_an_invalid_continuation_gives_no_configured_final_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            broken = as_resume(musubi_run())
            broken["continuation"]["target_step"] = "1200"
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(broken), encoding="utf-8")
            self.assertEqual(_configured_final_step(run_dir), (None, None))
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(as_resume(musubi_run())), encoding="utf-8")
            self.assertEqual(_configured_final_step(run_dir), (1200, 1000))

    def test_training_state_payload_names_one_location(self) -> None:
        self.assertEqual(training_state_payload("state-1"), "/workspace/artifacts/training-state/state-1/payload")
        self.assertEqual(training_state_payload("state-1", root=None), "artifacts/training-state/state-1/payload")


class ResumeStepParityTests(unittest.TestCase):
    @posix_only(DATASET_IO)
    def test_every_backend_tells_its_trainer_the_target_the_plan_and_lock_show(self) -> None:
        # Resume +200 from step 1000: every pinned trainer continues the logical step, so its own
        # target is the logical one.
        expected = {"musubi-tuner": (1000, 1200), "sd-scripts": (1000, 1200), "ai-toolkit": (1000, 1200)}
        for build in (musubi_run, sd_scripts_run, ai_toolkit_run):
            source = build()
            name = source["backend"]["name"]
            with self.subTest(backend=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = publish_source(root, source)
                run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"])
                steps = resume_steps(run)
                self.assertEqual((steps["native_start"], steps["native_end"]), expected[name])
                self.assertEqual((steps["source_step"], steps["target_step"], steps["additional_steps"]), (1000, 1200, 200))
                self.assertEqual(compiled_native_target(run, root / "compile"), steps["native_end"])

                run_dir = root / "runs" / "derived"
                planned = _resume_plan_payload(root, run, run_dir)
                self.assertEqual((planned["native_start"], planned["native_target"]), expected[name])

                lock = compile_resume_lock(root, run, run_dir / "resolved")
                self.assertEqual(resume_steps(run, lock=lock), steps)
                self.assertEqual(lock["native_state_path"], training_state_payload(manifest["id"]))
                planned_from_lock = _resume_plan_payload(root, run, run_dir)
                self.assertEqual((planned_from_lock["native_start"], planned_from_lock["native_target"]), expected[name])

    def test_logical_step_adds_the_source_step_only_to_process_local_progress(self) -> None:
        self.assertEqual(logical_step(150, {"source_step": 1000, "native_progress": "process_local"}), 1150)
        self.assertEqual(logical_step(1150, {"source_step": 1000, "native_progress": "logical"}), 1150)
        self.assertEqual(logical_step(150, None), 150)


class ResumeExecutorParityTests(unittest.TestCase):
    def _derived(self, root: Path, *, process_local: bool = False) -> Path:
        # A Musubi Tuner Resume +200 from 1000. Its lock froze `logical` progress (the patched
        # trainer prints 1150/1200), or `process_local` when it was compiled before the patch
        # (the trainer printed 150/200).
        source = musubi_run()
        manifest = publish_source(root, source)
        run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"])
        run_dir = root / "runs" / "derived"
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        compile_resume_lock(root, run, run_dir / "resolved")
        progress = "150/200"
        if process_local:
            # Compiled by an older Kura: no state runner, so no marker, and a process-local lock.
            lock_path = run_dir / "resolved" / "training-state-source.lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock.update(native_progress="process_local", native_target="process_local")
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
        else:
            (run_dir / "resolved" / "musubi").mkdir()
            (run_dir / "resolved" / "musubi" / "state-runner.py").write_text("runner\n", encoding="utf-8")
            progress = "1150/1200"
        (run_dir / "logs" / "stdout.log").write_text(f"steps:  75%|███| {progress} [00:10<00:04, 1.0s/it, avr_loss=0.5]\n", encoding="utf-8")
        return run_dir

    def test_a_musubi_run_compiled_under_the_process_local_contract_is_read_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._derived(root, process_local=True)
            status: dict[str, Any] = {}
            _materialize_stdout_progress(run_dir, status, state="running")
            self.assertEqual((status["last_step"], status["total_steps"]), (1150, 1200))
            self.assertEqual((status["current_run_step"], status["current_run_total_steps"]), (150, 200))
            # Its states carry no marker and are placed by name through the lock, as before; the
            # same marker-less state of a run compiled with the runner is not complete yet.
            _write_state(run_dir / "outputs" / "derived-step0150-state")
            self.assertEqual([item["observed_step"] for item in publish_completed_training_states(root, run_dir)], [1150])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._derived(root)
            _write_state(run_dir / "outputs" / "derived-step1150-state")
            self.assertEqual(publish_completed_training_states(root, run_dir), [])

    def test_docker_and_runpod_place_progress_and_state_at_one_logical_step(self) -> None:
        import kura.executors.common as common
        import kura.run_commands.runpod_ssh as runpod_ssh
        import kura.training_artifacts as training_artifacts

        published: dict[str, list[int]] = {}
        statuses: dict[str, int] = {}
        for executor in ("docker", "runpod"):
            with tempfile.TemporaryDirectory() as directory, patch.object(
                training_artifacts, "state_logical_step", wraps=training_artifacts.state_logical_step,
            ) as owner, patch.object(common, "logical_step", wraps=logical_step) as status_owner, patch.object(
                runpod_ssh, "state_logical_step", wraps=training_artifacts.state_logical_step,
            ) as remote_owner:
                root = Path(directory)
                run_dir = self._derived(root)
                status: dict[str, Any] = {}
                _materialize_stdout_progress(run_dir, status, state="running")
                statuses[executor] = status["last_step"]
                status_owner.assert_called()
                name = "derived-step0150-state"
                if executor == "docker":
                    _write_state_marker(_write_state(run_dir / "outputs" / name), "musubi-tuner", 1150)
                    items = publish_completed_training_states(root, run_dir)
                else:
                    marked = _write_state(root / "remote" / name)
                    _write_state_marker(marked, "musubi-tuner", 1150)
                    item = {
                        "path": f"/workspace/runs/derived/outputs/{name}",
                        "name": name,
                        # The Pod reports the step the state's marker records.
                        "marked_step": 1150,
                        "files": [
                            {"path": file.relative_to(marked).as_posix(), "size": file.stat().st_size, "mtime_ns": 1}
                            for file in sorted(marked.iterdir())
                        ],
                    }

                    def fake_scp(command: list[str], **_: object) -> subprocess.CompletedProcess:
                        _write_state_marker(_write_state(Path(command[-1])), "musubi-tuner", 1150)
                        return subprocess.CompletedProcess(command, 0, "", "")

                    with patch.object(runpod_ssh, "_run_bounded", side_effect=fake_scp), patch.object(
                        runpod_ssh, "_runpod_remote_training_states", return_value=[item],
                    ), patch.object(runpod_ssh, "_workspace_config", return_value={}):
                        items = _pull_remote_training_state_items(
                            run_dir, {"ip": "example", "port": 22, "key": root / "key"}, workspace="/workspace", items=[item],
                        )
                    remote_owner.assert_called()
                owner.assert_called()
                published[executor] = [entry["observed_step"] for entry in items]
        self.assertEqual(statuses, {"docker": 1150, "runpod": 1150})
        self.assertEqual(published, {"docker": [1150], "runpod": [1150]})


    def test_an_unreadable_source_lock_is_refused_rather_than_read_as_not_a_resume(self) -> None:
        # Read as not a Resume, native step 150 would be published and shown as logical step 150.
        # The run was compiled before the state runner, so its state is placed through the lock.
        def non_int_steps(lock_path: Path) -> None:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["source_step"] = "1000"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

        def broken_json(lock_path: Path) -> None:
            lock_path.write_text("{not json", encoding="utf-8")

        for corrupt in (non_int_steps, broken_json):
            with self.subTest(corrupt=corrupt.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = self._derived(root, process_local=True)
                corrupt(run_dir / "resolved" / "training-state-source.lock.json")
                run = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
                _write_state(run_dir / "outputs" / "derived-step0150-state")
                checks = {
                    "status": lambda: _materialize_stdout_progress(run_dir, {}, state="running"),
                    "plan": lambda: _resume_plan_payload(root, run, run_dir),
                    "publish": lambda: publish_completed_training_states(root, run_dir),
                }
                for site, check in checks.items():
                    with self.assertRaisesRegex(ValueError, "training-state source lock", msg=site):
                        check()
                observed = [
                    json.loads(path.read_text(encoding="utf-8")).get("observed_step")
                    for path in (root / "artifacts" / "training-state").glob("*/manifest.json")
                ]
                self.assertNotIn(150, observed)


class SdScriptsLogicalResumeTests(unittest.TestCase):
    """The pinned sd-scripts image continues the logical step on Resume
    (docker/sd-scripts/patches/0002-resume-continues-the-logical-step.patch): its step
    counter, checkpoint and state names, save cadence, stop, and progress bar all count
    logical steps, also when the source step is not at an epoch boundary."""

    def _run_dir(self, root: Path) -> tuple[dict[str, Any], Path]:
        # Six items train six steps per epoch; Resume 15 -> 32 starts in the third epoch
        # and stops in the sixth, which is not a multiple of the steps per epoch.
        source = sd_scripts_run()
        source["recipe"]["steps"] = 15
        source["backend"]["config"]["save_every_n_steps"] = 5
        source["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
        manifest = publish_source(root, source, step=15)
        run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"], source_step=15, additional=17)
        run_dir = root / "runs" / "derived"
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        compile_resume_lock(root, run, run_dir / "resolved")
        return run, run_dir

    def _progress(self, run_dir: Path, line: str) -> dict[str, Any]:
        (run_dir / "logs" / "stdout.log").write_text(line + "\n", encoding="utf-8")
        status: dict[str, Any] = {}
        _materialize_stdout_progress(run_dir, status, state="running")
        return status

    def test_a_multi_epoch_resume_saves_on_logical_multiples_and_stops_at_its_target(self) -> None:
        from kura.training_artifacts import peak_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, run_dir = self._run_dir(root)
            script = command_sd_scripts(run)["argv"][2]
            # The trainer is given the logical target and the configured cadence, uncapped.
            self.assertIn("--max_train_steps 32", script)
            self.assertIn("--save_every_n_steps 5", script)
            self.assertIn("--save_last_n_steps_state 5", script)
            self.assertIn("--skip_until_initial_step", script)
            steps = resume_steps(run)
            self.assertEqual(
                (steps["native_progress"], steps["native_start"], steps["native_end"]), ("logical", 15, 32),
            )
            # Saves at logical steps 20, 25, and 30, then the final weights at 32.
            self.assertEqual(peak_checkpoints(run, {"save_every_n_steps": 5}), {"count": 4, "trainer_default_saves": False})
            self.assertEqual([logical_step(step, steps) for step in (20, 25, 30)], [20, 25, 30])
            lock = json.loads((run_dir / "resolved" / "training-state-source.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["native_progress"], "logical")
            # The patched progress bar counts logical steps toward the logical target.
            status = self._progress(run_dir, "steps:  62%|██████    | 20/32 [00:10<00:06, 2.0it/s, avr_loss=0.5]")
            self.assertEqual((status["last_step"], status["total_steps"]), (20, 32))
            self.assertEqual((status["current_run_step"], status["current_run_total_steps"]), (5, 17))

    def test_a_run_compiled_under_the_process_local_contract_is_read_as_before(self) -> None:
        # A lock frozen before the patched image counted the trainer's progress from zero.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, run_dir = self._run_dir(root)
            lock_path = run_dir / "resolved" / "training-state-source.lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["native_progress"] = "process_local"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            steps = resume_steps(run, lock=lock)
            self.assertEqual((steps["native_progress"], steps["native_end"]), ("process_local", 32))
            self.assertEqual(logical_step(5, steps), 20)
            status = self._progress(run_dir, "steps:  29%|███       | 5/17 [00:10<00:24, 2.0it/s, avr_loss=0.5]")
            self.assertEqual((status["last_step"], status["total_steps"]), (20, 32))
            self.assertEqual((status["current_run_step"], status["current_run_total_steps"]), (5, 17))


class AccelerateStateRunnerTests(unittest.TestCase):
    """sd-scripts and Musubi Tuner run through one Accelerate state runner that marks each
    complete save with the step it holds; every state is published at its marker's step."""

    def test_both_accelerate_backends_launch_their_trainer_through_the_one_runner(self) -> None:
        from kura.backends.musubi_command import compile_musubi_tuner
        from kura.backends.sd_scripts import compile_sd_scripts
        from kura.container_scripts import script_source

        cases = (
            (musubi_run, compile_musubi_tuner, "musubi", "src/musubi_tuner/flux_2_train_network.py"),
            (sd_scripts_run, compile_sd_scripts, "sd-scripts", "train_network.py"),
        )
        for build, compile_backend, directory_name, entrypoint in cases:
            run = build()
            name = run["backend"]["name"]
            with self.subTest(backend=name), tempfile.TemporaryDirectory() as directory:
                resolved = Path(directory) / "resolved"
                freeze_fixture(run, resolved)
                script = compile_backend(run, resolved / directory_name)["argv"][2]
                runner = f"/workspace/runs/derived/resolved/{directory_name}/state-runner.py {name} {entrypoint} "
                self.assertIn(runner, script)
                self.assertEqual(
                    (resolved / directory_name / "state-runner.py").read_text(encoding="utf-8"),
                    script_source("accelerate_state.py") + "\n",
                )
                unmanaged = build()
                unmanaged["recovery"] = {"training_state": {"enabled": False}}
                unmanaged_resolved = Path(directory) / "unmanaged"
                freeze_fixture(unmanaged, unmanaged_resolved)
                script = compile_backend(unmanaged, unmanaged_resolved / directory_name)["argv"][2]
                self.assertNotIn("state-runner.py", script)
                self.assertIn(f" {entrypoint} ", script)
                self.assertFalse((unmanaged_resolved / directory_name / "state-runner.py").exists())

    def _save(self, backend: str, scheduler: dict[str, Any], optimizer_step: int) -> tuple[Path, dict[str, Any]]:
        from kura.container_scripts import script_source

        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("accelerate_state.py"), namespace)

        class Accelerator:
            def save_state(self, output_dir):
                output = Path(output_dir)
                output.mkdir(parents=True)
                for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                    (output / name).write_bytes(_state_bytes(name))
                if backend == "sd-scripts":
                    (output / "train_state.json").write_text('{"current_epoch":1,"current_step":1}\n', encoding="utf-8")
                return "saved"

        torch = __import__("types").ModuleType("torch")
        torch.load = lambda path, **kwargs: (
            scheduler if Path(path).name == "scheduler.bin" else {"state": {0: {"step": optimizer_step}}}
        )
        accelerate = __import__("types").ModuleType("accelerate")
        accelerate.Accelerator = Accelerator
        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root)
        output = root / "derived-state"
        with patch.dict(sys.modules, {"torch": torch, "accelerate": accelerate}):
            namespace["install_hooks"](backend)
            self.assertEqual(Accelerator().save_state(output), "saved")
        return output, json.loads((output / "kura-state-info.json").read_text(encoding="utf-8"))

    def test_the_runner_marks_a_musubi_save_with_its_scheduler_step_and_no_train_state(self) -> None:
        output, info = self._save("musubi-tuner", {"last_epoch": 1150, "_step_count": 1151}, 1150)
        self.assertEqual(info["backend"], "musubi-tuner")
        self.assertEqual(info["logical_step"], 1150)
        self.assertNotIn("train_state_sha256", info)
        self.assertFalse((output / "train_state.json").exists())
        _, info = self._save("sd-scripts", {"last_epoch": 7, "_step_count": 8}, 7)
        self.assertEqual((info["backend"], info["logical_step"]), ("sd-scripts", 7))
        self.assertIn("train_state_sha256", info)

    def test_the_runner_refuses_a_save_whose_optimizer_and_scheduler_disagree(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "optimizer step 1149 does not match scheduler step 1150"):
            self._save("musubi-tuner", {"last_epoch": 1150, "_step_count": 1151}, 1149)

    def _musubi_run_dir(self, root: Path, run: dict[str, Any]) -> Path:
        run_dir = root / "runs" / "derived"
        (run_dir / "resolved" / "musubi").mkdir(parents=True)
        (run_dir / "resolved" / "musubi" / "state-runner.py").write_text("runner\n", encoding="utf-8")
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        return run_dir

    def test_musubi_final_state_is_published_at_the_step_its_marker_records(self) -> None:
        # `<name>-state` carries no step in its name; before the marker it was never published.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = musubi_run()
            run["recipe"]["steps"] = 20
            run_dir = self._musubi_run_dir(root, run)
            final = _write_state(run_dir / "outputs" / "derived-state")
            self.assertEqual(publish_completed_training_states(root, run_dir, allow_final_state=True), [])
            _write_state_marker(final, "musubi-tuner", 20)
            self.assertEqual(publish_completed_training_states(root, run_dir), [])
            published = publish_completed_training_states(root, run_dir, allow_final_state=True)
            self.assertEqual([item["observed_step"] for item in published], [20])
            self.assertIn("kura-state-info.json", {item["path"] for item in published[0]["files"]})

    def test_a_musubi_artifact_published_before_the_marker_still_resumes(self) -> None:
        # Resume reads an artifact's own inventory, never the contract's required files.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = musubi_run()
            manifest = publish_source(root, source)
            self.assertNotIn("kura-state-info.json", {item["path"] for item in manifest["files"]})
            run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"])
            lock = compile_resume_lock(root, run, root / "runs" / "derived" / "resolved")
            self.assertEqual({item["path"] for item in lock["files"]}, set(STATE_FILES))
            script = command_musubi_tuner(run)["argv"][2]
            self.assertIn(f"--resume {training_state_payload(manifest['id'])}", script)
            self.assertIn("training-state-source.lock.json", script)


class ResumeSourceVerificationTests(unittest.TestCase):
    def _workspace(self, root: Path, *, mutate) -> tuple[dict[str, Any], Path]:
        source = sd_scripts_run()
        manifest = publish_source(root, source)
        run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"])
        mutate(run)
        run_dir = root / "runs" / "derived"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "realizations").mkdir()
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
        return run, run_dir

    def _errors(self, root: Path, run: dict[str, Any], run_dir: Path) -> dict[str, str]:
        checks = {
            "plan": lambda: _resume_plan_payload(root, run, run_dir),
            "compile": lambda: compile_resume_lock(root, run, root / "compile-out"),
            "runpod-stage": lambda: stage_runpod(
                workspace=root, run_dir=run_dir, dataset_ids=[],
                config={"runpod": {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"]}},
            ),
            "artifact-directory": lambda: resume_artifact_directory(root, run),
            "dataset-transfer": lambda: _resume_entries(root, run, verify=True),
        }
        errors: dict[str, str] = {}
        for site, check in checks.items():
            with self.assertRaises(ValueError, msg=site) as raised:
                check()
            errors[site] = str(raised.exception)
        return errors

    def test_a_digest_mismatch_is_the_same_error_wherever_the_source_is_read(self) -> None:
        def mutate(run: dict[str, Any]) -> None:
            run["continuation"]["source"]["manifest_sha256"] = "f" * 64

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, run_dir = self._workspace(root, mutate=mutate)
            errors = self._errors(root, run, run_dir)
        artifact_id = run["continuation"]["source"]["artifact_id"]
        self.assertEqual(set(errors.values()), {f"training-state manifest digest mismatch: {artifact_id}"}, errors)

    def test_a_malformed_continuation_is_refused_with_one_message_everywhere(self) -> None:
        def mutate(run: dict[str, Any]) -> None:
            run["continuation"]["target_step"] = 1300

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, run_dir = self._workspace(root, mutate=mutate)
            errors = self._errors(root, run, run_dir)
        self.assertEqual(set(errors.values()), {"continuation.target_step does not match the requested Resume target"}, errors)


def _plan_workspace(root: Path) -> None:
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    (root / "datasets" / "tiny").mkdir(parents=True)


def _plannable_musubi_run() -> dict[str, Any]:
    run = musubi_run()
    run["id"] = "derived"
    run["experiment"] = "steps"
    run["compute"] = {"executor": "docker"}
    run["backend"]["config"] = {
        "architecture": "flux2", "model_bundle": "none", "save_every_n_steps": 500,
        "model_downloads": {"dit": {"repo": "repo/model", "filename": "weights.safetensors"}},
    }
    return run


class ResumeStepsDisplayTests(unittest.TestCase):
    def test_a_resume_shows_the_logical_step_it_reaches_in_plan_experiment_and_monitor(self) -> None:
        # Resume +50 from step 1000 of a 1000-step recipe: every reader shows 1050.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _plan_workspace(root)
            source = _plannable_musubi_run()
            manifest = publish_source(root, source)
            run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"], additional=50)
            run_dir = root / "runs" / "derived"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 200}):
                    payload = plan.plan_run("derived")
                text = plan.format_run_plan(payload)
            finally:
                os.chdir(previous)
            facts = experiment._display_mapping(run_dir, run)
            key_config = monitor._key_config("train", run, run_dir)
        self.assertEqual(payload["recipe"]["steps"], 1050)
        self.assertRegex(text, r"\nRecipe\n  steps +1050\n")
        self.assertEqual(payload["experiment"]["runs"][0]["facts"]["steps"], 1050)
        self.assertEqual(facts["steps"], 1050)
        self.assertEqual(key_config["steps"], 1050)

    def test_experiment_and_monitor_read_one_display_rule(self) -> None:
        import kura.training_artifacts as training_artifacts

        run = as_resume(musubi_run(), additional=50)
        broken = deepcopy(run)
        broken["continuation"]["target_step"] = 1
        for module, call in (
            (experiment, lambda item: experiment._display_mapping(Path("missing"), item).get("steps")),
            (monitor, lambda item: monitor._key_config("train", item, Path("missing"))["steps"]),
        ):
            with self.subTest(module=module.__name__), patch.object(
                module, "displayed_final_step", wraps=training_artifacts.displayed_final_step,
            ) as owner:
                self.assertEqual(call(run), 1050)
                self.assertIsNone(call(broken))
                owner.assert_called()


if __name__ == "__main__":
    unittest.main()
