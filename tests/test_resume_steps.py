"""Resume step arithmetic has one owner, and every backend, executor, and reader agrees with it."""

from __future__ import annotations

import json
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
from tests.test_training_resume import _safetensors_bytes, _torch_archive_bytes  # noqa: E402


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
        return int(re.search(r'"--max_train_steps","(\d+)"', command_sd_scripts(run)["argv"][2]).group(1))
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
        # Resume +200 from step 1000: the trainer's own target is in its native step space.
        expected = {"musubi-tuner": (0, 200), "sd-scripts": (1000, 1200), "ai-toolkit": (1000, 1200)}
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
    def _derived(self, root: Path) -> Path:
        source = musubi_run()
        manifest = publish_source(root, source)
        run = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"])
        run_dir = root / "runs" / "derived"
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        compile_resume_lock(root, run, run_dir / "resolved")
        (run_dir / "logs" / "stdout.log").write_text("steps:  75%|███| 150/200 [00:10<00:04, 1.0s/it, avr_loss=0.5]\n", encoding="utf-8")
        return run_dir

    def test_docker_and_runpod_place_process_local_progress_and_state_at_one_logical_step(self) -> None:
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
                    _write_state(run_dir / "outputs" / name)
                    items = publish_completed_training_states(root, run_dir)
                else:
                    item = {
                        "path": f"/workspace/runs/derived/outputs/{name}",
                        "name": name,
                        # The Pod reports the step the state's marker records.
                        "marked_step": 1150,
                        "files": [{"path": file, "size": len(_state_bytes(file)), "mtime_ns": 1} for file in STATE_FILES],
                    }

                    def fake_scp(command: list[str], **_: object) -> subprocess.CompletedProcess:
                        _write_state(Path(command[-1]))
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
        def non_int_steps(lock_path: Path) -> None:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["source_step"] = "1000"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

        def broken_json(lock_path: Path) -> None:
            lock_path.write_text("{not json", encoding="utf-8")

        for corrupt in (non_int_steps, broken_json):
            with self.subTest(corrupt=corrupt.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = self._derived(root)
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


if __name__ == "__main__":
    unittest.main()
