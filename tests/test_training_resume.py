from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.cli import cmd_run_resume
from kura.backends.ai_toolkit import compile_ai_toolkit, command_ai_toolkit, training_state_contract_ai_toolkit
from kura.backends.sd_scripts import training_state_contract_sd_scripts
from kura.container_scripts import script_source
from kura.executors.docker import reconcile_docker
from kura.executors.common import _materialize_stdout_progress
from kura.executors.runpod import stage_runpod
from kura.run_commands.runpod_ssh import _download_run_unlocked, _local_reusable_training_state_sources, _pull_remote_training_state_items, _same_remote_training_state_version
from kura.run_commands.plan import format_run_plan
from kura.run_envelope import resume_intent, training_state_policy
from kura.training_artifacts import compile_resume_lock, load_training_state, publish_completed_training_states, publish_training_state, recipe_fingerprint, select_training_state, verify_training_state


def _safetensors_bytes(content: bytes) -> bytes:
    header = json.dumps({"value": {"dtype": "U8", "shape": [len(content)], "data_offsets": [0, len(content)]}}, separators=(",", ":")).encode()
    return len(header).to_bytes(8, "little") + header + content


def _torch_archive_bytes(content: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"state": content}, protocol=4))
        archive.writestr("archive/version", "3\n")
        archive.writestr("archive/.data/serialization_id", "0")
    return output.getvalue()


def _write_state_marker(candidate: Path, backend: str, logical_step: int) -> None:
    if backend == "ai-toolkit":
        document = {
            "schema_version": 1,
            "backend": backend,
            "logical_step": logical_step,
            "weight_sha256": hashlib.sha256((candidate / "model.safetensors").read_bytes()).hexdigest(),
            "optimizer_sha256": hashlib.sha256((candidate / "optimizer.pt").read_bytes()).hexdigest(),
            "rng_sha256": hashlib.sha256((candidate / "rng.pt").read_bytes()).hexdigest(),
        }
        name = "state-info.json"
    elif backend == "sd-scripts":
        document = {
            "schema_version": 1,
            "backend": backend,
            "logical_step": logical_step,
            "train_state_sha256": hashlib.sha256((candidate / "train_state.json").read_bytes()).hexdigest(),
            "optimizer_sha256": hashlib.sha256((candidate / "optimizer.bin").read_bytes()).hexdigest(),
            "scheduler_sha256": hashlib.sha256((candidate / "scheduler.bin").read_bytes()).hexdigest(),
        }
        name = "kura-state-info.json"
    else:
        raise AssertionError(backend)
    (candidate / name).write_text(json.dumps(document) + "\n", encoding="utf-8")


class TrainingStateArtifactTests(unittest.TestCase):
    def test_sd_scripts_resume_contract_uses_normalized_selectors(self) -> None:
        for architecture in ("flux", "SDXL", "sd-1.5"):
            with self.subTest(architecture=architecture):
                contract = training_state_contract_sd_scripts(
                    {"backend": {"name": "sd-scripts", "config": {"architecture": architecture, "mode": "LORA"}}}
                )
                self.assertEqual(contract["capability"], "best_effort_resume")
                self.assertIn("kura-state-info.json", contract["required_files"])

        anima = training_state_contract_sd_scripts(
            {"backend": {"name": "sd-scripts", "config": {"architecture": "Anima", "mode": "LORA"}}}
        )
        self.assertEqual(anima["capability"], "unsupported")

    def test_ai_toolkit_resume_contract_requires_and_restores_rng_state(self) -> None:
        contract = training_state_contract_ai_toolkit(
            {"backend": {"name": "ai-toolkit", "config": {"optimizer_type": "adamw"}}}
        )

        self.assertIn("rng.pt", contract["required_files"])
        self.assertIn("rng_state_at_pre_iterator_hook", contract["restoration_contract"]["restored"])
        self.assertIn("exact_rng_position", contract["restoration_contract"]["not_restored"])
        self.assertEqual(contract["state_step"]["digests"]["rng_sha256"], "rng.pt")

    def test_ai_toolkit_current_resume_contract_requires_rng_at_runtime(self) -> None:
        run = {
            "id": "derived",
            "parent_run": "source",
            "backend": {"name": "ai-toolkit", "config": {"optimizer_type": "adamw"}},
            "recipe": {"steps": 100, "seed": 1},
        }
        contract = training_state_contract_ai_toolkit(run)["restoration_contract"]
        run["continuation"] = {
            "mode": "resume",
            "source": {
                "artifact_id": "state-1",
                "manifest_sha256": "a" * 64,
                "observed_step": 50,
                "recipe_sha256": "b" * 64,
            },
            "additional_steps": 50,
            "target_step": 100,
            "restoration_contract": contract,
        }

        command = command_ai_toolkit(run)

        self.assertIn('"rng_required":true', " ".join(command["argv"]))

    def test_ai_toolkit_disabled_capture_uses_the_native_runner(self) -> None:
        run = {
            "id": "source",
            "backend": {"name": "ai-toolkit", "config": {}},
            "recipe": {"steps": 10, "seed": 1},
            "recovery": {"training_state": {"enabled": False, "keep_generations": 1}},
        }
        command = command_ai_toolkit(run)
        self.assertEqual(command["argv"], ["python", "run.py", "/workspace/runs/source/resolved/ai-toolkit.yaml"])
        self.assertEqual(command["env"], {"SEED": "1"})

    def test_ai_toolkit_accumulated_training_does_not_capture_mislabeled_optimizer_updates(self) -> None:
        run = {
            "id": "source",
            "backend": {"name": "ai-toolkit", "config": {"gradient_accumulation_steps": 2}},
            "recipe": {"steps": 10, "seed": 1},
        }

        command = command_ai_toolkit(run)

        self.assertEqual(command["argv"], ["python", "run.py", "/workspace/runs/source/resolved/ai-toolkit.yaml"])

    def test_ai_toolkit_unsupported_capture_contract_does_not_reject_native_ema_training(self) -> None:
        run = {
            "id": "source",
            "backend": {
                "name": "ai-toolkit",
                "config": {
                    "gradient_accumulation_steps": 2,
                    "native_config": {"ema_config": {"use_ema": True}},
                },
            },
            "model": {"base": "example/model"},
            "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 10, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            command = compile_ai_toolkit(run, Path(directory) / "ai-toolkit")

        self.assertEqual(command["argv"], ["python", "run.py", "/workspace/runs/source/resolved/ai-toolkit.yaml"])

    def test_ai_toolkit_supported_capture_contract_still_rejects_ema(self) -> None:
        run = {
            "id": "source",
            "backend": {
                "name": "ai-toolkit",
                "config": {"native_config": {"ema_config": {"use_ema": True}}},
            },
            "model": {"base": "example/model"},
            "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 10, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not support.*EMA"):
                compile_ai_toolkit(run, Path(directory) / "ai-toolkit")

    def test_ai_toolkit_optimizer_without_a_verified_update_counter_does_not_claim_resume(self) -> None:
        run = {
            "id": "source",
            "backend": {"name": "ai-toolkit", "config": {"optimizer_type": "sgd"}},
            "recipe": {"steps": 10, "seed": 1},
        }

        command = command_ai_toolkit(run)

        self.assertEqual(command["argv"], ["python", "run.py", "/workspace/runs/source/resolved/ai-toolkit.yaml"])

    def test_ai_toolkit_runner_publishes_weight_optimizer_pair_atomically(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_root = root / "outputs" / "source"
            save_root.mkdir(parents=True)
            (save_root / "source_000000010.safetensors").write_bytes(b"weight")

            torch = types.ModuleType("torch")
            torch.float32 = "torch.float32"
            torch.save = lambda value, path: Path(path).write_text(json.dumps(value), encoding="utf-8")
            torch.load = lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8"))
            toolkit = types.ModuleType("toolkit")
            metadata = types.ModuleType("toolkit.metadata")
            metadata.load_metadata_from_safetensors = lambda path: {"training_info": {"step": 10}}
            process = types.SimpleNamespace(
                accelerator=types.SimpleNamespace(is_main_process=True),
                step_num=10,
                save_root=str(save_root),
                job=types.SimpleNamespace(name="source"),
                network=types.SimpleNamespace(
                    save_weights=lambda path, **kwargs: Path(path).write_bytes(
                        (save_root / "source_000000010.safetensors").read_bytes()
                    )
                ),
                optimizer=types.SimpleNamespace(state_dict=lambda: {"state": {0: {"step": 10}}}),
            )
            namespace["weight_metadata_with_logical_step"] = lambda source, step: {"training_info": json.dumps({"step": step})}
            namespace["safetensors_tensor_dtypes"] = lambda path: {"F32"}
            namespace["capture_rng_state"] = lambda torch: {"schema_version": 1}
            with patch.dict(sys.modules, {"torch": torch, "toolkit": toolkit, "toolkit.metadata": metadata}):
                namespace["publish_generation"](
                    process,
                    10,
                    {"run_id": "source", "state_root": str(root / "outputs"), "keep_generations": 2},
                )
            state = root / "outputs" / "source-step00000010-state"
            self.assertTrue((state / "model.safetensors").is_file())
            self.assertTrue((state / "optimizer.pt").is_file())
            self.assertEqual(json.loads((state / "state-info.json").read_text())["logical_step"], 10)
            self.assertFalse(any((root / "outputs").glob(".source-step*.partial")))

    def test_ai_toolkit_runner_captures_live_network_as_float32_resume_weight(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_root = root / "outputs" / "source"
            save_root.mkdir(parents=True)
            regular_weight = save_root / "source_000000010.safetensors"
            regular_weight.write_bytes(b"distribution-fp16")

            torch = types.ModuleType("torch")
            torch.float32 = "torch.float32"
            torch.save = lambda value, path: Path(path).write_text(json.dumps(value), encoding="utf-8")
            torch.load = lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8"))
            toolkit = types.ModuleType("toolkit")
            metadata = types.ModuleType("toolkit.metadata")
            metadata.load_metadata_from_safetensors = lambda path: {"training_info": {"step": 10}}
            saved: dict[str, object] = {}

            def save_weights(path, *, dtype, metadata):
                saved.update(dtype=dtype, metadata=metadata)
                Path(path).write_bytes(b"resume-fp32")

            process = types.SimpleNamespace(
                accelerator=types.SimpleNamespace(is_main_process=True),
                save_root=str(save_root),
                job=types.SimpleNamespace(name="source"),
                network=types.SimpleNamespace(save_weights=save_weights),
                optimizer=types.SimpleNamespace(state_dict=lambda: {"state": {0: {"step": 10}}}),
            )
            namespace["weight_metadata_with_logical_step"] = lambda source, step: {"training_info": json.dumps({"step": step})}
            namespace["safetensors_tensor_dtypes"] = lambda path: {"F32"}
            namespace["capture_rng_state"] = lambda torch: {"schema_version": 1}
            with patch.dict(sys.modules, {"torch": torch, "toolkit": toolkit, "toolkit.metadata": metadata}):
                namespace["publish_generation"](
                    process,
                    10,
                    {"run_id": "source", "state_root": str(root / "outputs"), "keep_generations": 2},
                )

            state = root / "outputs" / "source-step00000010-state"
            self.assertEqual((state / "model.safetensors").read_bytes(), b"resume-fp32")
            self.assertEqual(regular_weight.read_bytes(), b"distribution-fp16")
            self.assertEqual(saved["dtype"], "torch.float32")
            self.assertEqual(json.loads(saved["metadata"]["training_info"])["step"], 10)
            self.assertEqual(json.loads((state / "state-info.json").read_text())["model_dtype"], "float32")

    def test_ai_toolkit_runner_coalesces_semantically_equal_periodic_and_final_state(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_root = root / "outputs" / "source"
            save_root.mkdir(parents=True)
            for name in ("source_000000001.safetensors", "source.safetensors"):
                (save_root / name).write_text("same-weight", encoding="utf-8")
            saves = 0

            def save(value, path):
                nonlocal saves
                saves += 1
                Path(path).write_text(json.dumps({"nonce": saves, "value": value}), encoding="utf-8")

            torch = types.ModuleType("torch")
            torch.float32 = "torch.float32"
            torch.save = save
            torch.load = lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8"))["value"]
            process = types.SimpleNamespace(
                accelerator=types.SimpleNamespace(is_main_process=True),
                step_num=1,
                save_root=str(save_root),
                job=types.SimpleNamespace(name="source"),
                network=types.SimpleNamespace(
                    save_weights=lambda path, **kwargs: Path(path).write_text("same-weight-logical-2", encoding="utf-8")
                ),
                optimizer=types.SimpleNamespace(state_dict=lambda: {"state": {0: {"step": 2, "moment": [1, 2]}}}),
            )
            namespace["weight_metadata_with_logical_step"] = lambda source, step: {"training_info": json.dumps({"step": step})}
            namespace["safetensors_tensor_dtypes"] = lambda path: {"F32"}
            namespace["capture_rng_state"] = lambda torch: {"schema_version": 1}
            namespace["metadata_step"] = lambda path: int(Path(path).read_text().rsplit("-", 1)[1])

            with patch.dict(sys.modules, {"torch": torch}):
                namespace["publish_generation"](
                    process, 1, {"run_id": "source", "state_root": str(root / "outputs"), "keep_generations": 2}
                )
                namespace["publish_generation"](
                    process, None, {"run_id": "source", "state_root": str(root / "outputs"), "keep_generations": 2}
                )
                process.optimizer.state_dict = lambda: {"state": {0: {"step": 2, "moment": [9, 9]}}}
                with self.assertRaisesRegex(RuntimeError, "conflicting training-state generation"):
                    namespace["publish_generation"](
                        process, None, {"run_id": "source", "state_root": str(root / "outputs"), "keep_generations": 2}
                    )

            states = list((root / "outputs").glob("source-step*-state"))
            self.assertEqual([path.name for path in states], ["source-step00000002-state"])

    def test_ai_toolkit_runner_does_not_coalesce_states_with_different_rng(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            staged = root / "staged"
            existing.mkdir()
            staged.mkdir()
            for state in (existing, staged):
                (state / "model.safetensors").write_bytes(b"same-weight")
                (state / "optimizer.pt").write_text('{"step": 2}', encoding="utf-8")
            (existing / "rng.pt").write_text('{"position": 10}', encoding="utf-8")
            (staged / "rng.pt").write_text('{"position": 11}', encoding="utf-8")
            torch = types.ModuleType("torch")
            torch.is_tensor = lambda value: False
            torch.load = lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8"))

            with patch.dict(sys.modules, {"torch": torch}):
                self.assertFalse(namespace["saved_states_equivalent"](existing, staged))

    def test_ai_toolkit_runner_uses_completed_optimizer_updates_instead_of_stale_process_step(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_root = root / "outputs" / "source"
            save_root.mkdir(parents=True)
            weight = save_root / "source_000000001.safetensors"
            weight.write_text("native-step-1", encoding="utf-8")

            torch = types.ModuleType("torch")
            torch.float32 = "torch.float32"
            torch.save = lambda value, path: Path(path).write_text(json.dumps(value), encoding="utf-8")
            torch.load = lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8"))
            process = types.SimpleNamespace(
                accelerator=types.SimpleNamespace(is_main_process=True),
                step_num=1,
                save_root=str(save_root),
                job=types.SimpleNamespace(name="source"),
                network=types.SimpleNamespace(
                    save_weights=lambda path, **kwargs: Path(path).write_text("logical-step-2", encoding="utf-8")
                ),
                optimizer=types.SimpleNamespace(state_dict=lambda: {"state": {0: {"step": 2}, 1: {"step": 2}}}),
            )
            namespace["weight_metadata_with_logical_step"] = lambda source, step: {"training_info": json.dumps({"step": step})}
            namespace["safetensors_tensor_dtypes"] = lambda path: {"F32"}
            namespace["capture_rng_state"] = lambda torch: {"schema_version": 1}
            namespace["metadata_step"] = lambda path: int(Path(path).read_text(encoding="utf-8").rsplit("-", 1)[1])

            with patch.dict(sys.modules, {"torch": torch}):
                namespace["publish_generation"](
                    process,
                    1,
                    {"run_id": "source", "state_root": str(root / "outputs"), "keep_generations": 2},
                )

            state = root / "outputs" / "source-step00000002-state"
            self.assertTrue(state.is_dir())
            self.assertFalse((root / "outputs" / "source-step00000001-state").exists())
            self.assertEqual(json.loads((state / "state-info.json").read_text())["logical_step"], 2)

    def test_ai_toolkit_runner_hard_fails_optimizer_before_original_train_hook(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)

        class Base:
            def load_weights(self, path):
                return path

            def save(self, step=None):
                return step

        class Trainer(Base):
            def hook_before_train_loop(self):
                self.original_hook_called = True

        torch = types.ModuleType("torch")
        torch.load = lambda path, **kwargs: {"state": {}}
        modules = {
            "torch": torch,
            "extensions_built_in": types.ModuleType("extensions_built_in"),
            "extensions_built_in.sd_trainer": types.ModuleType("extensions_built_in.sd_trainer"),
            "extensions_built_in.sd_trainer.SDTrainer": types.ModuleType("extensions_built_in.sd_trainer.SDTrainer"),
            "jobs": types.ModuleType("jobs"),
            "jobs.process": types.ModuleType("jobs.process"),
            "jobs.process.BaseSDTrainProcess": types.ModuleType("jobs.process.BaseSDTrainProcess"),
        }
        modules["extensions_built_in.sd_trainer.SDTrainer"].SDTrainer = Trainer
        modules["jobs.process.BaseSDTrainProcess"].BaseSDTrainProcess = Base
        expected_weight = Path("/tmp/expected.safetensors")
        with patch.dict(sys.modules, modules):
            namespace["install_hooks"](
                {"resume": {"source_step": 10}}, expected_weight
            )
        process = types.SimpleNamespace(
            _kura_loaded_weight=expected_weight,
            step_num=10,
            start_step=10,
            network=types.SimpleNamespace(did_change_weights=False),
            save_root="/tmp",
            optimizer=types.SimpleNamespace(
                param_groups=[{"lr": 0.1}],
                load_state_dict=lambda state: (_ for _ in ()).throw(ValueError("incompatible")),
            ),
            original_hook_called=False,
        )
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "before the first update"):
                Trainer.hook_before_train_loop(process)
        self.assertFalse(process.original_hook_called)

    def test_ai_toolkit_runner_restores_rng_after_original_before_loop(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        events: list[str] = []

        class Base:
            def load_weights(self, path):
                return path

            def save(self, step=None):
                return step

        class Trainer(Base):
            def hook_before_train_loop(self):
                events.append("original")
                return "prepared"

        torch = types.ModuleType("torch")
        torch.load = lambda path, **kwargs: {"state": {0: {"step": 50}}}
        modules = {
            "torch": torch,
            "extensions_built_in": types.ModuleType("extensions_built_in"),
            "extensions_built_in.sd_trainer": types.ModuleType("extensions_built_in.sd_trainer"),
            "extensions_built_in.sd_trainer.SDTrainer": types.ModuleType("extensions_built_in.sd_trainer.SDTrainer"),
            "jobs": types.ModuleType("jobs"),
            "jobs.process": types.ModuleType("jobs.process"),
            "jobs.process.BaseSDTrainProcess": types.ModuleType("jobs.process.BaseSDTrainProcess"),
        }
        modules["extensions_built_in.sd_trainer.SDTrainer"].SDTrainer = Trainer
        modules["jobs.process.BaseSDTrainProcess"].BaseSDTrainProcess = Base
        expected_weight = Path("/tmp/expected.safetensors")
        rng_directory = tempfile.TemporaryDirectory()
        self.addCleanup(rng_directory.cleanup)
        rng_path = Path(rng_directory.name) / "rng.pt"
        rng_path.write_bytes(b"rng")
        namespace["restore_rng_state"] = lambda path, torch: events.append("rng")
        with patch.dict(sys.modules, modules):
            namespace["install_hooks"](
                {"resume": {"source_step": 50, "payload": rng_directory.name, "rng_required": True}}, expected_weight
            )
        process = types.SimpleNamespace(
            _kura_loaded_weight=expected_weight,
            step_num=50,
            start_step=50,
            network=types.SimpleNamespace(did_change_weights=False),
            save_root="/tmp",
            optimizer=types.SimpleNamespace(
                param_groups=[{"lr": 1.0e-6, "initial_lr": 1.0e-6}],
                load_state_dict=lambda state: None,
            ),
        )

        with patch.dict(sys.modules, {"torch": torch}):
            result = Trainer.hook_before_train_loop(process)

        self.assertEqual(result, "prepared")
        self.assertEqual(events, ["original", "rng"])

    def test_ai_toolkit_runner_rejects_missing_required_rng_state(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)

        class Base:
            def load_weights(self, path):
                return path

            def save(self, step=None):
                return step

        class Trainer(Base):
            def hook_before_train_loop(self):
                return "prepared"

        torch = types.ModuleType("torch")
        torch.load = lambda path, **kwargs: {"state": {0: {"step": 50}}}
        modules = {
            "torch": torch,
            "extensions_built_in": types.ModuleType("extensions_built_in"),
            "extensions_built_in.sd_trainer": types.ModuleType("extensions_built_in.sd_trainer"),
            "extensions_built_in.sd_trainer.SDTrainer": types.ModuleType("extensions_built_in.sd_trainer.SDTrainer"),
            "jobs": types.ModuleType("jobs"),
            "jobs.process": types.ModuleType("jobs.process"),
            "jobs.process.BaseSDTrainProcess": types.ModuleType("jobs.process.BaseSDTrainProcess"),
        }
        modules["extensions_built_in.sd_trainer.SDTrainer"].SDTrainer = Trainer
        modules["jobs.process.BaseSDTrainProcess"].BaseSDTrainProcess = Base
        expected_weight = Path("/tmp/expected.safetensors")
        rng_directory = tempfile.TemporaryDirectory()
        self.addCleanup(rng_directory.cleanup)
        with patch.dict(sys.modules, modules):
            namespace["install_hooks"](
                {"resume": {"source_step": 50, "payload": rng_directory.name, "rng_required": True}}, expected_weight
            )
        process = types.SimpleNamespace(
            _kura_loaded_weight=expected_weight,
            step_num=50,
            start_step=50,
            network=types.SimpleNamespace(did_change_weights=False),
            save_root="/tmp",
            optimizer=types.SimpleNamespace(
                param_groups=[{"lr": 1.0e-6, "initial_lr": 1.0e-6}],
                load_state_dict=lambda state: None,
            ),
        )

        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "required RNG state"):
                Trainer.hook_before_train_loop(process)

    def test_ai_toolkit_rng_runtime_error_uses_training_state_failure_context(self) -> None:
        import random

        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("ai_toolkit_state.py"), namespace)
        numpy = types.ModuleType("numpy")
        numpy.uint32 = "uint32"
        numpy.asarray = lambda value, dtype=None: value
        numpy.random = types.SimpleNamespace(set_state=lambda state: None)
        state = {
            "schema_version": 1,
            "python": random.getstate(),
            "numpy": {
                "bit_generator": "MT19937",
                "state": [1, 2, 3],
                "position": 0,
                "has_gauss": 0,
                "cached_gaussian": 0.0,
            },
            "torch_cpu": "bad-state",
            "torch_cuda": [],
        }
        torch = types.SimpleNamespace(
            load=lambda path, **kwargs: state,
            set_rng_state=lambda value: (_ for _ in ()).throw(RuntimeError("wrong state size")),
            cuda=types.SimpleNamespace(is_available=lambda: False),
        )

        with patch.dict(sys.modules, {"numpy": numpy}):
            with self.assertRaisesRegex(RuntimeError, "training-state failure: Resume RNG state is incomplete"):
                namespace["restore_rng_state"]("rng.pt", torch)

    def test_sd_scripts_runner_normalizes_saved_application_step_from_persisted_training_state(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("sd_scripts_state.py"), namespace)

        class Accelerator:
            def save_state(self, output_dir):
                output = Path(output_dir)
                output.mkdir(parents=True)
                (output / "train_state.json").write_text('{"current_epoch":3,"current_step":1}\n', encoding="utf-8")
                (output / "scheduler.bin").write_bytes(b"scheduler")
                (output / "optimizer.bin").write_bytes(b"optimizer")
                return "saved"

        torch = types.ModuleType("torch")
        torch.load = lambda path, **kwargs: (
            {"last_epoch": 3, "_step_count": 4}
            if Path(path).name == "scheduler.bin"
            else {"state": {0: {"step": 3}, 1: {"step": 3}}}
        )
        accelerate = types.ModuleType("accelerate")
        accelerate.Accelerator = Accelerator

        with patch.dict(sys.modules, {"torch": torch, "accelerate": accelerate}):
            namespace["install_hooks"]()
            output = Path(tempfile.mkdtemp()) / "state"
            try:
                self.assertEqual(Accelerator().save_state(output), "saved")
                self.assertEqual(json.loads((output / "train_state.json").read_text())["current_step"], 3)
                info = json.loads((output / "kura-state-info.json").read_text(encoding="utf-8"))
                self.assertEqual(info["logical_step"], 3)
            finally:
                if output.parent.is_dir():
                    __import__("shutil").rmtree(output.parent)

    def test_sd_scripts_save_wrapper_invalidates_a_stale_completion_marker_first(self) -> None:
        namespace: dict[str, object] = {"__name__": "container_test"}
        exec(script_source("sd_scripts_state.py"), namespace)
        observed: list[bool] = []

        class Accelerator:
            def save_state(self, output_dir):
                output = Path(output_dir)
                observed.append((output / "kura-state-info.json").exists())
                raise RuntimeError("interrupted native save")

        accelerate = types.ModuleType("accelerate")
        accelerate.Accelerator = Accelerator
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"accelerate": accelerate}):
            output = Path(directory) / "state"
            output.mkdir()
            (output / "kura-state-info.json").write_text('{"logical_step":2}\n', encoding="utf-8")
            namespace["install_hooks"]()
            with self.assertRaisesRegex(RuntimeError, "interrupted native save"):
                Accelerator().save_state(output)
            self.assertEqual(observed, [False])
            self.assertFalse((output / "kura-state-info.json").exists())

    def test_completed_output_projection_excludes_training_state_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "source"
            state = run_dir / "outputs" / "source-step00000010-state"
            state.mkdir(parents=True)
            (state / "optimizer.bin").write_bytes(b"optimizer")
            (run_dir / "outputs" / "source.safetensors").write_bytes(b"weight")
            status: dict[str, object] = {}
            _materialize_stdout_progress(run_dir, status, state="completed")
            self.assertEqual(status["outputs"], ["outputs/source.safetensors"])

    def test_musubi_resume_progress_projects_current_and_cumulative_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            manifest = {
                "id": "derived",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {}},
                "recipe": {"steps": 100, "seed": 1},
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {"artifact_id": "state-1", "observed_step": 100},
                    "target_step": 150,
                },
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (run_dir / "resolved" / "training-state-source.lock.json").write_text(
                json.dumps({"source_step": 100, "target_step": 150, "additional_steps": 50, "native_progress": "process_local"}),
                encoding="utf-8",
            )
            (run_dir / "logs" / "stdout.log").write_text("steps:  20%|██| 10/50 [00:10<00:40, 1.0s/it, avr_loss=0.5]\n", encoding="utf-8")
            status: dict[str, object] = {}
            _materialize_stdout_progress(run_dir, status, state="running")
            self.assertEqual(status["last_step"], 110)
            self.assertEqual(status["total_steps"], 150)
            self.assertEqual(status["current_run_step"], 10)
            self.assertEqual(status["current_run_total_steps"], 50)
            _materialize_stdout_progress(run_dir, status, state="running")
            self.assertEqual(status["last_step"], 110)

    def test_completed_resume_materializes_all_requested_additional_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved" / "training-state-source.lock.json").write_text(
                json.dumps(
                    {
                        "source_step": 2,
                        "target_step": 3,
                        "additional_steps": 1,
                        "native_progress": "logical",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "logs" / "stdout.log").write_text(
                "derived: 67%|###### | 2/3 [00:16<?, ?it/s, lr: 1.0e-06 loss: 5.2e-02]\n",
                encoding="utf-8",
            )
            status: dict[str, object] = {}

            _materialize_stdout_progress(run_dir, status, state="completed")

            self.assertEqual((status["last_step"], status["total_steps"]), (3, 3))
            self.assertEqual((status["current_run_step"], status["current_run_total_steps"]), (1, 1))

    def test_container_verifier_accepts_locked_payload_and_rejects_digest_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "artifacts" / "training-state" / "state-1" / "payload"
            payload.mkdir(parents=True)
            state = payload / "optimizer.bin"
            state.write_bytes(b"valid-state")
            lock = root / "runs" / "derived" / "resolved" / "training-state-source.lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text(
                json.dumps(
                    {
                        "artifact_id": "state-1",
                        "native_state_path": "/workspace/artifacts/training-state/state-1/payload",
                        "files": [
                            {
                                "path": "optimizer.bin",
                                "size": len(b"valid-state"),
                                "sha256": hashlib.sha256(b"valid-state").hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            command = [sys.executable, "-c", script_source("training_state_verify.py"), str(lock), str(root)]
            accepted = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("training state verified", accepted.stdout)

            state.write_bytes(b"evil!-state")
            rejected = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("digest mismatch", rejected.stderr)

    def test_remote_training_state_version_requires_identical_file_inventory(self) -> None:
        before = {
            "path": "/workspace/runs/source/outputs/source-step00000010-state",
            "files": [
                {"path": "optimizer.bin", "size": 10, "mtime_ns": 1},
                {"path": "scheduler.bin", "size": 5, "mtime_ns": 2},
            ],
        }
        self.assertTrue(_same_remote_training_state_version(before, json.loads(json.dumps(before))))
        changed = json.loads(json.dumps(before))
        changed["files"][0]["size"] = 11
        self.assertFalse(_same_remote_training_state_version(before, changed))
        added = json.loads(json.dumps(before))
        added["files"].append({"path": "random_states_0.pkl", "size": 3, "mtime_ns": 3})
        self.assertFalse(_same_remote_training_state_version(before, added))

    def test_remote_directory_is_published_only_after_stable_recursive_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "recovery").mkdir()
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            names = ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl")
            contents = {
                name: _safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode())
                for name in names
            }
            item = {
                "path": "/workspace/runs/source/outputs/source-step00000010-state",
                "name": "source-step00000010-state",
                "step": 10,
                "files": [{"path": name, "size": len(contents[name]), "mtime_ns": index} for index, name in enumerate(names, 1)],
            }

            def fake_scp(command: list[str], **_: object):
                destination = Path(command[-1])
                destination.mkdir(parents=True, exist_ok=True)
                for name in names:
                    (destination / name).write_bytes(contents[name])
                return __import__("subprocess").CompletedProcess(command, 0, "", "")

            with patch("kura.run_commands.runpod_ssh._run_bounded", side_effect=fake_scp), patch(
                "kura.run_commands.runpod_ssh._runpod_remote_training_states", return_value=[item]
            ):
                published = _pull_remote_training_state_items(
                    run_dir,
                    {"ip": "example", "port": 22, "key": root / "key"},
                    workspace="/workspace",
                    items=[item],
                )
            self.assertEqual([entry["observed_step"] for entry in published], [10])
            self.assertEqual(select_training_state(root, "source")["id"], published[0]["id"])

    def test_remote_process_local_state_is_not_retransferred_after_logical_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            run = {
                "id": "derived",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 2, "seed": 1},
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {
                        "artifact_id": "state-2", "manifest_sha256": "a" * 64,
                        "observed_step": 2, "recipe_sha256": "b" * 64,
                    },
                    "additional_steps": 1,
                    "target_step": 3,
                    "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
                },
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            candidate = root / "candidate"
            candidate.mkdir()
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                payload = _safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode())
                (candidate / name).write_bytes(payload)
            existing = publish_training_state(
                root, source_run="derived", source_realization=None, backend="musubi-tuner",
                observed_step=3, candidate=candidate, native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
            )
            item = {
                "path": "/workspace/runs/derived/outputs/derived-step00000001-state",
                "name": "derived-step00000001-state",
                "step": 1,
                "files": [{"path": "model.safetensors", "size": 1, "mtime_ns": 1}],
            }

            with patch("kura.run_commands.runpod_ssh._run_bounded") as transfer, patch(
                "kura.training_artifacts._sha256_file"
            ) as digest:
                published = _pull_remote_training_state_items(
                    run_dir, {"ip": "example", "port": 22, "key": root / "key"},
                    workspace="/workspace", items=[item],
                )

            transfer.assert_not_called()
            digest.assert_not_called()
            self.assertEqual([entry["id"] for entry in published], [existing["id"]])

    def test_remote_state_older_than_local_retention_window_is_not_retransferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {}},
                "recipe": {"steps": 3, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            protected = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="musubi-tuner",
                observed_step=1,
                candidate=self._candidate(root, "protected-1", b"1"),
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                keep_generations=2,
            )
            derived = root / "runs" / "derived"
            derived.mkdir()
            (derived / "run.yaml").write_text(yaml.safe_dump({
                "id": "derived",
                "continuation": {"mode": "resume", "source": {"artifact_id": protected["id"]}},
            }), encoding="utf-8")
            for step in (3, 4):
                publish_training_state(
                    root,
                    source_run="source",
                    source_realization=None,
                    backend="musubi-tuner",
                    observed_step=step,
                    candidate=self._candidate(root, f"retained-{step}", str(step).encode()),
                    native_format="accelerate-state-directory",
                    restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                    keep_generations=2,
                )
            self.assertTrue((root / "artifacts" / "training-state" / protected["id"]).is_dir())
            old_remote = {
                "path": "/workspace/runs/source/outputs/source-step00000002-state",
                "name": "source-step00000002-state",
                "step": 2,
                "logical_step": 2,
                "files": [{"path": "model.safetensors", "size": 1, "mtime_ns": 1}],
            }

            with patch(
                "kura.run_commands.runpod_ssh._run_bounded",
                side_effect=AssertionError("old state must not be transferred"),
            ):
                published = _pull_remote_training_state_items(
                    run_dir,
                    {"ip": "example", "port": 22, "key": root / "key"},
                    workspace="/workspace",
                    items=[old_remote],
                )

            self.assertEqual(published, [])

    def _candidate(self, root: Path, name: str, content: bytes) -> Path:
        candidate = root / name
        candidate.mkdir()
        (candidate / "model.safetensors").write_bytes(_safetensors_bytes(content))
        (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer-" + content))
        (candidate / "scheduler.bin").write_bytes(_torch_archive_bytes(b"scheduler-" + content))
        (candidate / "random_states_0.pkl").write_bytes(_torch_archive_bytes(b"rng-" + content))
        return candidate

    def test_publication_hashes_complete_state_and_retains_latest_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs" / "source").mkdir(parents=True)
            published = []
            for step in (100, 200, 300):
                published.append(
                    publish_training_state(
                        root,
                        source_run="source",
                        source_realization="realizations/launch.json",
                        backend="sd-scripts",
                        observed_step=step,
                        candidate=self._candidate(root, f"candidate-{step}", str(step).encode()),
                        native_format="accelerate-state-directory",
                        restoration_contract={"level": "best_effort_resume", "restored": ["model", "optimizer", "scheduler", "rng"], "not_restored": ["exact_dataloader_position"]},
                        runtime_identity={"image": "example@sha256:fixed"},
                        compatibility={"recipe_sha256": "sha256:recipe"},
                    )
                )

            manifests = sorted((root / "artifacts" / "training-state").glob("*/manifest.json"))
            self.assertEqual([json.loads(path.read_text())["observed_step"] for path in manifests], [200, 300])
            selected = select_training_state(root, "source")
            self.assertEqual(selected["observed_step"], 300)
            payload = verify_training_state(root, selected)
            self.assertTrue((payload / "model.safetensors").is_file())
            model = next(item for item in selected["files"] if item["path"] == "model.safetensors")
            self.assertEqual(model["sha256"], hashlib.sha256(_safetensors_bytes(b"300")).hexdigest())
            self.assertFalse((root / "artifacts" / "training-state" / published[0]["id"]).exists())

    def test_retention_preserves_a_valid_step_zero_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            published = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=0,
                candidate=self._candidate(root, "candidate-0", b"0"),
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                keep_generations=1,
            )

            self.assertTrue((root / "artifacts" / "training-state" / published["id"]).is_dir())
            self.assertEqual(select_training_state(root, "source")["observed_step"], 0)

    def test_retention_failure_does_not_relabel_a_completed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2):
                publish_training_state(
                    root,
                    source_run="source",
                    source_realization=None,
                    backend="sd-scripts",
                    observed_step=step,
                    candidate=self._candidate(root, f"candidate-{step}", str(step).encode()),
                    native_format="accelerate-state-directory",
                    restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                    keep_generations=2,
                )
            broken = root / "runs" / "broken"
            broken.mkdir(parents=True)
            (broken / "run.yaml").write_text("continuation: [\n", encoding="utf-8")
            published = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=3,
                candidate=self._candidate(root, "candidate-3", b"3"),
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                keep_generations=1,
            )

            self.assertEqual(published["observed_step"], 3)
            retained_steps = sorted(
                json.loads(path.read_text(encoding="utf-8"))["observed_step"]
                for path in (root / "artifacts" / "training-state").glob("*/manifest.json")
            )
            self.assertEqual(retained_steps, [1, 2, 3])

    def test_invalid_utf8_reference_does_not_relabel_a_completed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "runs" / "broken"
            broken.mkdir(parents=True)
            (broken / "run.yaml").write_bytes(b"\xff")

            published = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=1,
                candidate=self._candidate(root, "candidate-1", b"1"),
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                keep_generations=1,
            )

            self.assertEqual(published["observed_step"], 1)
            self.assertTrue((root / "artifacts" / "training-state" / published["id"]).is_dir())

    def test_publication_rejects_structurally_invalid_model_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "model.safetensors").write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "safetensors"):
                publish_training_state(
                    root,
                    source_run="source",
                    source_realization=None,
                    backend="sd-scripts",
                    observed_step=10,
                    candidate=candidate,
                    native_format="accelerate-state-directory",
                    restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                )
            self.assertFalse(any((root / "artifacts" / "training-state").glob("*/manifest.json")))

    def test_compiled_reference_protects_artifact_after_source_run_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runs" / "source"
            source.mkdir(parents=True)
            first = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="musubi-tuner",
                observed_step=100,
                candidate=self._candidate(root, "candidate-100", b"100"),
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
            )
            publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="musubi-tuner",
                observed_step=200,
                candidate=self._candidate(root, "candidate-200", b"200"),
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
            )
            resolved = root / "runs" / "derived" / "resolved"
            resolved.mkdir(parents=True)
            (resolved / "training-state-source.lock.json").write_text(
                json.dumps({"artifact_id": first["id"]}), encoding="utf-8"
            )
            source.rmdir()
            for step in (300, 400):
                publish_training_state(
                    root,
                    source_run="source",
                    source_realization=None,
                    backend="musubi-tuner",
                    observed_step=step,
                    candidate=self._candidate(root, f"candidate-{step}", str(step).encode()),
                    native_format="accelerate-state-directory",
                    restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                )
            self.assertEqual(load_training_state(root, first["id"])["observed_step"], 100)

    def test_verification_rejects_changed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root, "candidate", b"valid")
            manifest = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="musubi-tuner",
                observed_step=10,
                candidate=candidate,
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": ["optimizer"], "not_restored": ["data_position"]},
            )
            payload = root / manifest["payload"]
            original = (payload / "optimizer.bin").read_bytes()
            (payload / "optimizer.bin").write_bytes(b"x" * len(original))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_training_state(root, manifest)

    def test_recovery_surface_defaults_on_and_rejects_unknown_fields(self) -> None:
        self.assertEqual(training_state_policy({}), {"enabled": True, "keep_generations": 2})
        self.assertEqual(
            training_state_policy({"recovery": {"training_state": {"enabled": False, "keep_generations": 1}}}),
            {"enabled": False, "keep_generations": 1},
        )
        with self.assertRaisesRegex(ValueError, "unknown recovery.training_state fields"):
            training_state_policy({"recovery": {"training_state": {"keep": 2}}})
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            training_state_policy({"recovery": {"training_state": {"keep_generations": 3}}})

    def test_resume_surface_requires_one_target_and_frozen_source(self) -> None:
        intent = resume_intent(
            {
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 10, "recipe_sha256": "b" * 64},
                    "additional_steps": 5,
                    "target_step": 15,
                    "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
                },
            }
        )
        self.assertEqual(intent["target_step"], 15)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resume_intent(
                {
                    "parent_run": "source",
                    "continuation": {
                        "mode": "resume",
                        "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 10, "recipe_sha256": "b" * 64},
                        "additional_steps": 5,
                        "to_step": 15,
                        "target_step": 15,
                        "restoration_contract": {},
                    },
                }
            )
        unsupported = {
            "parent_run": "source",
            "continuation": {
                "mode": "resume",
                "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 10, "recipe_sha256": "b" * 64},
                "additional_steps": 5,
                "target_step": 15,
                "restoration_contract": {"level": "unsupported", "restored": [], "not_restored": [], "limitations": ["not implemented"]},
            },
        }
        with self.assertRaisesRegex(ValueError, "unsupported.*not implemented"):
            resume_intent(unsupported)

    def test_compile_lock_rejects_recipe_change_and_freezes_verified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_recipe = {
                "backend": {"name": "sd-scripts", "config": {"architecture": "sd15"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 100, "seed": 1},
            }
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer"))
            manifest = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=100,
                candidate=candidate,
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": ["optimizer"], "not_restored": ["data_position"]},
                compatibility={"recipe_sha256": recipe_fingerprint(source_recipe)},
            )
            run = {
                **source_recipe,
                "id": "derived",
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {"artifact_id": manifest["id"], "manifest_sha256": manifest["manifest_sha256"], "observed_step": 100, "recipe_sha256": recipe_fingerprint(source_recipe)},
                    "additional_steps": 50,
                    "target_step": 150,
                    "restoration_contract": manifest["restoration_contract"],
                },
            }
            resolved = root / "runs" / "derived" / "resolved"
            lock = compile_resume_lock(root, run, resolved)
            self.assertEqual(lock["native_state_path"], f"/workspace/artifacts/training-state/{manifest['id']}/payload")
            self.assertEqual(lock["target_step"], 150)
            self.assertTrue((resolved / "training-state-source.lock.json").is_file())
            changed = json.loads(json.dumps(run))
            changed["backend"]["config"]["learning_rate"] = 0.2
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                compile_resume_lock(root, changed, root / "other")

    def test_compile_lock_rejects_changed_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {
                "backend": {"name": "sd-scripts", "config": {"architecture": "sd15"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 100, "seed": 1},
            }
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer"))
            source_runtime = {
                "adapter_source": {"kind": "tree", "sha256": "a" * 64},
                "actual_executor": "docker",
                "actual_image_identity": {"reference": "image@sha256:old", "pinning": {"strength": "content-hash", "value": "sha256:old"}},
                "local_image_identity": {"reference": "image@sha256:old"},
                "remote_image_identity": {"reference": "image@sha256:remote"},
            }
            manifest = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=100,
                candidate=candidate,
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                runtime_identity=source_runtime,
                compatibility={"recipe_sha256": recipe_fingerprint(recipe)},
            )
            run = {
                **recipe,
                "id": "derived",
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {"artifact_id": manifest["id"], "manifest_sha256": manifest["manifest_sha256"], "observed_step": 100, "recipe_sha256": recipe_fingerprint(recipe)},
                    "additional_steps": 10,
                    "target_step": 110,
                    "restoration_contract": manifest["restoration_contract"],
                },
            }
            changed_runtime = {
                "adapter_source": source_runtime["adapter_source"],
                "declared_executor": "docker",
                "local_image_identity": {"reference": "image@sha256:new"},
                "remote_image_identity": source_runtime["remote_image_identity"],
                "selected_image_identity": {"reference": "image@sha256:new", "pinning": {"strength": "content-hash", "value": "sha256:new"}},
            }
            with self.assertRaisesRegex(ValueError, "runtime image identity differs"):
                compile_resume_lock(root, run, root / "resolved", target_runtime_identity=changed_runtime)

    def test_compile_lock_allows_same_runpod_mutable_reference_as_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {
                "backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 100, "seed": 1},
            }
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer"))
            identity = {"reference": "example/image:v1", "pinning": {"strength": "mutable-reference", "observation": "not-observed"}}
            runtime = {
                "adapter_source": {"kind": "tree", "sha256": "a" * 64},
                "actual_executor": "runpod",
                "actual_image_identity": identity,
            }
            manifest = publish_training_state(
                root, source_run="source", source_realization=None, backend="musubi-tuner",
                observed_step=100, candidate=candidate, native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": [], "not_restored": []},
                runtime_identity=runtime, compatibility={"recipe_sha256": recipe_fingerprint(recipe)},
            )
            run = {
                **recipe, "id": "derived", "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {
                        "artifact_id": manifest["id"], "manifest_sha256": manifest["manifest_sha256"],
                        "observed_step": 100, "recipe_sha256": recipe_fingerprint(recipe),
                    },
                    "additional_steps": 10, "target_step": 110,
                    "restoration_contract": manifest["restoration_contract"],
                },
            }
            target = {
                "adapter_source": runtime["adapter_source"],
                "declared_executor": "runpod",
                "selected_image_identity": identity,
            }

            lock = compile_resume_lock(root, run, root / "resolved", target_runtime_identity=target)

            self.assertEqual(lock["target_step"], 110)

    def test_backend_state_directory_is_published_only_with_required_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "outputs" / "source-step00000010-state").mkdir(parents=True)
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            state = run_dir / "outputs" / "source-step00000010-state"
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                (state / name).write_bytes(_safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode()))
            published = publish_completed_training_states(root, run_dir)
            self.assertEqual([item["observed_step"] for item in published], [10])
            self.assertEqual(published[0]["restoration_contract"]["level"], "best_effort_resume")
            (run_dir / "outputs" / "source-step00000020-state").mkdir()
            self.assertEqual([item["observed_step"] for item in publish_completed_training_states(root, run_dir)], [10])

    def test_process_local_resume_state_is_published_at_its_logical_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            state = run_dir / "outputs" / "derived-step00000001-state"
            state.mkdir(parents=True)
            run = {
                "id": "derived",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 2, "seed": 1},
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {
                        "artifact_id": "state-2",
                        "manifest_sha256": "a" * 64,
                        "observed_step": 2,
                        "recipe_sha256": "b" * 64,
                    },
                    "additional_steps": 1,
                    "target_step": 3,
                    "restoration_contract": {
                        "level": "best_effort_resume",
                        "restored": ["model", "optimizer"],
                        "not_restored": ["application_global_step"],
                    },
                },
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                payload = _safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode())
                (state / name).write_bytes(payload)

            published = publish_completed_training_states(root, run_dir)

            self.assertEqual([item["observed_step"] for item in published], [3])
            self.assertEqual(published[0]["save_event_id"], "derived:step:3")

    def test_sd_scripts_resume_state_normalizes_application_step_for_the_next_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            state = run_dir / "outputs" / "derived-step00000001-state"
            state.mkdir(parents=True)
            restoration = {
                "level": "best_effort_resume",
                "restored": ["model", "optimizer", "scheduler", "rng"],
                "not_restored": ["application_global_step", "application_epoch_counter", "exact_dataloader_position"],
            }
            run = {
                "id": "derived",
                "type": "train",
                "backend": {
                    "name": "sd-scripts",
                    "config": {"architecture": "sd15", "mode": "lora", "output_name": "source"},
                },
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 2, "seed": 1},
                "parent_run": "source",
                "continuation": {
                    "mode": "resume",
                    "source": {
                        "artifact_id": "state-2",
                        "manifest_sha256": "a" * 64,
                        "observed_step": 2,
                        "recipe_sha256": "b" * 64,
                    },
                    "additional_steps": 1,
                    "target_step": 3,
                    "restoration_contract": restoration,
                },
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                payload = _safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode())
                (state / name).write_bytes(payload)
            (state / "train_state.json").write_text('{"current_epoch":3,"current_step":3}\n', encoding="utf-8")
            _write_state_marker(state, "sd-scripts", 3)

            published = publish_completed_training_states(root, run_dir)

            self.assertEqual([item["observed_step"] for item in published], [3])
            payload = root / published[0]["payload"]
            normalized = json.loads((payload / "train_state.json").read_text(encoding="utf-8"))
            self.assertEqual(normalized, {"current_epoch": 3, "current_step": 3})

    def test_backend_final_state_without_an_embedded_step_is_not_published_at_an_assumed_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            state = run_dir / "outputs" / "source-state"
            state.mkdir(parents=True)
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                (state / name).write_bytes(_safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode()))
            published = publish_completed_training_states(root, run_dir, allow_final_state=True)
            self.assertEqual(published, [])

    def test_retention_counts_distinct_logical_steps_as_generations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_ids = []
            for step, content in ((1, b"step-1"), (2, b"step-2-periodic"), (2, b"step-2-final")):
                candidate = root / f"candidate-{len(artifact_ids)}"
                candidate.mkdir()
                (candidate / "model.safetensors").write_bytes(_safetensors_bytes(content))
                manifest = publish_training_state(
                    root,
                    source_run="source",
                    source_realization=None,
                    backend="sd-scripts",
                    observed_step=step,
                    candidate=candidate,
                    native_format="accelerate-state-directory",
                    restoration_contract={"level": "best_effort_resume", "restored": ["model"], "not_restored": []},
                    keep_generations=2,
                )
                artifact_ids.append(manifest["id"])

            retained = {path.parent.name for path in (root / "artifacts" / "training-state").glob("*/manifest.json")}
            self.assertEqual(retained, set(artifact_ids))

    def test_ai_toolkit_pair_is_published_as_partial_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "ai-toolkit", "config": {}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "failed"}), encoding="utf-8")
            state = run_dir / "outputs" / "source-step00000010-state"
            state.mkdir(parents=True)
            (state / "model.safetensors").write_bytes(_safetensors_bytes(b"weight"))
            (state / "optimizer.pt").write_bytes(_torch_archive_bytes(b"optimizer"))
            (state / "rng.pt").write_bytes(_torch_archive_bytes(b"rng"))
            _write_state_marker(state, "ai-toolkit", 10)
            published = publish_completed_training_states(root, run_dir)
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0]["restoration_contract"]["level"], "partial_resume")
            self.assertEqual(
                published[0]["restoration_contract"]["not_restored"],
                ["scheduler", "exact_rng_position", "exact_dataloader_position"],
            )

    def test_completion_marker_with_a_stale_payload_digest_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            state = run_dir / "outputs" / "source-step00000010-state"
            state.mkdir(parents=True)
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "ai-toolkit", "config": {}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (state / "model.safetensors").write_bytes(_safetensors_bytes(b"weight"))
            (state / "optimizer.pt").write_bytes(_torch_archive_bytes(b"optimizer"))
            (state / "rng.pt").write_bytes(_torch_archive_bytes(b"rng"))
            _write_state_marker(state, "ai-toolkit", 10)
            (state / "optimizer.pt").write_bytes(_torch_archive_bytes(b"overwritten-after-marker"))

            self.assertEqual(publish_completed_training_states(root, run_dir), [])

    def test_periodic_and_final_views_of_one_state_are_reported_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "ai-toolkit", "config": {}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 2, "seed": 1},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            periodic = run_dir / "outputs" / "source-step00000002-state"
            final = run_dir / "outputs" / "source-state"
            for state in (periodic, final):
                state.mkdir(parents=True)
                (state / "model.safetensors").write_bytes(_safetensors_bytes(b"weight"))
                (state / "optimizer.pt").write_bytes(_torch_archive_bytes(b"optimizer"))
                (state / "rng.pt").write_bytes(_torch_archive_bytes(b"rng"))
                _write_state_marker(state, "ai-toolkit", 2)

            published = publish_completed_training_states(root, run_dir, allow_final_state=True)

            self.assertEqual(len(published), 1)
            self.assertEqual(published[0]["observed_step"], 2)


class ResumeRunTests(unittest.TestCase):
    def test_terminal_snapshot_reuses_exact_published_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "source"
            run_dir.mkdir(parents=True)
            candidate = workspace / "candidate"
            candidate.mkdir()
            (candidate / "model.safetensors").write_bytes(_safetensors_bytes(b"weight"))
            (candidate / "state-info.json").write_text(
                json.dumps({"schema_version": 1, "logical_step": 50}), encoding="utf-8"
            )
            artifact = publish_training_state(
                workspace,
                source_run="source",
                source_realization=None,
                backend="ai-toolkit",
                observed_step=50,
                candidate=candidate,
                native_format="ai-toolkit-kura-state-v1",
                restoration_contract={"level": "partial_resume"},
            )
            state_root = "outputs/source-step00000050-state"
            remote_manifest = [
                {
                    "path": f"{state_root}/{item['path']}",
                    "size": item["size"],
                    "mtime_ns": 10,
                    "sha256": item["sha256"],
                }
                for item in artifact["files"]
            ]

            reusable = _local_reusable_training_state_sources(run_dir, remote_manifest)

            self.assertEqual(set(reusable), {item["path"] for item in remote_manifest})
            payload = workspace / artifact["payload"]
            for item in artifact["files"]:
                self.assertTrue(reusable[f"{state_root}/{item['path']}"].samefile(payload / item["path"]))

    def test_runpod_download_refuses_terminal_snapshot_without_required_state(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "source"
            downloaded = run_dir / "downloads" / "source"
            (downloaded / "outputs").mkdir(parents=True)
            (downloaded / "realizations").mkdir()
            (downloaded / "realizations" / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}), encoding="utf-8"
            )
            (run_dir / "resolved").mkdir(parents=True)
            manifest = {
                "id": "source",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {}},
                "recipe": {"steps": 100, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            os.chdir(root)
            try:
                self.assertEqual(_download_run_unlocked("source"), 1)
            finally:
                os.chdir(previous)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")

    def test_runpod_download_materializes_logical_resume_target(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "derived"
            downloaded = run_dir / "downloads" / "derived"
            (downloaded / "outputs").mkdir(parents=True)
            (downloaded / "realizations").mkdir()
            (downloaded / "realizations" / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}), encoding="utf-8"
            )
            (run_dir / "resolved").mkdir(parents=True)
            manifest = {
                "id": "derived",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {}},
                "recipe": {"steps": 100, "seed": 1},
                "continuation": {
                    "mode": "resume",
                    "source": {"observed_step": 100},
                    "target_step": 150,
                },
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            os.chdir(root)
            try:
                self.assertEqual(_download_run_unlocked("derived"), 0)
            finally:
                os.chdir(previous)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((status["last_step"], status["total_steps"]), (150, 150))
            self.assertEqual((status["current_run_step"], status["current_run_total_steps"]), (50, 50))

    def test_plan_formatter_separates_resume_contract_from_native_steps(self) -> None:
        output = format_run_plan(
            {
                "id": "derived",
                "type": "train",
                "compiled": True,
                "backend": {"name": "musubi-tuner", "config": {}},
                "model": {},
                "compute": {},
                "resume": {
                    "source_run": "source",
                    "artifact_id": "state-1",
                    "source_step": 2000,
                    "target_step": 3000,
                    "additional_steps": 1000,
                    "restoration_level": "best_effort_resume",
                    "restored": ["optimizer", "scheduler"],
                    "not_restored": ["exact_dataloader_position"],
                    "scheduler_behavior": "restored",
                    "native_start": 0,
                    "native_target": 1000,
                    "state_bytes": 1024,
                },
                "training_state": {
                    "enabled": True,
                    "keep_generations": 2,
                    "cadence_steps": 1000,
                    "capability": "best_effort_resume",
                },
                "datasets": [],
                "recipe": {},
                "sampling": {},
                "resources": {},
                "preflight": [],
            }
        )
        self.assertIn("Resume", output)
        self.assertIn("source_step  2000", output)
        self.assertIn("target_step  3000", output)
        self.assertIn("native_steps 0 -> 1000", output)
        self.assertIn("exact_dataloader_position", output)
        self.assertIn("continuity   exact equivalence is not guaranteed; missing: exact_dataloader_position", output)
        self.assertNotIn("HIGH", output)
        self.assertNotIn("CAUTION", output)
        self.assertIn("Training state", output)
        self.assertIn("keep         2", output)

    def test_resume_refuses_a_capture_only_backend_before_creating_a_run(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            source = self._source_run(root)
            source_run = yaml.safe_load((source / "run.yaml").read_text(encoding="utf-8"))
            source_run["backend"]["config"]["architecture"] = "anima"
            for path in (source / "run.yaml", source / "resolved" / "manifest.lock.yaml"):
                path.write_text(yaml.safe_dump(source_run, sort_keys=False), encoding="utf-8")
            candidate = root / "state"
            candidate.mkdir()
            (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer"))
            publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=2000,
                candidate=candidate,
                native_format="accelerate-state-directory",
                restoration_contract={"level": "partial_resume", "restored": [], "not_restored": ["global_step"]},
            )
            os.chdir(root)
            stderr = io.StringIO()
            try:
                with patch("sys.stderr", stderr):
                    code = cmd_run_resume(argparse.Namespace(source_run="source", additional_steps=100, to_step=None, artifact=None, slug="more", executor=None, gpu=None))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            self.assertIn("unsupported", stderr.getvalue().lower())
            self.assertEqual([path.name for path in (root / "runs").iterdir()], ["source"])

    def _source_run(self, root: Path) -> Path:
        run_dir = root / "runs" / "source"
        run_dir.mkdir(parents=True)
        run = {
            "schema_version": 2,
            "id": "source",
            "type": "train",
            "experiment": "exp",
            "created": "2026-08-01T00:00:00+09:00",
            "created_by": "human",
            "parent_run": None,
            "intent": "source",
            "backend": {"name": "sd-scripts", "version": None, "adapter_version": 1, "config": {"architecture": "sd15", "learning_rate": 0.0001}},
            "model": {"base": "example/model", "revision": "fixed"},
            "datasets": [{"id": "tiny", "digest": "sha256:data", "role": None}],
            "recipe": {"steps": 2000, "seed": 1},
            "compute": {"executor": "runpod", "gpu": "NVIDIA A40", "capacity": {"mode": "immediate"}},
            "sampling": {"prompts": [], "cadence_steps": None},
            "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
        }
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        return run_dir

    def test_resume_creates_derived_run_and_freezes_latest_artifact(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            source = self._source_run(root)
            candidate = root / "state"
            candidate.mkdir()
            (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer"))
            manifest = publish_training_state(
                root,
                source_run="source",
                source_realization="realizations/launch.json",
                backend="sd-scripts",
                observed_step=2000,
                candidate=candidate,
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": ["optimizer"], "not_restored": ["data_position"]},
            )
            os.chdir(root)
            stdout = io.StringIO()
            try:
                with patch("sys.stdout", stdout):
                    code = cmd_run_resume(argparse.Namespace(source_run="source", additional_steps=1000, to_step=None, artifact=None, slug="more", executor=None, gpu=None))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            run_id = stdout.getvalue().strip()
            derived = yaml.safe_load((root / "runs" / run_id / "run.yaml").read_text(encoding="utf-8"))
            self.assertEqual(derived["parent_run"], "source")
            self.assertEqual(derived["recipe"], yaml.safe_load((source / "run.yaml").read_text())["recipe"])
            self.assertEqual(derived["continuation"]["mode"], "resume")
            self.assertEqual(derived["continuation"]["additional_steps"], 1000)
            self.assertEqual(derived["continuation"]["source"]["artifact_id"], manifest["id"])
            self.assertRegex(derived["continuation"]["source"]["manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(derived["continuation"]["source"]["observed_step"], 2000)

    def test_runpod_stage_contains_only_selected_resume_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "realizations").mkdir()
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            candidate = root / "state"
            candidate.mkdir()
            (candidate / "optimizer.bin").write_bytes(_torch_archive_bytes(b"optimizer"))
            manifest = publish_training_state(
                root,
                source_run="source",
                source_realization=None,
                backend="sd-scripts",
                observed_step=10,
                candidate=candidate,
                native_format="accelerate-state-directory",
                restoration_contract={"level": "best_effort_resume", "restored": ["optimizer"], "not_restored": []},
            )
            run = {"id": "derived", "continuation": {"mode": "resume", "source": {"artifact_id": manifest["id"], "manifest_sha256": manifest["manifest_sha256"]}}}
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")

            record = stage_runpod(workspace=root, run_dir=run_dir, dataset_ids=[], config={"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"]})
            archive_path = run_dir / record["archive"]
            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            artifact_prefix = f"artifacts/training-state/{manifest['id']}"
            self.assertIn(f"{artifact_prefix}/manifest.json", names)
            self.assertIn(f"{artifact_prefix}/payload/optimizer.bin", names)
            self.assertFalse(any(name.startswith("runs/source/") for name in names))

    def test_failed_local_process_publishes_last_complete_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "realizations").mkdir()
            state = run_dir / "outputs" / "source-step00000010-state"
            state.mkdir(parents=True)
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                (state / name).write_bytes(_safetensors_bytes(name.encode()) if name == "model.safetensors" else _torch_archive_bytes(name.encode()))
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2"}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            realization = {"id": "launch", "container": {"id": "container-1"}}
            (run_dir / "realizations" / "launch.json").write_text(json.dumps(realization), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/launch.json"}), encoding="utf-8")
            docker_state = {"Running": False, "ExitCode": 137, "FinishedAt": "2026-08-27T00:00:00Z", "Error": ""}
            result = __import__("subprocess").CompletedProcess([], 0, json.dumps(docker_state), "")
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "failed")
            selected = select_training_state(root, "source")
            self.assertEqual(selected["observed_step"], 10)

    def test_completed_local_process_records_missing_required_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "source"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "realizations").mkdir()
            run = {
                "id": "source",
                "type": "train",
                "backend": {"name": "ai-toolkit", "config": {}},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny", "digest": "sha256:data"}],
                "recipe": {"steps": 20, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            realization = {"id": "launch", "container": {"id": "container-1"}}
            (run_dir / "realizations" / "launch.json").write_text(json.dumps(realization), encoding="utf-8")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "last_realization": "realizations/launch.json"}),
                encoding="utf-8",
            )
            docker_state = {"Running": False, "ExitCode": 0, "FinishedAt": "2026-08-27T00:00:00Z", "Error": ""}
            result = subprocess.CompletedProcess([], 0, json.dumps(docker_state), "")

            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)

            self.assertEqual(status["state"], "completed")
            self.assertIn("no valid training-state artifact", status["training_state_sync_error"])


if __name__ == "__main__":
    unittest.main()
