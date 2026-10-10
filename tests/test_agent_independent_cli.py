"""Acceptance proof that frozen files, not an agent session, drive Kura."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import yaml

from kura.cli import cmd_init, cmd_run_compile, cmd_run_new, cmd_run_plan, cmd_run_status
from kura.run_commands.launch import launch_run
from kura.model_requirements import model_requirements
from kura.provenance import artifact_pinning
from kura.run_envelope import backend_config, common_recipe
from kura.backends import BACKENDS
from tests.platform_support import POSIX_PATHS, posix_only


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


class AgentIndependentCliTests(unittest.TestCase):
    def test_removed_backend_config_spelling_is_rejected(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {}}, "backend_overrides": {}}
        with self.assertRaisesRegex(ValueError, "is not supported"):
            backend_config(run)

    def test_recipe_rejects_backend_dependent_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "put them under backend.config"):
            common_recipe({"recipe": {"learning_rate": 0.0001}})

    def test_removed_params_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "params is not supported"):
            common_recipe({"params": {"steps": 1}})

    def test_model_requirement_exposes_pinning_strength(self) -> None:
        pinned = artifact_pinning({"kind": "huggingface-file", "repo_id": "org/model", "revision": "0123456789abcdef0123456789abcdef01234567"}, observable=True)
        mutable = artifact_pinning({"kind": "huggingface-file", "repo_id": "org/model", "revision": "main"}, observable=True)
        self.assertEqual(pinned["strength"], "immutable-revision")
        self.assertEqual(mutable["strength"], "mutable-reference")

    def test_ai_toolkit_requirement_records_no_revision_it_cannot_pin(self) -> None:
        requirement = model_requirements({"backend": {"name": "ai-toolkit"}, "model": {"base": "org/model", "revision": "main"}})[0]
        self.assertEqual(requirement["identity"], {"kind": "huggingface-repository", "repo_id": "org/model"})
        self.assertEqual(requirement["pinning"]["strength"], "mutable-reference")

    def _dataset(self, root: Path) -> None:
        dataset = root / "datasets" / "tiny"
        (dataset / "images").mkdir(parents=True)
        (dataset / "images" / "001.png").write_bytes(PNG)
        (dataset / "images" / "001.txt").write_text("a tiny test image\n", encoding="utf-8")
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({
            "id": "tiny", "items_schema_version": 2, "stats": {"count": 1},
        }), encoding="utf-8")
        (dataset / "items.jsonl").write_text(json.dumps({
            "id": "001",
            "files": [{"type": "file", "role": "target", "path": "images/001.png"}],
            "caption": {"file": {"type": "file", "path": "images/001.txt"}},
        }) + "\n", encoding="utf-8")

    def _exercise(self, backend: str) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                self._dataset(root)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="file-only", slug=backend, backend=backend, executor="docker", gpu="cpu")), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["intent"] = "prove the CLI can execute authored files without agent state"
                run["model"] = {"base": "example/model", "revision": "0123456789abcdef0123456789abcdef01234567"}
                run["datasets"] = [{"id": "tiny", "digest": None, "role": None}]
                run["recipe"] = {"steps": 1, "seed": 1}
                if backend == "ai-toolkit":
                    run["backend"]["config"] = {"model_arch": "sdxl", "network_dim": 4, "network_alpha": 4, "learning_rate": 0.0001, "batch_size": 1}
                else:
                    run["backend"]["config"] = {"architecture": "flux2", "model_version": "klein-base-4b", "network_dim": 4, "learning_rate": 0.0001, "resolution": [64, 64], "batch_size": 1, "model_paths": {"dit": "/workspace/cache/models/dit.safetensors", "vae": "/workspace/cache/models/vae.safetensors", "text_encoder": "/workspace/cache/models/text.safetensors"}}
                    models = root / "cache" / "models"
                    models.mkdir(parents=True, exist_ok=True)
                    for name in ("dit.safetensors", "vae.safetensors", "text.safetensors"):
                        (models / name).write_bytes(b"model")
                run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")

                self.assertEqual(cmd_run_compile(argparse.Namespace(run_id=run_id)), 0)
                frozen = json.loads((root / "runs" / run_id / "resolved" / "backend-command.lock.json").read_text(encoding="utf-8"))
                self.assertEqual(frozen["backend"], backend)
                self.assertIn("adapter_source", frozen)
                command_was_recomputed = Mock(side_effect=AssertionError("launch recomputed adapter command"))
                with patch.dict(BACKENDS, {backend: replace(BACKENDS[backend], command=command_was_recomputed)}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id=run_id, executor="docker", json=False)), 0)
                    self.assertEqual(launch_run(run_id, executor="docker", dry_run=True, image=None), 0)
                    self.assertEqual(cmd_run_status(argparse.Namespace(run_id=run_id)), 0)
                command_was_recomputed.assert_not_called()
                manifest = yaml.safe_load((root / "runs" / run_id / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
                serialized = yaml.safe_dump(manifest)
                self.assertNotIn("conversation", serialized)
                self.assertNotIn("session_id", serialized)
            finally:
                os.chdir(previous)

    def _compile_with_model(self, backend: str, model: dict) -> str:
        """Compile one shared run with the given model block; return the refusal, or "" when it compiled."""
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                self._dataset(root)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="no-base", slug=backend, backend=backend, executor="docker", gpu="cpu")), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["intent"] = "compile with an authored model block"
                run["model"] = model
                run["datasets"] = [{"id": "tiny", "digest": None, "role": None}]
                run["recipe"] = {"steps": 1, "seed": 1}
                models = root / "cache" / "models"
                models.mkdir(parents=True, exist_ok=True)
                for name in ("base.safetensors", "dit.safetensors", "vae.safetensors", "text.safetensors"):
                    (models / name).write_bytes(b"model")
                run["backend"]["config"] = {
                    "ai-toolkit": {"model_arch": "sdxl", "network_dim": 4, "network_alpha": 4, "learning_rate": 0.0001, "batch_size": 1},
                    "musubi-tuner": {"architecture": "flux2", "model_version": "klein-base-4b", "network_dim": 4, "learning_rate": 0.0001, "resolution": [64, 64], "batch_size": 1, "model_paths": {"dit": "/workspace/cache/models/dit.safetensors", "vae": "/workspace/cache/models/vae.safetensors", "text_encoder": "/workspace/cache/models/text.safetensors"}},
                    "sd-scripts": {"architecture": "sd15", "mode": "lora", "network_dim": 4, "learning_rate": 0.0001, "model_paths": {"base": "/workspace/cache/models/base.safetensors"}, "dataset_config": {"general": {"resolution": [64, 64], "caption_extension": ".txt"}, "datasets": [{"batch_size": 1, "subsets": [{"dataset_id": "tiny", "num_repeats": 1}]}]}},
                }[backend]
                run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
                stderr = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    code = cmd_run_compile(argparse.Namespace(run_id=run_id))
                self.assertEqual(code != 0, bool(stderr.getvalue()), stderr.getvalue())
                return stderr.getvalue()
            finally:
                os.chdir(previous)

    @posix_only(POSIX_PATHS)
    def test_only_ai_toolkit_requires_model_base(self) -> None:
        # Only AI-Toolkit trains from model.base; the others read their own model sources.
        for model in ({"base": "", "revision": None}, {"revision": None}):
            for backend in ("sd-scripts", "musubi-tuner"):
                with self.subTest(backend=backend, model=model):
                    self.assertEqual(self._compile_with_model(backend, model), "")
            with self.subTest(backend="ai-toolkit", model=model):
                self.assertRegex(self._compile_with_model("ai-toolkit", model), "AI-Toolkit trains from model.base")

    @posix_only(POSIX_PATHS)
    def test_model_revision_stays_a_label_for_every_backend(self) -> None:
        # Earlier runs wrote it, and Resume fingerprints include it, so no backend refuses it.
        for backend in ("ai-toolkit", "sd-scripts", "musubi-tuner"):
            with self.subTest(backend=backend):
                self.assertEqual(self._compile_with_model(backend, {"base": "example/model", "revision": "main"}), "")

    @posix_only(POSIX_PATHS)
    def test_ai_toolkit_file_only_lifecycle(self) -> None:
        self._exercise("ai-toolkit")

    @posix_only(POSIX_PATHS)
    def test_musubi_file_only_lifecycle(self) -> None:
        self._exercise("musubi-tuner")


if __name__ == "__main__":
    unittest.main()
