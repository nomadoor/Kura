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
from unittest.mock import patch

import yaml

from kura.cli import cmd_init, cmd_run_compile, cmd_run_launch, cmd_run_new, cmd_run_plan, cmd_run_status
from kura.model_requirements import model_requirements
from kura.run_envelope import backend_config, common_recipe


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
        pinned = model_requirements({"backend": {"name": "ai-toolkit"}, "model": {"base": "org/model", "revision": "0123456789abcdef0123456789abcdef01234567"}})
        mutable = model_requirements({"backend": {"name": "ai-toolkit"}, "model": {"base": "org/model", "revision": "main"}})
        self.assertEqual(pinned[0]["pinning"]["strength"], "immutable-revision")
        self.assertEqual(mutable[0]["pinning"]["strength"], "mutable-reference")

    def _dataset(self, root: Path) -> None:
        dataset = root / "datasets" / "tiny"
        (dataset / "images").mkdir(parents=True)
        (dataset / "images" / "001.png").write_bytes(PNG)
        (dataset / "images" / "001.txt").write_text("a tiny test image\n", encoding="utf-8")
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({"id": "tiny", "stats": {"count": 1}}), encoding="utf-8")
        (dataset / "items.jsonl").write_text(json.dumps({"id": "001", "path": "images/001.png", "caption": "a tiny test image"}) + "\n", encoding="utf-8")

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
                    run["backend"]["config"] = {"model_arch": "sdxl", "config": {"network": {"linear": 4, "linear_alpha": 4}, "train": {"lr": 0.0001, "batch_size": 1}}}
                else:
                    run["backend"]["config"] = {"architecture": "flux2", "model_version": "klein-base-4b", "network_dim": 4, "learning_rate": 0.0001, "dataset_config": {"general": {"resolution": [64, 64], "batch_size": 1}}, "model_paths": {"dit": "/workspace/cache/models/dit.safetensors", "vae": "/workspace/cache/models/vae.safetensors", "text_encoder": "/workspace/cache/models/text.safetensors"}}
                run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")

                self.assertEqual(cmd_run_compile(argparse.Namespace(run_id=run_id)), 0)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id=run_id, executor="docker", json=False)), 0)
                    self.assertEqual(cmd_run_launch(argparse.Namespace(run_id=run_id, executor="docker", dry_run=True, image=None, notify=None, wait=False)), 0)
                    self.assertEqual(cmd_run_status(argparse.Namespace(run_id=run_id)), 0)
                manifest = yaml.safe_load((root / "runs" / run_id / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
                serialized = yaml.safe_dump(manifest)
                self.assertNotIn("conversation", serialized)
                self.assertNotIn("session_id", serialized)
            finally:
                os.chdir(previous)

    def test_ai_toolkit_file_only_lifecycle(self) -> None:
        self._exercise("ai-toolkit")

    def test_musubi_file_only_lifecycle(self) -> None:
        self._exercise("musubi-tuner")


if __name__ == "__main__":
    unittest.main()
