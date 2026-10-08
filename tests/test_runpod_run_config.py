"""Training and render launches, and the plan, read a run's GPU choice and the image's CUDA floor the same way."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


class GpuChoiceTests(unittest.TestCase):
    def test_a_gpu_list_on_a_render_is_honored_as_on_a_training_run(self) -> None:
        from kura.run_commands import render_runpod

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\nimages:\n  comfyui: remote/comfy\nrunpod:\n  gpu_type_ids: [A]\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({
                "type": "render", "generator": {"name": "comfyui"}, "executor": {"name": "runpod"},
                "compute": {"gpu": ["NVIDIA RTX A5000", "NVIDIA A40"]}, "comfyui_models": [], "comfyui_model_registry": {},
            }), encoding="utf-8")
            (run_dir / "status.json").write_text('{"state": "compiled"}', encoding="utf-8")
            prepared: dict = {}
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(render_runpod, "confirm_runpod_billing"):
                    self.assertEqual(render_runpod.launch_render_runpod("example", dry_run=False, yes=True, check_only=True, prepared=prepared), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(prepared["runpod_config"]["gpu_type_ids"], ["NVIDIA RTX A5000", "NVIDIA A40"])

    def test_every_reader_takes_the_gpu_choice_and_cuda_floor_from_one_place(self) -> None:
        from kura.executors import runpod
        from kura.run_commands import common, launch, plan, render_runpod

        for module in (launch, plan, render_runpod):
            with self.subTest(module=module.__name__):
                self.assertIs(module.requested_gpu_types, common.requested_gpu_types)
        from kura import images

        for module in (plan, runpod):
            with self.subTest(module=module.__name__):
                self.assertIs(module.runpod_min_cuda_for, images.runpod_min_cuda_for)


if __name__ == "__main__":
    unittest.main()
