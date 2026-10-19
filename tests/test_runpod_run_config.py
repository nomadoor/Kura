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

    def test_a_render_launches_the_image_frozen_at_compile(self) -> None:
        from kura.run_commands import render_runpod

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\nimages:\n  comfyui: remote/comfy@sha256:" + "2" * 64 + "\nrunpod:\n  gpu_type_ids: [A]\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({
                "type": "render", "generator": {"name": "comfyui"}, "executor": {"name": "runpod"},
                "comfyui_models": [], "comfyui_model_registry": {},
            }), encoding="utf-8")
            frozen = "remote/comfy@sha256:" + "1" * 64
            (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump({"selected_image": frozen, "image_origin": "override"}), encoding="utf-8")
            (run_dir / "status.json").write_text('{"state": "compiled"}', encoding="utf-8")
            prepared: dict = {}
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(render_runpod, "confirm_runpod_billing"):
                    self.assertEqual(render_runpod.launch_render_runpod("example", dry_run=False, yes=True, check_only=True, prepared=prepared), 0)
                # A render compiled by an older Kura froze no image and launches the current one.
                (run_dir / "resolved" / "env.lock").write_text("generator: comfyui\n", encoding="utf-8")
                older: dict = {}
                with patch.object(render_runpod, "confirm_runpod_billing"):
                    self.assertEqual(render_runpod.launch_render_runpod("example", dry_run=False, yes=True, check_only=True, prepared=older), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(prepared["remote_image"], frozen)
            self.assertEqual(older["remote_image"], "remote/comfy@sha256:" + "2" * 64)

    def test_every_reader_takes_the_cuda_floor_from_one_place(self) -> None:
        from kura.executors import runpod
        from kura.run_commands import plan
        from kura import images

        for module in (plan, runpod):
            with self.subTest(module=module.__name__):
                self.assertIs(module.runpod_min_cuda_version, images.runpod_min_cuda_version)
                self.assertFalse(hasattr(module, "runpod_min_cuda_for"))

    def test_every_launch_and_the_plan_read_their_runpod_settings_from_one_owner(self) -> None:
        from kura.run_commands import common, launch, plan, render_runpod

        for module in (launch, plan, render_runpod):
            with self.subTest(module=module.__name__):
                self.assertIs(module.runpod_launch_settings, common.runpod_launch_settings)
        self.assertFalse(hasattr(common, "runpod_settings_for_adapter"))
        self.assertFalse(hasattr(render_runpod, "_render_runpod_config"))


class RunPodLaunchSettingsTests(unittest.TestCase):
    TRAIN = {"type": "train", "backend": {"name": "musubi-tuner"}}
    RENDER = {"type": "render", "generator": {"name": "comfyui"}}

    def settings(self, run: dict, config: dict) -> dict:
        from kura.run_commands.common import runpod_launch_settings

        return runpod_launch_settings(run, config)

    def test_a_run_gpu_choice_replaces_the_workspace_list_in_its_own_order_for_training_and_render(self) -> None:
        config = {"runpod": {"gpu_type_ids": ["NVIDIA A40", "NVIDIA RTX A5000"], "gpu_type_priority": "availability"}}
        for run in (self.TRAIN, self.RENDER):
            with self.subTest(run=run["type"]):
                settings = self.settings({**run, "compute": {"gpu": ["NVIDIA A40"]}}, config)
                self.assertEqual(settings["gpu_type_ids"], ["NVIDIA A40"])
                self.assertEqual(settings["gpu_type_priority"], "custom")
                untouched = self.settings(run, config)
                self.assertEqual(untouched["gpu_type_ids"], ["NVIDIA A40", "NVIDIA RTX A5000"])
                self.assertEqual(untouched["gpu_type_priority"], "availability")

    def test_ports_follow_one_rule_for_training_and_render(self) -> None:
        # backend_ports.<image> first, then the section's ports, then the image's own defaults.
        cases = [
            ({}, ["8675/http", "22/tcp"], ["22/tcp"]),
            ({"ports": ["22/tcp", "8888/http"]}, ["22/tcp", "8888/http"], ["22/tcp", "8888/http"]),
            ({"ports": ["22/tcp", "8888/http"], "backend_ports": {"ai-toolkit": ["22/tcp"], "comfyui": ["22/tcp", "8188/http"]}}, ["22/tcp"], ["22/tcp", "8188/http"]),
        ]
        for runpod, training_ports, render_ports in cases:
            with self.subTest(runpod=runpod):
                config = {"runpod": {"gpu_type_ids": ["A"], **runpod}}
                self.assertEqual(self.settings({"type": "train", "backend": {"name": "ai-toolkit"}}, config)["ports"], training_ports)
                self.assertEqual(self.settings(self.RENDER, config)["ports"], render_ports)
        musubi = self.settings(self.TRAIN, {"runpod": {"gpu_type_ids": ["A"]}})
        self.assertEqual(musubi["ports"], ["22/tcp"])

    def test_a_render_lays_comfyui_runpod_over_runpod_and_training_does_not(self) -> None:
        config = {"runpod": {"gpu_type_ids": ["A"], "container_disk_gb": 150}, "comfyui": {"runpod": {"container_disk_gb": 80, "ports": ["22/tcp"]}}}
        self.assertEqual(self.settings(self.RENDER, config)["container_disk_gb"], 80)
        self.assertEqual(self.settings(self.TRAIN, config)["container_disk_gb"], 150)

    def test_the_plan_prices_the_gpu_list_the_launch_will_try(self) -> None:
        from kura.run_commands import plan

        run = {"type": "train", "backend": {"name": "musubi-tuner"}, "compute": {"executor": "runpod", "gpu": ["NVIDIA A40"]}}
        config = {"runpod": {"gpu_type_ids": ["NVIDIA A40", "NVIDIA RTX A5000"], "gpu_type_priority": "availability"}}
        measured = {"status": "ok", "candidates": [{"gpu_type_id": "NVIDIA A40", "clouds": [{"cloud_type": "SECURE", "price_per_hour": 0.5, "available": True}]}]}
        with patch.object(plan, "runpod_gpu_availability", return_value=measured) as measure:
            payload = plan._runpod_capacity_payload(run, config)
        assert payload is not None
        settings = measure.call_args.args[0]
        self.assertEqual(settings["gpu_type_priority"], "custom")
        self.assertEqual(settings["gpu_type_ids"], ["NVIDIA A40"])
        self.assertEqual(payload["cost_ceiling"]["hourly_price"], 0.5)


if __name__ == "__main__":
    unittest.main()
