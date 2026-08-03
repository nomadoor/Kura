"""Regression tests for developer real-smoke harnesses."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("musubi_real_smoke", ROOT / "scripts" / "musubi_real_smoke.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SD_SPEC = importlib.util.spec_from_file_location("sd_scripts_real_smoke", ROOT / "scripts" / "sd_scripts_real_smoke.py")
assert SD_SPEC is not None and SD_SPEC.loader is not None
SD_MODULE = importlib.util.module_from_spec(SD_SPEC)
sys.modules[SD_SPEC.name] = SD_MODULE
SD_SPEC.loader.exec_module(SD_MODULE)


class MusubiRealSmokeHarnessTests(unittest.TestCase):
    def test_flux_kontext_smoke_avoids_broken_upstream_fp8_t5_path(self) -> None:
        self.assertNotIn("fp8_t5", MODULE.SPECS["flux_kontext"].extra_override)

    def test_dataset_generation_starts_from_an_empty_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.ensure_generated_dataset(root, "flux-kontext-smoke")
            self.assertTrue((root / "datasets" / "flux2-klein-tiny" / "images" / "00001.png").is_file())
            self.assertTrue((root / "datasets" / "flux-kontext-smoke" / "pose" / "target" / "0001.png").is_file())

    def test_video_generation_keeps_the_container_image_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = __import__("subprocess").CompletedProcess([], 0, "", "")
            with patch.object(MODULE.shutil, "which", return_value="/usr/bin/docker"), patch.object(
                MODULE, "run", return_value=completed
            ) as run:
                MODULE.ensure_generated_dataset(root, "musubi-video-smoke", image="example/musubi:test")
            self.assertIn("example/musubi:test", run.call_args.args[0])

    def test_validate_result_reads_the_common_frozen_command(self) -> None:
        spec = MODULE.SPECS["wan"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "smoke"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "outputs").mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({"state": "completed", "exit_code": 0, "last_step": 1, "total_steps": 1}),
                encoding="utf-8",
            )
            (run_dir / "logs" / "stdout.log").write_text("avr_loss=0.1\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(
                json.dumps({"backend": "musubi-tuner", "cwd": "/opt/musubi-tuner", "argv": ["python", spec.expected_script], "env": {}}),
                encoding="utf-8",
            )
            for index in range(spec.expected_outputs):
                (run_dir / "outputs" / f"result-{index}.safetensors").write_bytes(b"result")

            report = MODULE.validate_result(root, "smoke", spec)

        self.assertTrue(report["checks"]["script_seen"])
        self.assertTrue(report["ok"])

    def test_runpod_result_requires_recovery_and_pod_stop(self) -> None:
        spec = MODULE.SPECS["wan"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "smoke"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "outputs").mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({"state": "completed", "exit_code": 0, "last_step": 1, "total_steps": 1, "host": "runpod", "recovery_required": False}),
                encoding="utf-8",
            )
            (run_dir / "logs" / "stdout.log").write_text("avr_loss=0.1\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(
                json.dumps({"backend": "musubi-tuner", "cwd": "/opt/musubi-tuner", "argv": ["python", spec.expected_script], "env": {}}),
                encoding="utf-8",
            )
            for index in range(spec.expected_outputs):
                (run_dir / "outputs" / f"result-{index}.safetensors").write_bytes(b"result")

            report = MODULE.validate_result(root, "smoke", spec)
            self.assertFalse(report["checks"]["pod_stopped"])
            self.assertFalse(report["ok"])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            status["pod_stopped_at"] = "2026-01-01T00:00:00+00:00"
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            self.assertTrue(MODULE.validate_result(root, "smoke", spec)["ok"])


class SdScriptsRealSmokeHarnessTests(unittest.TestCase):
    def test_generated_lllite_dataset_is_paired_and_identity_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_id = SD_MODULE.ensure_dataset(root, paired=True)
            before = SD_MODULE.dataset_identity(root, dataset_id)
            self.assertTrue((root / "datasets" / dataset_id / "images" / "0001.png").is_file())
            self.assertTrue((root / "datasets" / dataset_id / "conditioning" / "0001.png").is_file())
            self.assertEqual(before, SD_MODULE.dataset_identity(root, dataset_id))

    def test_model_roles_require_every_selector_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.yaml"
            models.write_text("dit: /models/dit.safetensors\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "qwen3, vae"):
                SD_MODULE.load_models(models, SD_MODULE.SPECS["anima-lora"])

    def test_tier1_run_uses_bounded_architecture_specific_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "datasets").mkdir()
            cases = {
                "sd15": ("base: /models/sd15.safetensors\n", [256, 256]),
                "sdxl": ("base: /models/sdxl.safetensors\n", [512, 512]),
                "flux1": ("dit: /models/flux.safetensors\nclip_l: /models/clip.safetensors\nt5xxl: /models/t5.safetensors\nae: /models/ae.safetensors\n", [512, 512]),
                "anima-lora": ("dit: /models/anima.safetensors\nqwen3: /models/qwen.safetensors\nvae: /models/vae.safetensors\n", [512, 512]),
                "anima-lllite": ("dit: /models/anima.safetensors\nqwen3: /models/qwen.safetensors\nvae: /models/vae.safetensors\n", [512, 512]),
            }
            for selector, (models_text, resolution) in cases.items():
                models = root / f"{selector}.yaml"
                models.write_text(models_text, encoding="utf-8")
                run_id, _ = SD_MODULE.write_run(root, selector, SD_MODULE.SPECS[selector], models_file=models, executor="docker", gpu="gpu")
                run_yaml = __import__("yaml").safe_load((root / "runs" / run_id / "run.yaml").read_text(encoding="utf-8"))
                config = run_yaml["backend"]["config"]
                self.assertEqual(config["dataset_config"]["general"]["resolution"], resolution)
                if selector == "flux1":
                    self.assertTrue(config["fp8_base"])
                    self.assertEqual(config["blocks_to_swap"], 16)
                    self.assertEqual(config["timestep_sampling"], "flux_shift")
                    self.assertEqual(config["network_dim"], 16)
                if selector == "sdxl":
                    self.assertEqual(config["network_dim"], 8)
                    self.assertEqual(config["network_alpha"], 4)
                    self.assertTrue(config["network_train_unet_only"])
                if selector == "anima-lora":
                    self.assertEqual(config["learning_rate"], 0.0001)
                    self.assertEqual(config["network_dim"], 8)
                    self.assertTrue(config["network_train_unet_only"])
                    self.assertEqual(config["timestep_sampling"], "sigmoid")
                if selector == "anima-lllite":
                    self.assertNotIn("network_dim", config)
                    self.assertEqual(config["learning_rate"], 0.00005)
                    self.assertEqual(config["lllite_mlp_dim"], 64)
                    self.assertEqual(config["timestep_sampling"], "shift")
                    self.assertEqual(config["discrete_flow_shift"], 3.0)

    def test_lllite_validation_checks_real_metadata_and_dataset_identity(self) -> None:
        spec = SD_MODULE.SPECS["anima-lllite"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_id = SD_MODULE.ensure_dataset(root, paired=True)
            run_dir = root / "runs" / "smoke"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "outputs").mkdir()
            (run_dir / "realizations").mkdir()
            observation = "realizations/local.observed.json"
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "exit_code": 0, "last_step": 1, "total_steps": 1, "host": "docker", "last_observation": observation}), encoding="utf-8")
            (run_dir / observation).write_text(json.dumps({"state": "completed", "exit_code": 0, "container_id": "container"}), encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(spec.expected_script, encoding="utf-8")
            (run_dir / "realizations" / "dataset-before.json").write_text(json.dumps(SD_MODULE.dataset_identity(root, dataset_id)), encoding="utf-8")
            header = {"__metadata__": {"lllite.version": "2"}, "lllite_conditioning1.conv1.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            encoded = json.dumps(header).encode("utf-8")
            import struct
            (run_dir / "outputs" / "result.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 0.0))
            self.assertTrue(SD_MODULE.validate_result(root, "smoke", dataset_id, spec)["ok"])
            (run_dir / observation).unlink()
            report = SD_MODULE.validate_result(root, "smoke", dataset_id, spec)
            self.assertFalse(report["checks"]["docker_terminal_observed"])
            self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
