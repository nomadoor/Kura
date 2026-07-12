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


class MusubiRealSmokeHarnessTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
