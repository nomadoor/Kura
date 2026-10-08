"""Both executors count a finished run's steps with the same function, from the log when it says and from the run's recipe when it does not."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml


def _run(directory: str, stdout: str) -> Path:
    run_dir = Path(directory) / "runs" / "example"
    (run_dir / "resolved").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({
        "id": "example", "type": "train", "backend": {"name": "musubi-tuner", "config": {}}, "recipe": {"steps": 10, "seed": 1},
    }), encoding="utf-8")
    (run_dir / "logs" / "stdout.log").write_text(stdout, encoding="utf-8")
    return run_dir


class ProgressParityTests(unittest.TestCase):
    def test_runpod_collection_uses_the_shared_progress_rule(self) -> None:
        from kura.executors import common
        from kura.run_commands import runpod_ssh

        self.assertIs(runpod_ssh._apply_stdout_progress, common._apply_stdout_progress)

    def test_a_completed_run_counts_its_steps_from_the_log_or_else_from_its_recipe(self) -> None:
        from kura.executors.common import _apply_stdout_progress

        for label, stdout in (("log says", "steps: 100%|##########| 10/10 [00:10<00:00, 1.0it/s, avr_loss=0.5]\n"), ("log silent", "")):
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                run_dir = _run(directory, stdout)
                status = {"state": "completed", "last_realization": "realizations/r1.json"}
                _apply_stdout_progress(run_dir, status, state="completed")
                self.assertEqual((status.get("last_step"), status.get("total_steps")), (10, 10))


    def test_speed_comes_only_from_training_progress_lines(self) -> None:
        # Model downloads and latent caching print their own it/s; they are not training speed.
        from kura.executors.common import _stdout_progress

        download = "model.safetensors:  45%|####5     | 6.3G/14.0G [01:02<01:10, 110MB/s]\nFetching 17 files:  47%|####7     | 8/17 [00:05<00:01,  1.51it/s]\n"
        training = "steps:  52%|#####2    | 26/50 [00:40<00:30,  1.20s/it, avr_loss=0.08]\n"
        caching = "caching latents: 100%|##########| 3/3 [00:02<00:00,  8.11it/s]\n"
        for label, stdout, expected in (
            ("download only", download, None),
            ("training, then caching", training + caching, 1.20),
        ):
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                run_dir = _run(directory, stdout)
                self.assertEqual(_stdout_progress(run_dir)[2], expected)

if __name__ == "__main__":
    unittest.main()
