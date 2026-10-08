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


if __name__ == "__main__":
    unittest.main()
