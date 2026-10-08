"""A run whose trainer log stops moving is visible as such, in status and while following, from one rule."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


def _running(directory: str, *, quiet_sec: float) -> Path:
    run_dir = Path(directory) / "runs" / "example"
    (run_dir / "logs").mkdir(parents=True)
    log = run_dir / "logs" / "stdout.log"
    log.write_text("steps: 2/50\n", encoding="utf-8")
    then = time.time() - quiet_sec
    os.utime(log, (then, then))
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    return run_dir


class LogSilenceTests(unittest.TestCase):
    def test_status_says_how_long_a_running_trainer_has_been_silent(self) -> None:
        from kura.cli import cmd_run_status

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60)
            stdout = io.StringIO()
            with patch("kura.cli._run_path", return_value=run_dir), patch("sys.stdout", stdout):
                cmd_run_status(argparse.Namespace(run_id="example"))
            summary = json.loads(stdout.getvalue())["summary"]
            self.assertGreaterEqual(summary["log_silent_minutes"], 19)
            self.assertIn("hung", summary["log_silence"])

    def test_following_says_once_that_the_trainer_has_gone_silent(self) -> None:
        from kura import runner
        from kura.fsio import file_lock

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60)
            root = run_dir.parent.parent
            (run_dir / "realizations").mkdir()
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)

            def launched(**status) -> None:
                (run_dir / "realizations" / "r1.json").write_text(json.dumps({"controlled_by": {"request": request.name, "epoch": 1}}), encoding="utf-8")
                (run_dir / "status.json").write_text(json.dumps({"last_realization": "realizations/r1.json", **status}), encoding="utf-8")

            launched(state="running")
            polls = {"n": 0}

            def sleep(_) -> None:
                polls["n"] += 1
                if polls["n"] == 3:
                    launched(state="failed", publication_state="not-required")

            out = io.StringIO()
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                runner.follow(root, run_dir, request, sleep=sleep, out=out)
            self.assertEqual(out.getvalue().count("no new trainer output"), 1)

    def test_one_rule_decides_silence(self) -> None:
        from kura import cli, runner
        from kura.executors import common

        self.assertIs(cli.log_silence_seconds, common.log_silence_seconds)
        self.assertIs(runner.log_silence_seconds, common.log_silence_seconds)
        self.assertIs(cli.log_silence_notice, common.log_silence_notice)
        self.assertIs(runner.log_silence_notice, common.log_silence_notice)


if __name__ == "__main__":
    unittest.main()
