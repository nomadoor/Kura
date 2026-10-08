"""A running run that shows no progress is visible as such, from one rule, in status, while following, and in the monitor."""

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


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def _running(directory: str, *, quiet_sec: float, executor: str = "docker") -> Path:
    run_dir = Path(directory) / "runs" / "example"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "realizations").mkdir()
    (run_dir / "run.yaml").write_text(f"id: example\ntype: train\ncompute: {{executor: {executor}}}\n", encoding="utf-8")
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": executor}), encoding="utf-8")
    _age(run_dir / "realizations" / "r1.json", quiet_sec + 60)
    log = run_dir / "logs" / "stdout.log"
    log.write_text("steps: 2/50\n", encoding="utf-8")
    _age(log, quiet_sec)
    # A RunPod sync rewrites status on every pass whether or not the trainer moved.
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
    return run_dir


def _status_summary(run_dir: Path) -> dict:
    from kura.cli import cmd_run_status

    stdout = io.StringIO()
    with patch("kura.cli._run_path", return_value=run_dir), patch("sys.stdout", stdout):
        cmd_run_status(argparse.Namespace(run_id="example"))
    return json.loads(stdout.getvalue())["summary"]


class QuietRunTests(unittest.TestCase):
    def test_status_says_how_long_a_running_run_has_shown_no_progress(self) -> None:
        for executor in ("docker", "runpod"):
            with self.subTest(executor=executor), tempfile.TemporaryDirectory() as directory:
                summary = _status_summary(_running(directory, quiet_sec=20 * 60, executor=executor))
                self.assertGreaterEqual(summary["quiet_minutes"], 19)
                self.assertIn("may be hung", summary["quiet"])

    def test_a_relaunch_counts_from_its_launch_not_from_the_old_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=3 * 3600)
            (run_dir / "realizations" / "r2.json").write_text(json.dumps({"id": "r2"}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r2.json"}), encoding="utf-8")
            summary = _status_summary(run_dir)
            self.assertEqual(summary["quiet_minutes"], 0)
            self.assertNotIn("quiet", summary)

    def test_a_finished_run_is_not_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60)
            (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            self.assertNotIn("quiet_minutes", _status_summary(run_dir))

    def test_the_monitor_marks_the_same_run_stale(self) -> None:
        from kura.monitor import _format_time_cell, _staleness_label, collect_run_summary

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60, executor="runpod")
            summary = collect_run_summary(run_dir.parent.parent, "example")
            self.assertTrue(summary.is_stale)
            self.assertEqual(_staleness_label(summary), " stale")
            self.assertRegex(_format_time_cell(summary), r"^20m\d+s ago stale$")

    def test_the_monitor_and_status_agree_on_a_short_quiet_spell(self) -> None:
        from kura.monitor import collect_run_summary

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=3 * 60)
            self.assertFalse(collect_run_summary(run_dir.parent.parent, "example").is_stale)
            self.assertNotIn("quiet", _status_summary(run_dir))

    def test_a_recorded_launch_phase_is_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60, executor="runpod")
            (run_dir / "realizations" / "r1.phases.jsonl").write_text('{"phase": "upload_started"}\n', encoding="utf-8")
            self.assertNotIn("quiet", _status_summary(run_dir))

    def test_an_empty_realization_reference_is_not_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60)
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": ""}), encoding="utf-8")
            self.assertIn("quiet", _status_summary(run_dir))

    def test_following_says_once_that_the_run_has_gone_quiet(self) -> None:
        from kura import runner
        from kura.fsio import file_lock

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _running(directory, quiet_sec=20 * 60)
            root = run_dir.parent.parent
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            launch = run_dir / "realizations" / "r1.json"

            def launched(**status) -> None:
                launch.write_text(json.dumps({"controlled_by": {"request": request.name, "epoch": 1}}), encoding="utf-8")
                _age(launch, 30 * 60)
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
            self.assertEqual(out.getvalue().count("may be hung"), 1)

    def test_one_rule_decides_quiet(self) -> None:
        from kura import cli, monitor, runner
        from kura.executors import common

        for module in (cli, runner, monitor):
            self.assertIs(module.run_quiet_since, common.run_quiet_since)
        for module in (cli, runner):
            self.assertIs(module.quiet_run_notice, common.quiet_run_notice)


if __name__ == "__main__":
    unittest.main()
