"""A training run starts one way, `kura run execute`, so no other command can start it with other lifecycle options."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from kura.cli import main


def _parse(*argv: str) -> tuple[int | None, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), patch("sys.argv", ["kura", *argv]), \
            patch("kura.cli._load_secrets"), patch("kura.cli._refresh_managed_files"):
        try:
            main()
        except SystemExit as exit:
            return exit.code, out.getvalue()
    return None, out.getvalue()


class OneWayToStartTests(unittest.TestCase):
    def test_the_other_training_entry_points_are_gone(self) -> None:
        for command in ("remote", "launch", "stage", "upload"):
            with self.subTest(command=command):
                code, output = _parse("run", command, "some-run")
                self.assertEqual(code, 2)
                self.assertIn("invalid choice", output)

    def test_execute_has_no_review_hold(self) -> None:
        code, output = _parse("run", "execute", "--help")
        self.assertEqual(code, 0)
        self.assertNotIn("--hold-for", output)
        self.assertIn("--max-lease", output)

    def test_renders_still_start_with_render_launch(self) -> None:
        code, output = _parse("render", "launch", "--help")
        self.assertEqual(code, 0)
        self.assertIn("--executor", output)


    def test_a_runpod_run_waits_for_its_gpu_unless_it_says_otherwise(self) -> None:
        from kura.run_envelope import capacity_policy

        self.assertEqual(capacity_policy({"compute": {"executor": "runpod"}}), {"mode": "wait", "timeout": "24h", "poll_interval": "30s"})
        self.assertEqual(capacity_policy({"compute": {"capacity": {"mode": "immediate"}}})["mode"], "immediate")
        self.assertEqual(capacity_policy({"compute": {"capacity": {"timeout": "6h"}}})["timeout"], "6h")

    def test_one_rule_decides_the_capacity_default(self) -> None:
        from kura import cli, run_envelope
        from kura.run_commands import launch, plan

        for module in (cli, launch, plan):
            self.assertIs(module.capacity_policy, run_envelope.capacity_policy)

    def test_a_request_written_with_the_review_hold_still_runs(self) -> None:
        from pathlib import Path

        from kura import runner

        with patch("kura.run_commands.launch._run_remote_locked", return_value=0) as remote:
            runner._remote(Path("runs/example"), Path("requests/r.json"), {"options": {"max_lease": "3h", "hold_for": "30m", "notify_repeat_interval": "10m"}}, reattach=False)
        self.assertNotIn("hold_for", remote.call_args.kwargs)
        self.assertNotIn("notify_repeat_interval", remote.call_args.kwargs)
        self.assertEqual(remote.call_args.kwargs["max_lease"], 3 * 3600)

    def test_a_run_waiting_for_its_gpu_can_be_staged_and_launched_again(self) -> None:
        # A follower that stops mid-wait leaves the run queued; the runner resumes it by staging and launching.
        from kura.executors.common import can_start

        waiting = {"state": "queued", "capacity_wait": {"started_at": "2026-10-09T00:00:00+09:00"}}
        self.assertTrue(can_start(waiting))
        self.assertTrue(can_start({"state": "compiled"}))
        self.assertFalse(can_start({"state": "queued"}))
        self.assertFalse(can_start({"state": "running"}))
        self.assertFalse(can_start({"state": "completed"}))

    def test_stage_and_launch_ask_the_same_rule(self) -> None:
        from kura.executors import common
        from kura.run_commands import launch, plan

        self.assertIs(plan.can_start, common.can_start)
        self.assertIs(launch.can_start, common.can_start)

    def test_a_capacity_mode_that_is_not_a_word_is_a_clear_error(self) -> None:
        from kura.run_envelope import capacity_policy

        with self.assertRaisesRegex(ValueError, "compute.capacity.mode must be immediate or wait"):
            capacity_policy({"compute": {"capacity": {"mode": ["wait"]}}})

if __name__ == "__main__":
    unittest.main()
