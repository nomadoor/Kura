"""Starting a training run takes one confirmation, locally as on RunPod: `y` at a terminal, or `--yes`."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import yaml

from kura import runner
from kura.run_commands import launch


@contextmanager
def _docker_run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({"id": "example", "compute": {"executor": "docker"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield run_dir
        finally:
            os.chdir(previous)


def _start(*, yes: bool, tty: bool, answer: str = "") -> tuple[int, str]:
    """Start the compiled local run without following it; the checks before the confirmation pass."""
    stderr = io.StringIO()
    with patch.object(launch, "launch_run", return_value=0), patch.object(runner, "env_only_secrets", return_value=[]), \
            patch.object(runner, "ensure_runner", return_value=False), patch.object(runner, "await_claim", return_value=True), \
            patch("sys.stdin.isatty", return_value=tty), patch("sys.stdin.readline", return_value=answer), redirect_stderr(stderr):
        code = launch._launch_docker_through_runner("example", image=None, follow=False, yes=yes)
    return code, stderr.getvalue()


def _requests(run_dir: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "requests").glob("*.launch.json"))] if (run_dir / "requests").is_dir() else []


class LocalLaunchConfirmationTests(unittest.TestCase):
    def test_without_a_terminal_a_local_launch_needs_yes(self) -> None:
        with _docker_run() as run_dir:
            code, text = _start(yes=False, tty=False)
            self.assertEqual(code, 1)
            self.assertIn("--yes", text)
            self.assertIn("kura run plan", text)
            self.assertEqual(_requests(run_dir), [])

    def test_yes_starts_it_and_the_request_records_when(self) -> None:
        with _docker_run() as run_dir:
            code, _ = _start(yes=True, tty=False)
            self.assertEqual(code, 0)
            (request,) = _requests(run_dir)
            self.assertTrue(request["confirmed_at"])
            self.assertNotIn("confirmed", request)

    def test_at_a_terminal_only_y_starts_it(self) -> None:
        for answer, expected in (("y\n", 0), ("n\n", 1), ("", 1)):
            with self.subTest(answer=answer), _docker_run() as run_dir:
                code, text = _start(yes=False, tty=True, answer=answer)
                self.assertEqual(code, expected)
                self.assertIn("Start training run example on this machine?", text)
                self.assertEqual(len(_requests(run_dir)), 1 if expected == 0 else 0)
                if expected:
                    # An agent host with a pseudo-terminal reads nothing; it is told the next step.
                    self.assertIn("--yes", text)

    def test_following_a_launch_in_progress_asks_nothing(self) -> None:
        with _docker_run() as run_dir:
            runner.write_launch_request(run_dir, executor="docker", confirmed_at="2026-10-27T10:00:00+09:00")
            code, text = _start(yes=False, tty=False)
            self.assertEqual(code, 0)
            self.assertNotIn("--yes", text)
            self.assertEqual(len(_requests(run_dir)), 1)


class OneConfirmationTests(unittest.TestCase):
    def test_local_and_runpod_launches_take_the_same_confirmation(self) -> None:
        from kura.executors import common, runpod

        self.assertIs(launch.confirm_launch, common.confirm_launch)
        self.assertIs(runpod.confirm_launch, common.confirm_launch)

    def test_a_runpod_launch_without_a_terminal_still_needs_yes(self) -> None:
        from kura.executors import runpod

        with patch("sys.stdin.isatty", return_value=False), patch.object(runpod, "runpod_gpu_availability", side_effect=AssertionError("measured")):
            with self.assertRaisesRegex(ValueError, "--yes"):
                runpod._confirm_runpod_launch({}, {"gpu_type_ids": ["A"], "gpu_count": 1, "cloud_types": ["SECURE"]}, yes=False, max_lease_sec=3600)

    def test_a_request_an_older_kura_confirmed_for_billing_reads_as_confirmed(self) -> None:
        from kura.executors.common import request_confirmed_at

        self.assertEqual(request_confirmed_at({"billing_confirmed_at": "t1"}), "t1")
        self.assertEqual(request_confirmed_at({"confirmed_at": "t2"}), "t2")
        self.assertIsNone(request_confirmed_at({"confirmed": True}))


if __name__ == "__main__":
    unittest.main()
