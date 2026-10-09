"""Kura never starts a RunPod Pod without its self-delete timer: a maximum lease of zero is refused everywhere."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MaxLeaseRequiredTests(unittest.TestCase):
    def test_one_reader_turns_a_max_lease_into_positive_seconds(self) -> None:
        from kura.executors.common import DEFAULT_MAX_LEASE_SEC
        from kura.run_commands.plan import max_lease_seconds

        self.assertEqual(max_lease_seconds(None), DEFAULT_MAX_LEASE_SEC)
        self.assertEqual(max_lease_seconds("3h"), 3 * 3600)
        self.assertEqual(max_lease_seconds(600), 600)
        for value in (0, "0", "0s", "0h", -5, ""):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "--max-lease must be a positive duration"):
                max_lease_seconds(value)

    def test_training_refuses_a_zero_lease_before_staging_or_creating_a_pod(self) -> None:
        from kura.run_commands import launch

        for value in ("0", "0s"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory) / "runs" / "example"
                run_dir.mkdir(parents=True)
                with (
                    patch.object(launch, "_run_path", return_value=run_dir),
                    patch.object(launch, "stage_run") as stage,
                    patch.object(launch, "launch_runpod") as create,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    code = launch._run_remote_locked(
                        "example", upload_timeout=1, job_timeout=None, download_attempts=1, download_interval=1, max_lease=value, yes=True,
                    )
                self.assertEqual(code, 1)
                stage.assert_not_called()
                create.assert_not_called()
                self.assertIn("--max-lease must be a positive duration", stderr.getvalue())

    def test_the_launch_check_refuses_a_zero_lease_before_billing_confirmation(self) -> None:
        from kura.run_commands import launch

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            with (
                patch.object(launch, "_run_path", return_value=run_dir),
                patch.object(launch, "_load_yaml", return_value={"compute": {"executor": "runpod"}}),
                patch.object(launch, "confirm_runpod_billing") as confirm,
                patch.object(launch, "launch_runpod") as create,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                code = launch.launch_run("example", executor="runpod", dry_run=False, check_only=True, max_lease="0s")
            self.assertEqual(code, 1)
            confirm.assert_not_called()
            create.assert_not_called()
            self.assertIn("--max-lease must be a positive duration", stderr.getvalue())

    def test_a_runpod_render_refuses_a_zero_lease_before_any_request_or_pod(self) -> None:
        from kura import runner
        from kura.run_commands import launch

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            with (
                patch.object(launch, "_run_path", return_value=run_dir),
                patch.object(launch, "_workspace", return_value=Path(directory)),
                patch.object(launch, "_load_yaml", return_value={"type": "render", "executor": {"name": "runpod"}}),
                patch.object(launch, "launch_render_runpod") as create,
                patch.object(runner, "write_launch_request") as request,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                through_runner = launch._launch_render_through_runner("example", follow=False, runpod={"image": None, "yes": True, "max_lease": "0"})
                direct = launch.launch_run("example", executor="runpod", dry_run=False, max_lease="0s")
            self.assertEqual((through_runner, direct), (1, 1))
            create.assert_not_called()
            request.assert_not_called()
            self.assertIn("--max-lease must be a positive duration", stderr.getvalue())

    def test_an_old_runner_request_without_a_lease_runs_with_the_default(self) -> None:
        from kura import runner
        from kura.executors.common import DEFAULT_MAX_LEASE_SEC

        with patch("kura.run_commands.launch._run_remote_locked", return_value=0) as remote:
            runner._remote(Path("runs/example"), Path("requests/r.json"), {"options": {"max_lease": "0"}}, reattach=False)
        self.assertEqual(remote.call_args.kwargs["max_lease"], DEFAULT_MAX_LEASE_SEC)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            request = run_dir / "r.json"
            request.write_text(json.dumps({"billing_confirmed_at": "2026-10-10T00:00:00+09:00", "options": {"max_lease_sec": 0}}), encoding="utf-8")
            with (
                patch("kura.run_commands.render_runpod.launch_render_runpod") as render,
                patch.object(runner, "_delete_pod", return_value=True),
                patch.object(runner, "_status", return_value={}),
                patch.object(runner, "_record_request_outcome"),
            ):
                runner._work_render_runpod(Path(directory), run_dir, request)
            self.assertEqual(render.call_args.kwargs["max_lease_sec"], DEFAULT_MAX_LEASE_SEC)

    def test_every_pod_creation_path_requires_a_lease(self) -> None:
        import inspect

        from kura.executors import runpod

        # No caller can create a Pod without the start-time self-delete by leaving the lease out.
        for function in (runpod.launch_runpod, runpod._pod_start_script, runpod._confirm_runpod_launch):
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters["max_lease_sec"]
                self.assertIs(parameter.default, inspect.Parameter.empty)
                self.assertEqual(parameter.annotation, "int")
        self.assertIn("kura_lease_initial", runpod._pod_start_script("true", max_lease_sec=60, log_path="/tmp/log"))

    def test_both_runner_paths_read_an_old_request_lease_through_one_reader(self) -> None:
        from kura import runner
        from kura.executors.common import DEFAULT_MAX_LEASE_SEC
        from kura.run_commands import plan

        cases = ((None, DEFAULT_MAX_LEASE_SEC), (0, DEFAULT_MAX_LEASE_SEC), ("0", DEFAULT_MAX_LEASE_SEC), (3600, 3600), ("2h", 7200))
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(plan.request_max_lease_seconds(value), expected)
                self.assertEqual(self._training_lease(runner, value), expected)
                self.assertEqual(self._render_lease(runner, value), expected)
        for value in ("abc", -5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    plan.request_max_lease_seconds(value)
                with self.assertRaises(ValueError):
                    self._training_lease(runner, value)
                with self.assertRaises(ValueError):
                    self._render_lease(runner, value)
        with patch.object(plan, "request_max_lease_seconds", return_value=99) as reader:
            self.assertEqual((self._training_lease(runner, "1h"), self._render_lease(runner, 3600)), (99, 99))
        self.assertEqual(reader.call_count, 2)

    def _training_lease(self, runner, value):
        options = {} if value is None else {"max_lease": value}
        with patch("kura.run_commands.launch._run_remote_locked", return_value=0) as remote:
            runner._remote(Path("runs/example"), Path("requests/r.json"), {"options": options}, reattach=False)
        return remote.call_args.kwargs["max_lease"]

    def _render_lease(self, runner, value):
        options = {} if value is None else {"max_lease_sec": value}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            request = run_dir / "r.json"
            request.write_text(json.dumps({"billing_confirmed_at": "2026-10-10T00:00:00+09:00", "options": options}), encoding="utf-8")
            with (
                patch("kura.run_commands.render_runpod.launch_render_runpod") as render,
                patch.object(runner, "_delete_pod", return_value=True),
                patch.object(runner, "_status", return_value={}),
                patch.object(runner, "_record_request_outcome"),
            ):
                runner._work_render_runpod(Path(directory), run_dir, request)
            return render.call_args.kwargs["max_lease_sec"]


if __name__ == "__main__":
    unittest.main()
