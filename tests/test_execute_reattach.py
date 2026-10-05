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

from kura.run_commands import launch


@contextmanager
def _runpod_run(status: dict):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\nrunpod: {api_key_env: RUNPOD_API_KEY}\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({"id": "example", "executor": {"name": "runpod"}, "compute": {"executor": "runpod"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.object(launch, "observe_run", side_effect=lambda run_dir, **_: json.loads((run_dir / "status.json").read_text(encoding="utf-8"))):
                yield run_dir
        finally:
            os.chdir(previous)


class ExecuteReattachTests(unittest.TestCase):
    def test_a_running_job_is_followed_and_collected_without_launching_again(self) -> None:
        with _runpod_run({"state": "running", "pod_id": "pod-1", "remote_job_started_at": "t", "last_realization": "realizations/r1.json"}):
            with patch.object(launch, "follow_running_runpod_job", return_value=0) as follow, \
                    patch.object(launch, "stage_run") as stage, patch.object(launch, "launch_run") as launch_run, \
                    patch.object(launch, "download_with_retries", return_value=0) as download, \
                    patch.object(launch, "stop_run", return_value=0) as stop, \
                    patch.object(launch, "format_run_completion", return_value="done"), patch.object(launch, "_notify"), \
                    redirect_stderr(io.StringIO()), patch("sys.stdout", io.StringIO()):
                code = launch.execute_run("example", yes=True)
        self.assertEqual(code, 0)
        follow.assert_called_once()
        stage.assert_not_called()
        launch_run.assert_not_called()
        download.assert_called_once()
        stop.assert_called_once_with("example")

    def test_a_running_pod_whose_job_never_started_is_refused(self) -> None:
        with _runpod_run({"state": "running", "pod_id": "pod-1"}):
            stderr = io.StringIO()
            with patch.object(launch, "stage_run") as stage, redirect_stderr(stderr):
                code = launch.execute_run("example", yes=True)
        self.assertEqual(code, 1)
        stage.assert_not_called()
        self.assertIn("kura run stop example", stderr.getvalue())

    def test_a_compiled_run_launches_as_before(self) -> None:
        with _runpod_run({"state": "compiled"}):
            with patch.object(launch, "run_remote", return_value=0) as run_remote:
                launch.execute_run("example", yes=True)
        self.assertFalse(run_remote.call_args.kwargs["reattach"])


    def test_a_second_controller_for_the_same_run_is_refused(self) -> None:
        from kura.executors.common import _run_operation_lock

        with _runpod_run({"state": "running", "pod_id": "pod-1", "remote_job_started_at": "t"}) as run_dir:
            stderr = io.StringIO()
            with _run_operation_lock(run_dir, "controller"), patch.object(launch, "follow_running_runpod_job") as follow, redirect_stderr(stderr):
                code = launch.run_remote("example", upload_timeout=1, job_timeout=0, download_attempts=1, download_interval=0, reattach=True)
        self.assertEqual(code, 1)
        follow.assert_not_called()
        self.assertIn("already controlling", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
