"""A Pod on which no remote job was ever asked to start holds nothing to collect, so the controller deletes it at once."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

RUNNING = {"state": "running", "host": "runpod", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}


@contextmanager
def _launched_run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield run_dir
        finally:
            os.chdir(previous)


def _controller(run_dir: Path, *, intent: bool) -> tuple[int, bool, str]:
    from tests.test_cli import _run_remote_in_process

    def launched(*_args, **_kwargs):
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(RUNNING), encoding="utf-8")
        return 0

    def ssh_never_ready(*_args, **_kwargs):
        if intent:
            (run_dir / "realizations" / "r1.remote-job-intent.json").write_text(json.dumps({"pid_path": "/tmp/x.pid"}), encoding="utf-8")
        raise ValueError("RunPod pod did not expose SSH before timeout")

    stderr = io.StringIO()
    with patch("kura.run_commands.launch.stage_run", return_value=0), \
         patch("kura.run_commands.launch.launch_run", side_effect=launched), \
         patch("kura.run_commands.launch._runpod_run_over_ssh", side_effect=ssh_never_ready), \
         patch("kura.run_commands.launch._notify"), \
         patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop, \
         patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
        code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1,
                                                         download_interval=1, max_lease="3h", yes=True))
    return code, stop.called, stderr.getvalue()


class UnstartedPodTests(unittest.TestCase):
    def test_a_pod_whose_job_was_never_asked_to_start_is_deleted_at_once(self) -> None:
        with _launched_run() as run_dir:
            code, stopped, said = _controller(run_dir, intent=False)
        self.assertEqual(code, 1)
        self.assertTrue(stopped, said)

    def test_a_pod_whose_job_may_have_started_is_kept_for_the_follower(self) -> None:
        # The start was asked for and its answer is unknown: only the Pod can say, so the runner asks it later.
        with _launched_run() as run_dir:
            code, stopped, _ = _controller(run_dir, intent=True)
        self.assertEqual(code, 1)
        self.assertFalse(stopped)

    def test_the_runner_uses_the_same_rule(self) -> None:
        from kura.executors.common import remote_job_never_requested

        with _launched_run() as run_dir:
            self.assertTrue(remote_job_never_requested(run_dir, "r1"))
            (run_dir / "realizations" / "r1.remote-job-intent.json").write_text("{}", encoding="utf-8")
            self.assertFalse(remote_job_never_requested(run_dir, "r1"))


if __name__ == "__main__":
    unittest.main()
