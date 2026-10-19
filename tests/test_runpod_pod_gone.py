"""A RunPod Pod is recorded as gone only when RunPod confirms it twice: its lookup says not found, and the account's Pods do not include it."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.executors import runpod
from kura.executors.runpod import RunPodAPIError

NOT_FOUND = RunPodAPIError("RunPod Pod not found", status_code=404)
RUNNING = {"state": "running", "host": "runpod", "pod_id": "pod-1", "last_realization": "realizations/r1.json",
           "remote_job_started_at": "t"}


@contextmanager
def _run(status: dict = RUNNING):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1", "name": "kura-example-r1"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), patch("kura.executors.runpod.time.sleep"):
                yield run_dir
        finally:
            os.chdir(previous)


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def _listed(*pod_ids: str):
    return patch("kura.executors.runpod._runpod_pods", return_value=[{"id": pod_id, "name": "x"} for pod_id in pod_ids])


class PodGoneTests(unittest.TestCase):
    def test_a_not_found_the_account_still_lists_is_never_recorded_by_any_caller(self) -> None:
        for source in ("explicit", "automatic"):
            with self.subTest(source=source), _run() as run_dir:
                with patch("kura.executors.runpod._runpod_request", side_effect=NOT_FOUND), _listed("pod-1"):
                    with self.assertRaises(ValueError):
                        runpod.reconcile_runpod(run_dir, source=source)
                status = _status(run_dir)
                self.assertEqual(status["state"], "running")
                self.assertNotIn("pod_missing_at", status)

    def test_a_confirmed_missing_pod_is_recorded_by_every_caller(self) -> None:
        for source in ("explicit", "automatic"):
            with self.subTest(source=source), _run() as run_dir:
                with patch("kura.executors.runpod._runpod_request", side_effect=NOT_FOUND), _listed("other-pod"):
                    status = runpod.reconcile_runpod(run_dir, source=source)
                self.assertEqual(status["state"], "interrupted")
                self.assertTrue(status["pod_missing_at"])

    def test_a_failed_delete_counts_as_done_only_when_the_pod_is_confirmed_gone(self) -> None:
        for listed, stopped in ((("pod-1",), False), ((), True)):
            with self.subTest(listed=listed), _run() as run_dir:
                with patch("kura.executors.runpod._runpod_request", side_effect=NOT_FOUND), _listed(*listed):
                    if stopped:
                        runpod.stop_runpod(run_dir)
                    else:
                        with self.assertRaises(ValueError):
                            runpod.stop_runpod(run_dir)
                self.assertEqual(bool(_status(run_dir).get("pod_stopped_at")), stopped)

    def test_the_follower_does_not_end_a_run_on_one_flaky_not_found(self) -> None:
        from kura import runner

        with _run({"state": "compiled"}) as run_dir:
            request = runner.write_launch_request(run_dir, executor="runpod", extra={
                "options": {"max_lease": "3h"}, "runpod_config": {"gpu_type_ids": ["A"]}, "billing_confirmed_at": "2026-10-19T00:00:00+00:00"})
            runner.claim_request(request, 1)
            runner._first_attempt(request)
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"},
                                                                          "controlled_by": {"request": request.name}}), encoding="utf-8")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps(RUNNING), encoding="utf-8")
            with patch("kura.executors.runpod._runpod_request", side_effect=NOT_FOUND), _listed("pod-1"), \
                 patch("kura.run_commands.launch._run_remote_locked") as remote:
                self.assertEqual(runner.work(run_dir.parent.parent, "example", request.name), 1)
            remote.assert_not_called()
            status = _status(run_dir)
            self.assertEqual(status["state"], "running")
            self.assertNotIn("pod_missing_at", status)

    def test_execute_reports_a_confirmed_missing_pod_instead_of_following_it(self) -> None:
        import io
        from contextlib import redirect_stderr

        from kura.run_commands import launch

        with _run() as run_dir:
            (run_dir / "realizations" / "r1.phases.jsonl").write_text(json.dumps({"phase": "remote_job_started", "at": "t"}) + "\n", encoding="utf-8")
            stderr = io.StringIO()
            with patch("kura.executors.runpod._runpod_request", side_effect=NOT_FOUND), _listed(), redirect_stderr(stderr):
                with self.assertRaisesRegex(ValueError, "outputs are gone"):
                    launch._running_remote_job("example")
            self.assertTrue(_status(run_dir)["pod_missing_at"])


if __name__ == "__main__":
    unittest.main()
