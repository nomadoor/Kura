"""One rule decides that a RunPod Pod is gone: its lookup answers not found (status 404).

Every caller (reconcile, stop, `kura run execute`'s reattach check, the runner's
follower) asks `runpod._runpod_pod_or_none`; any other answer is not "gone".
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from kura.executors import runpod
from kura.executors.runpod import RunPodAPIError

NOT_FOUND = RunPodAPIError("RunPod Pod not found", status_code=404)
# Answers that are not "gone": a server error, and a GraphQL error whose text merely mentions a 404.
NOT_GONE = (
    RunPodAPIError("RunPod GraphQL failed (500): busy", status_code=500),
    RunPodAPIError("RunPod GraphQL failed: upstream returned 404 pod not found", status_code=400),
    ValueError("RunPod API is unreachable: timed out"),
)
RUNNING = {"state": "running", "host": "runpod", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}


@contextmanager
def _run(status: dict = RUNNING):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "realizations" / "r1.json").write_text(
            json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1", "name": "kura-example-r1"}}), encoding="utf-8")
        (run_dir / "realizations" / "r1.phases.jsonl").write_text(
            json.dumps({"phase": "remote_job_started", "at": "t"}) + "\n", encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                yield run_dir
        finally:
            os.chdir(previous)


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def _api(lookup: Exception | dict, delete: Exception | None = None):
    """RunPod answering a Pod lookup with `lookup` and a delete with `delete` (None: accepted)."""

    def answer(method: str, path: str, *_args, **_kwargs):
        if method == "GET":
            if isinstance(lookup, Exception):
                raise lookup
            return lookup
        if method == "DELETE":
            if delete is not None:
                raise delete
            return {}
        raise AssertionError(f"unexpected RunPod request: {method} {path}")

    return patch("kura.executors.runpod._runpod_request", side_effect=answer)


def _owner():
    """Spy on the one owner of "the Pod is gone"."""
    return patch("kura.executors.runpod._runpod_pod_or_none", wraps=runpod._runpod_pod_or_none)


class OwnerTests(unittest.TestCase):
    def test_only_a_404_lookup_is_gone(self) -> None:
        with _api(NOT_FOUND):
            self.assertIsNone(runpod._runpod_pod_or_none("pod-1", "key"))
        with _api({"id": "pod-1", "desiredStatus": "RUNNING"}):
            self.assertEqual(runpod._runpod_pod_or_none("pod-1", "key")["id"], "pod-1")
        for error in NOT_GONE:
            with self.subTest(error=str(error)), _api(error), self.assertRaises(ValueError):
                runpod._runpod_pod_or_none("pod-1", "key")


class ReconcileTests(unittest.TestCase):
    def test_a_404_is_recorded_as_gone_by_every_source(self) -> None:
        for source in ("explicit", "automatic"):
            with self.subTest(source=source), _run() as run_dir:
                with _api(NOT_FOUND), _owner() as owner:
                    status = runpod.reconcile_runpod(run_dir, source=source)
                owner.assert_called_once()
                self.assertEqual(status["state"], "interrupted")
                self.assertTrue(status["pod_missing_at"])
                self.assertEqual(runpod.pods_left(run_dir, status), [])

    def test_anything_else_is_not_gone_for_any_source(self) -> None:
        for source in ("explicit", "automatic"):
            for error in NOT_GONE:
                with self.subTest(source=source, error=str(error)), _run() as run_dir:
                    with _api(error), _owner() as owner, self.assertRaises(ValueError):
                        runpod.reconcile_runpod(run_dir, source=source)
                    owner.assert_called_once()
                    status = _status(run_dir)
                    self.assertEqual(status["state"], "running")
                    self.assertNotIn("pod_missing_at", status)


class StopTests(unittest.TestCase):
    def test_a_failed_delete_of_a_pod_whose_lookup_is_404_is_already_gone(self) -> None:
        with _run() as run_dir:
            with _api(NOT_FOUND, delete=RunPodAPIError("RunPod GraphQL failed: no such pod", status_code=400)), _owner() as owner:
                runpod.stop_runpod(run_dir)
            owner.assert_called_once()
            self.assertTrue(_status(run_dir)["pod_stopped_at"])
            stop = json.loads(next((run_dir / "realizations").glob("*.stop-*.json")).read_text(encoding="utf-8"))
            self.assertEqual([target["result"] for target in stop["targets"]], ["already_gone"])

    def test_a_failed_delete_is_never_gone_on_its_error_text_alone(self) -> None:
        delete_errors = (RunPodAPIError("RunPod GraphQL failed: 404 pod not found", status_code=400),)
        lookups = ({"id": "pod-1", "desiredStatus": "RUNNING"}, *NOT_GONE)
        for delete in delete_errors:
            for lookup in lookups:
                with self.subTest(lookup=str(lookup)), _run() as run_dir:
                    with _api(lookup, delete=delete), _owner() as owner, self.assertRaises(ValueError) as raised:
                        runpod.stop_runpod(run_dir)
                    owner.assert_called_once()
                    self.assertIs(raised.exception, delete)  # the delete's own error, not the lookup's
                    self.assertNotIn("pod_stopped_at", _status(run_dir))
                    stop = json.loads(next((run_dir / "realizations").glob("*.stop-*.json")).read_text(encoding="utf-8"))
                    self.assertEqual((stop["outcome"], [target["result"] for target in stop["targets"]]), ("failed", ["failed"]))

    def test_an_accepted_delete_does_not_look_the_pod_up(self) -> None:
        with _run() as run_dir:
            with _api(AssertionError("no lookup expected")), _owner() as owner:
                runpod.stop_runpod(run_dir)
            owner.assert_not_called()
            self.assertTrue(_status(run_dir)["pod_stopped_at"])


class FollowerTests(unittest.TestCase):
    def _request(self, run_dir: Path):
        from kura import runner

        request = runner.write_launch_request(run_dir, executor="runpod", extra={
            "options": {"max_lease": "3h"}, "runpod_config": {"gpu_type_ids": ["A"]}, "billing_confirmed_at": "2026-10-19T00:00:00+00:00"})
        runner.claim_request(request, 1)
        runner._first_attempt(request)
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({
            "id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}, "controlled_by": {"request": request.name}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(RUNNING), encoding="utf-8")
        return request

    def test_a_404_ends_the_run_interrupted_without_following(self) -> None:
        from kura import runner

        with _run({"state": "compiled"}) as run_dir:
            request = self._request(run_dir)
            with _api(NOT_FOUND), _owner() as owner, patch("kura.run_commands.launch._run_remote_locked") as remote, \
                    patch.object(runner, "_notify_text"):
                self.assertEqual(runner.work(run_dir.parent.parent, "example", request.name), 0)
            owner.assert_called_once()
            remote.assert_not_called()
            status = _status(run_dir)
            self.assertEqual(status["state"], "interrupted")
            self.assertTrue(status["pod_missing_at"])

    def test_anything_else_keeps_the_run_and_does_not_follow_yet(self) -> None:
        from kura import runner

        for error in NOT_GONE:
            with self.subTest(error=str(error)), _run({"state": "compiled"}) as run_dir:
                request = self._request(run_dir)
                with _api(error), _owner() as owner, patch("kura.run_commands.launch._run_remote_locked") as remote, \
                        patch.object(runner, "_notify_text"):
                    self.assertEqual(runner.work(run_dir.parent.parent, "example", request.name), 1)
                owner.assert_called_once()
                remote.assert_not_called()
                status = _status(run_dir)
                self.assertEqual(status["state"], "running")
                self.assertNotIn("pod_missing_at", status)


if __name__ == "__main__":
    unittest.main()
