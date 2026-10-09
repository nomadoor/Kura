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

from kura.cli import cmd_run_reconcile
from kura.executors.runpod import resolve_runpod_create_intents, stop_runpod, unresolved_create_intents
from kura.fsio import file_lock
from kura.run_commands.plan import stop_run

RID = "20261004-101010-000001"
NAME = f"kura-example-{RID}"
CONFIG = {"gpu_type_ids": ["NVIDIA A40"]}


@contextmanager
def _crashed_launch(previous_pod: str | None = None):
    """A run whose launch wrote its create intent and then died."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\nrunpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "realizations" / f"{RID}.create-intent.json").write_text(
            json.dumps({"kind": "pod_create_intent", "schema_version": 1, "realization_id": RID, "pod_name": NAME, "requested_at": "2026-10-04T10:10:10+09:00", "request": {"name": NAME}}),
            encoding="utf-8",
        )
        status = {"state": "launching", "host": "runpod"}
        if previous_pod:
            status.update({"last_realization": "realizations/old.json", "pod_id": previous_pod})
            (run_dir / "realizations" / "old.json").write_text(json.dumps({"id": "old", "executor": "runpod", "pod": {"id": previous_pod}}), encoding="utf-8")
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


class CreateIntentRecoveryTests(unittest.TestCase):
    def test_a_pod_found_by_name_is_recorded_interrupted_and_stop_deletes_it_and_its_duplicate(self) -> None:
        with _crashed_launch(previous_pod="pod-old") as run_dir:
            pods = [{"id": "pod-b", "name": NAME, "runtime": {"uptimeInSeconds": 5}}, {"id": "pod-a", "name": NAME, "runtime": {"uptimeInSeconds": 90}}]
            with patch("kura.executors.runpod._runpod_pods_named", return_value=pods), patch("kura.executors.runpod._runpod_request") as request:
                lines = resolve_runpod_create_intents(run_dir, CONFIG)
                request.assert_not_called()
                self.assertIn("kura run stop", lines[0])
                self.assertEqual(unresolved_create_intents(run_dir), [])
                status = _status(run_dir)
                self.assertEqual((status["state"], status["pod_id"]), ("interrupted", "pod-a"))
                realization = json.loads((run_dir / "realizations" / f"{RID}.json").read_text(encoding="utf-8"))
                self.assertEqual(realization["duplicate_pod_ids"], ["pod-b"])
                stop_runpod(run_dir, CONFIG)
            deleted = [call.args[1] for call in request.call_args_list]
            self.assertEqual(deleted, ["/pods/pod-a", "/pods/pod-b"])

    def test_no_pod_found_records_a_failed_launch(self) -> None:
        with _crashed_launch() as run_dir:
            with patch("kura.executors.runpod._runpod_pods_named", return_value=[]) as listed:
                resolve_runpod_create_intents(run_dir, CONFIG)
            self.assertEqual(listed.call_count, 2)
            self.assertEqual(_status(run_dir)["state"], "launch_failed")
            self.assertNotIn("pod_id", _status(run_dir))

    def test_reconcile_settles_the_intent_first_even_without_a_realization(self) -> None:
        with _crashed_launch() as run_dir:
            stderr = io.StringIO()
            with patch("kura.executors.runpod._runpod_pods_named", return_value=[]), patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                cmd_run_reconcile(argparse.Namespace(run_id="example"))
            self.assertIn("no Pod named", stderr.getvalue())
            self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_reconcile_waits_while_a_launch_still_holds_its_lock(self) -> None:
        with _crashed_launch() as run_dir:
            (run_dir / ".locks").mkdir()
            stderr = io.StringIO()
            with file_lock(run_dir / ".locks" / "runpod-launch.lock"), patch("kura.executors.runpod._runpod_pods_named") as listed, patch("sys.stderr", stderr):
                self.assertEqual(cmd_run_reconcile(argparse.Namespace(run_id="example")), 1)
            listed.assert_not_called()
            self.assertIn("still creating", stderr.getvalue())

    def test_stop_and_relaunch_refuse_until_the_intent_is_settled(self) -> None:
        from kura.run_commands.launch import launch_run

        with _crashed_launch(previous_pod="pod-old") as run_dir:
            stderr = io.StringIO()
            with patch("kura.executors.runpod._runpod_request") as request, patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                self.assertEqual(stop_run("example"), 1)
                request.assert_not_called()
            self.assertIn("kura run reconcile", stderr.getvalue())
            with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                stop_runpod(run_dir, CONFIG)

    def test_a_recovered_live_pod_blocks_relaunch_until_stopped(self) -> None:
        from kura.executors.runpod import unstopped_recovered_pod

        with _crashed_launch() as run_dir:
            with patch("kura.executors.runpod._runpod_pods_named", return_value=[{"id": "pod-a", "name": NAME, "runtime": None}]):
                resolve_runpod_create_intents(run_dir, CONFIG)
            self.assertEqual(unstopped_recovered_pod(run_dir), "pod-a")
            with patch("kura.executors.runpod._runpod_request"):
                stop_runpod(run_dir, CONFIG)
            self.assertIsNone(unstopped_recovered_pod(run_dir))

    def test_a_null_uptime_does_not_break_recovery(self) -> None:
        with _crashed_launch() as run_dir:
            pods = [{"id": "pod-a", "name": NAME, "runtime": {"uptimeInSeconds": None}}, {"id": "pod-b", "name": NAME, "runtime": {"uptimeInSeconds": 7}}]
            with patch("kura.executors.runpod._runpod_pods_named", return_value=pods):
                resolve_runpod_create_intents(run_dir, CONFIG)
            self.assertEqual(_status(run_dir)["pod_id"], "pod-b")


class UnconfirmedCreateTests(unittest.TestCase):
    def test_a_failed_lookup_still_hands_the_run_over(self) -> None:
        from kura.executors.runpod import RunPodAPIError, launch_runpod_session

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "render"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod GraphQL failed (503): down", status_code=503)), \
                    patch("kura.executors.runpod._runpod_pods_named", side_effect=ValueError("RunPod API is unreachable: down")), patch("kura.executors.runpod.time.sleep"):
                with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                    launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render", dry_run=False, yes=True, max_lease_sec=3600)
            self.assertEqual(_status(run_dir)["state"], "interrupted")

    def test_an_unreadable_create_reply_counts_as_unconfirmed(self) -> None:
        from kura.executors.runpod import _create_outcome_uncertain

        self.assertTrue(_create_outcome_uncertain(ValueError("RunPod API returned invalid JSON")))
        self.assertTrue(_create_outcome_uncertain(ValueError("RunPod GraphQL create response did not contain a Pod")))
        self.assertFalse(_create_outcome_uncertain(ValueError("RunPod GraphQL failed: no capacity")))


class SessionCreateTests(unittest.TestCase):
    def test_a_render_session_hands_over_an_unconfirmed_create(self) -> None:
        from kura.executors.runpod import RunPodAPIError, launch_runpod_session

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "render"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod GraphQL failed (502): bad gateway", status_code=502)) as request, \
                    patch("kura.executors.runpod._runpod_pods_named", return_value=[]), patch("kura.executors.runpod.time.sleep"):
                with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                    launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render", dry_run=False, yes=True, max_lease_sec=3600)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(len(unresolved_create_intents(run_dir)), 1)


if __name__ == "__main__":
    unittest.main()
