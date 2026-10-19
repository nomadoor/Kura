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


def _write_upload_stage(run_dir: Path) -> None:
    """A compiled run with the upload stage `kura run execute` writes before it launches."""
    # Kept outside realizations/ so the tests' realization listings stay as they were.
    (run_dir / "transfer").mkdir(parents=True, exist_ok=True)
    (run_dir / "transfer" / "stage.json").write_text(json.dumps(
        {"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled", "last_stage": "transfer/stage.json"}), encoding="utf-8")

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


CAPACITY_REFUSED = "RunPod GraphQL failed: There are no longer any instances available with the requested specifications"
AVAILABLE = {"status": "ok", "checked_at": "2026-10-04T10:10:10+09:00", "gpu_count": 1, "candidates": [
    {"gpu_type_id": "NVIDIA A40", "display_name": "NVIDIA A40", "memory_gb": 48,
     "clouds": [{"cloud_type": "COMMUNITY", "stock_status": "Low", "available": True, "price_per_hour": 0.4, "available_gpu_counts": [1]}]}]}


@contextmanager
def _crash_after_capacity_wait():
    """A compiled run whose launch waited for capacity, created a Pod, recorded it, and died before status followed."""
    import yaml

    from kura.executors.runpod import RunPodAPIError, launch_runpod

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        _write_upload_stage(run_dir)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({
            "id": "example", "type": "train", "compute": {"executor": "runpod", "gpu": ["NVIDIA A40"]},
            "backend": {"name": "ai-toolkit", "config": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
        }), encoding="utf-8")
        (root / "workspace.yaml").write_text(yaml.safe_dump({
            "runpod": {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"], "cloud_types": ["COMMUNITY"]}, "docker": {}, "images": {"ai-toolkit": "local"},
        }), encoding="utf-8")
        replies = [RunPodAPIError(CAPACITY_REFUSED, status_code=400), {"id": "pod-1", "name": "created", "desiredStatus": "RUNNING"}]

        def create(*_args, **_kwargs):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), patch("kura.executors.runpod.time.sleep"):
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=AVAILABLE), \
                        patch("kura.executors.runpod._runpod_request", side_effect=create), \
                        patch("kura.executors.runpod.point_status_at_launch", side_effect=KeyboardInterrupt):
                    with unittest.TestCase().assertRaises(KeyboardInterrupt):
                        launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                                      image="registry/image:tag", config={"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"], "cloud_types": ["COMMUNITY"]},
                                      wait_for_capacity_sec=60, capacity_poll_interval_sec=5, yes=True)
                yield root, run_dir
        finally:
            os.chdir(previous)


class CrashAfterCapacityWaitTests(unittest.TestCase):
    """The records say a Pod exists while status still says the run waits for capacity."""

    def test_the_pod_the_records_name_counts_as_unresolved_while_status_says_queued(self) -> None:
        from kura.runner import pod_may_still_bill

        with _crash_after_capacity_wait() as (_root, run_dir):
            status = _status(run_dir)
            self.assertEqual(status["state"], "queued")
            self.assertNotIn("last_realization", status)
            [intent] = unresolved_create_intents(run_dir)
            realization = intent.with_name(intent.name.replace(".create-intent.json", ".json"))
            self.assertEqual(json.loads(realization.read_text(encoding="utf-8"))["pod"]["id"], "pod-1")
            self.assertIsNotNone(pod_may_still_bill(run_dir, status))
            with patch("kura.executors.runpod._runpod_pods_named") as listed, patch("kura.executors.runpod._runpod_request") as request:
                lines = resolve_runpod_create_intents(run_dir, CONFIG)
                listed.assert_not_called()
                self.assertIn("status now does", lines[0])
                self.assertEqual(unresolved_create_intents(run_dir), [])
                settled = _status(run_dir)
                self.assertEqual((settled["pod_id"], settled["last_realization"]), ("pod-1", f"realizations/{realization.name}"))
                self.assertNotIn("capacity_wait", settled)
                stop_runpod(run_dir, CONFIG)
            self.assertEqual([call.args[:2] for call in request.call_args_list], [("DELETE", "/pods/pod-1")])

    def test_stop_and_launch_refuse_until_reconcile_instead_of_creating_a_second_pod(self) -> None:
        from kura.run_commands.launch import launch_run
        from kura.run_commands.plan import stage_run

        with _crash_after_capacity_wait() as (_root, run_dir):
            stderr = io.StringIO()
            with patch("kura.executors.runpod._runpod_request") as request, patch("kura.run_commands.launch.launch_runpod") as launch, \
                    patch("kura.run_commands.plan.stage_runpod") as stage, patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                self.assertEqual(stop_run("example"), 1)
                # `kura run execute` outside the runner stages, then launches; neither goes ahead.
                self.assertEqual(stage_run("example"), 1)
                self.assertEqual(launch_run("example", executor="runpod", dry_run=False, image=None, yes=True), 1)
                with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                    stop_runpod(run_dir, CONFIG)
            request.assert_not_called()
            stage.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(stderr.getvalue().count("kura run reconcile"), 3)

    def test_the_runner_settles_from_the_records_and_follows_the_pod_instead_of_creating_another(self) -> None:
        from kura import runner

        with _crash_after_capacity_wait() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="runpod", extra={
                "options": {"max_lease": "3h"}, "runpod_config": {"gpu_type_ids": ["NVIDIA A40"]},
                "billing_confirmed_at": "2026-10-04T00:00:00+00:00",
            })
            runner.claim_request(request, 1)
            # The launch the request made is the one that crashed.
            [realization] = [path for path in (run_dir / "realizations").glob("*.json") if "." not in path.stem]
            record = json.loads(realization.read_text(encoding="utf-8"))
            record["controlled_by"] = {"request": request.name, "epoch": 1}
            realization.write_text(json.dumps(record), encoding="utf-8")
            with patch("kura.executors.runpod._runpod_pods_named") as listed, \
                    patch("kura.run_commands.launch._run_remote_locked") as remote, \
                    patch("kura.runner._continue_runpod", return_value=0) as follow:
                runner.work(root, "example", request.name)
            listed.assert_not_called()
            remote.assert_not_called()
            follow.assert_called_once()
            self.assertEqual(_status(run_dir)["pod_id"], "pod-1")


def _intent(run_dir: Path, rid: str, executor: str, *, recorded: dict | None = None) -> Path:
    """A create intent of `executor`, and its realization when `recorded` is given."""
    directory = run_dir / "realizations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{rid}.create-intent.json"
    draft = {"id": rid, "executor": executor, "container": {"id": None, "name": f"kura-example-{rid}", "labels": {}}}
    path.write_text(json.dumps({"realization_id": rid, "executor": executor, "pod_name": f"kura-example-{rid}", "realization": draft}), encoding="utf-8")
    if recorded is not None:
        (directory / f"{rid}.json").write_text(json.dumps({"id": rid, "executor": executor, **recorded}), encoding="utf-8")
    return path


class CreateIntentRuleTests(unittest.TestCase):
    """`unresolved_create_intents` decides from the records, for every executor."""

    def test_only_the_newest_launch_can_be_left_behind_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            _intent(run_dir, "20261004-101010-000001", "runpod", recorded={"state": "interrupted", "pod": {"id": "pod-old"}})
            _intent(run_dir, "20261004-111010-000001", "runpod", recorded={"state": "running", "pod": {"id": "pod-new"}})
            for state in ("running", "queued", "launching"):
                (run_dir / "status.json").write_text(json.dumps({"state": state, "last_realization": "realizations/20261004-111010-000001.json"}), encoding="utf-8")
                self.assertEqual(unresolved_create_intents(run_dir), [], state)

    def test_a_status_written_on_windows_names_its_realization_too(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            _intent(run_dir, RID, "runpod", recorded={"state": "completed", "pod": {"id": "pod-1"}})
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_realization": f"realizations\\{RID}.json"}), encoding="utf-8")
            self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_an_older_settled_launch_stays_settled_while_a_newer_one_is_unrecorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            _intent(run_dir, "20261004-101010-000001", "runpod", recorded={"state": "launch_failed", "pod": None})
            newer = _intent(run_dir, "20261004-111010-000001", "runpod")
            (run_dir / "status.json").write_text(json.dumps({"state": "launching", "last_realization": "realizations/20261004-101010-000001.json"}), encoding="utf-8")
            self.assertEqual(unresolved_create_intents(run_dir), [newer])

    def test_both_executors_settle_a_record_status_did_not_follow_through_the_one_owner(self) -> None:
        from kura.executors import common, docker, runpod
        from kura.executors.docker import resolve_docker_create_intents

        self.assertIs(docker.unresolved_create_intents, common.unresolved_create_intents)
        self.assertIs(runpod.unresolved_create_intents, common.unresolved_create_intents)
        outcomes = {}
        for executor, resolve in (("docker", resolve_docker_create_intents), ("runpod", lambda run_dir: resolve_runpod_create_intents(run_dir, CONFIG))):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory) / "example"
                _intent(run_dir, RID, executor, recorded={"state": "running", "launched_at": "2026-10-04T10:10:11+09:00"})
                (run_dir / "status.json").write_text(json.dumps({"state": "queued", "capacity_wait": {"attempts": 1}}), encoding="utf-8")
                calls = []
                with patch.object(docker, "unresolved_create_intents", side_effect=lambda *a, **k: calls.append(a) or common.unresolved_create_intents(*a, **k)), \
                        patch.object(runpod, "unresolved_create_intents", side_effect=lambda *a, **k: calls.append(a) or common.unresolved_create_intents(*a, **k)), \
                        patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                        patch("kura.executors.runpod._runpod_pods_named") as listed, patch("kura.executors.docker.subprocess.run") as ps:
                    resolve(run_dir)
                listed.assert_not_called()
                ps.assert_not_called()
                self.assertEqual(calls, [(run_dir, executor)])
                status = _status(run_dir)
                outcomes[executor] = (status["state"], status["last_realization"], "capacity_wait" in status, unresolved_create_intents(run_dir))
        self.assertEqual(outcomes["docker"], outcomes["runpod"])
        self.assertEqual(outcomes["runpod"], ("running", f"realizations/{RID}.json", False, []))


# A clock set back one hour (a DST fall-back): the later launch gets the smaller id.
EARLIER, LATER = "20261101-013000-000001", "20261101-011000-000001"
EARLIER_AT, LATER_AT = "2026-11-01T01:30:00-07:00", "2026-11-01T01:10:00-08:00"


def _timed_launch(run_dir: Path, rid: str, at: str, *, recorded: bool = True) -> Path:
    directory = run_dir / "realizations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{rid}.create-intent.json"
    path.write_text(json.dumps({"realization_id": rid, "executor": "runpod", "pod_name": f"kura-example-{rid}", "requested_at": at}), encoding="utf-8")
    if recorded:
        (directory / f"{rid}.json").write_text(json.dumps({"id": rid, "executor": "runpod", "state": "running", "launched_at": at, "pod": {"id": f"pod-{rid[-1]}{rid[9:13]}"}}), encoding="utf-8")
    return path


class LaunchOrderTests(unittest.TestCase):
    """The newest launch is the one its records say started last, not the one with the largest id."""

    def test_a_launch_after_the_clock_went_back_is_still_the_newest(self) -> None:
        from kura.executors import common
        from kura import status_projection

        self.assertIs(status_projection.latest_launch_id, common.latest_launch_id)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            _timed_launch(run_dir, EARLIER, EARLIER_AT)
            later = _timed_launch(run_dir, LATER, LATER_AT)
            self.assertEqual(common.latest_launch_id(run_dir / "realizations"), LATER)
            # Both settled: status follows the later launch, and the earlier one never pulls it back.
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": f"realizations/{LATER}.json"}), encoding="utf-8")
            self.assertEqual(unresolved_create_intents(run_dir), [])
            self.assertEqual(status_projection.project_status(run_dir)["last_realization"], f"realizations/{LATER}.json")
            # The later launch crashed before status followed it: it is the one left to settle.
            (run_dir / "status.json").write_text(json.dumps({"state": "launching", "last_realization": f"realizations/{EARLIER}.json"}), encoding="utf-8")
            self.assertEqual(unresolved_create_intents(run_dir), [later])

    def test_a_record_without_a_readable_time_is_ordered_by_its_id(self) -> None:
        from kura.executors import common

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "realizations" / "20261004-101010-000001.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
            (run_dir / "realizations" / "20261004-111010-000001.json").write_text("not json", encoding="utf-8")
            self.assertEqual(common.latest_launch_id(run_dir / "realizations"), "20261004-111010-000001")


class InterruptAfterCreateTests(unittest.TestCase):
    def test_an_interrupt_after_runpod_returned_the_pod_leaves_the_create_unconfirmed(self) -> None:
        """Never record "no Pod exists" once the create returned one."""
        import sys

        from kura.executors.runpod import launch_runpod

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "logs").mkdir(parents=True)
            _write_upload_stage(run_dir)

            def interrupt_once_the_pod_is_returned(frame, event, _arg):
                if frame.f_code is launch_runpod.__code__ and event == "line" and frame.f_locals.get("pod") is not None \
                        and frame.f_locals.get("controller_phase") == "probe":
                    sys.settrace(None)
                    frame.f_trace = None
                    raise KeyboardInterrupt
                return interrupt_once_the_pod_is_returned

            def tracer(frame, event, _arg):
                return interrupt_once_the_pod_is_returned if frame.f_code is launch_runpod.__code__ else None

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), patch("kura.executors.runpod.time.sleep"), \
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=AVAILABLE), \
                    patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}):
                sys.settrace(tracer)
                try:
                    with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                        launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                                      image="registry/image:tag", config={"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"], "cloud_types": ["COMMUNITY"]},
                                      wait_for_capacity_sec=60, capacity_poll_interval_sec=5, yes=True)
                finally:
                    sys.settrace(None)
            self.assertEqual([path for path in (run_dir / "realizations").glob("*.json") if "." not in path.stem], [])
            self.assertEqual(len(list((run_dir / "realizations").glob("*.create-unconfirmed.json"))), 1)
            self.assertEqual(len(unresolved_create_intents(run_dir)), 1)


class SettlingNeedsNoSecretTests(unittest.TestCase):
    def test_a_recorded_pod_is_settled_without_the_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            _intent(run_dir, RID, "runpod", recorded={"state": "running", "launched_at": "2026-10-04T10:10:11+09:00", "pod": {"id": "pod-1"}})
            (run_dir / "status.json").write_text(json.dumps({"state": "launching"}), encoding="utf-8")
            with patch("kura.executors.runpod.runpod_api_key", side_effect=ValueError("no RUNPOD_API_KEY")) as key, \
                    patch("kura.executors.runpod._runpod_pods_named") as listed:
                lines = resolve_runpod_create_intents(run_dir, CONFIG)
            key.assert_not_called()
            listed.assert_not_called()
            self.assertIn("status now does", lines[0])
            self.assertEqual(_status(run_dir)["pod_id"], "pod-1")


class RunnerSettlesBeforeDeletingTests(unittest.TestCase):
    def test_a_render_session_that_crashed_before_status_followed_is_deleted_by_the_runner(self) -> None:
        from kura import runner
        from kura.executors.runpod import launch_runpod_session

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-9", "desiredStatus": "RUNNING"}), \
                    patch("kura.executors.runpod.point_status_at_launch", side_effect=KeyboardInterrupt), patch("kura.executors.runpod.time.sleep"):
                with self.assertRaises(KeyboardInterrupt):
                    launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render", dry_run=False, yes=True, max_lease_sec=3600)
            self.assertNotIn("pod_id", _status(run_dir))
            self.assertEqual(len(unresolved_create_intents(run_dir)), 1)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_pods_named") as listed, patch("kura.executors.runpod._runpod_request") as request:
                self.assertTrue(runner._delete_pod(root, run_dir, {"runpod_config": CONFIG}, why="the render ended"))
            listed.assert_not_called()
            self.assertEqual([call.args[:2] for call in request.call_args_list], [("DELETE", "/pods/pod-9")])
            self.assertTrue(_status(run_dir).get("pod_stopped_at"))

    def test_a_stop_through_the_runner_settles_the_crashed_capacity_wait_and_deletes_its_pod(self) -> None:
        from kura import runner

        with _crash_after_capacity_wait() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="runpod", extra={
                "options": {"max_lease": "3h"}, "runpod_config": CONFIG, "billing_confirmed_at": "2026-10-04T00:00:00+00:00",
            })
            runner.claim_request(request, 1)
            with patch("kura.executors.runpod._runpod_pods_named") as listed, patch("kura.executors.runpod._runpod_request") as request_api:
                self.assertTrue(runner._stop_runpod_on_request(root, run_dir, request, runner._read_json(request)))
            listed.assert_not_called()
            self.assertEqual([call.args[:2] for call in request_api.call_args_list], [("DELETE", "/pods/pod-1")])
            self.assertTrue(runner._sibling(request, ".stop-done.json").exists())


def _no_price_lookup():
    # The launch measures stock and price before it creates; these tests are about the create.
    return patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "not measured in tests", "candidates": []})


class UnconfirmedCreateTests(unittest.TestCase):
    def test_a_failed_lookup_still_hands_the_run_over(self) -> None:
        from kura.executors.runpod import RunPodAPIError, launch_runpod_session

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "render"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod GraphQL failed (503): down", status_code=503)), \
                    patch("kura.executors.runpod._runpod_pods_named", side_effect=ValueError("RunPod API is unreachable: down")), patch("kura.executors.runpod.time.sleep"), \
                    _no_price_lookup():
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
                    patch("kura.executors.runpod._runpod_pods_named", return_value=[]), patch("kura.executors.runpod.time.sleep"), _no_price_lookup():
                with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                    launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render", dry_run=False, yes=True, max_lease_sec=3600)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(len(unresolved_create_intents(run_dir)), 1)


if __name__ == "__main__":
    unittest.main()
