"""The job runner: requests, claims, waking, serving, following, and epoch fencing."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura import __version__, runner
from kura.executors.common import StaleRunnerEpoch, _mutate_run_status, _run_operation_lock, run_finished
from kura.fsio import file_lock


@contextmanager
def _workspace():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
        yield root, run_dir


def _status(run_dir: Path, **fields) -> None:
    (run_dir / "status.json").write_text(json.dumps(fields), encoding="utf-8")


def _launched(run_dir: Path, request: Path, **status) -> None:
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "controlled_by": {"request": request.name, "epoch": 1}}), encoding="utf-8")
    _status(run_dir, last_realization="realizations/r1.json", **status)


class Child:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


class RequestTests(unittest.TestCase):
    def test_one_pending_request_per_run_and_a_claim_is_taken_once(self) -> None:
        with _workspace() as (_, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            payload = json.loads(request.read_text(encoding="utf-8"))
            self.assertEqual((payload["kind"], payload["kura_version"], payload["confirmed"]), ("launch_request", __version__, True))
            with self.assertRaisesRegex(ValueError, "already has a launch request"):
                runner.write_launch_request(run_dir, executor="docker")
            self.assertTrue(runner.claim_request(request, 3))
            self.assertFalse(runner.claim_request(request, 4))
            self.assertEqual(runner.pending_requests(run_dir), [])


class WakingTests(unittest.TestCase):
    def test_a_runner_is_started_only_when_none_holds_the_lock_and_none_was_stopped(self) -> None:
        with _workspace() as (root, _):
            spawned = []
            self.assertTrue(runner.ensure_runner(root, spawn=spawned.append))
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                self.assertTrue(runner.runner_alive(root))
                self.assertFalse(runner.ensure_runner(root, spawn=spawned.append))
            runner.stop_runner(root)
            self.assertFalse(runner.ensure_runner(root, spawn=spawned.append))
            # A launching command clears a deliberate stop.
            self.assertTrue(runner.ensure_runner(root, launching=True, spawn=spawned.append))
            self.assertEqual(len(spawned), 2)
            self.assertFalse(runner.stopped_on_purpose(root))

    def test_the_runner_environment_carries_no_secret(self) -> None:
        environment = runner.runner_environment({"PATH": "/bin", "HF_TOKEN": "x", "RUNPOD_API_KEY": "y", "KURA_NTFY_TOKEN": "z", "HOME": "/h"})
        self.assertEqual(environment, {"PATH": "/bin", "HOME": "/h"})


class ServeTests(unittest.TestCase):
    def test_claims_a_request_starts_one_follower_and_exits_when_done(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            children = []

            def spawn(workspace, run_id, claimed):
                children.append((run_id, claimed.name))
                _launched(run_dir, claimed, state="completed", publication_state="completed")
                return Child(0)

            code = runner.serve(root, spawn_child=spawn, sleep=lambda _: None)
            self.assertEqual(code, 0)
            self.assertEqual(children, [("example", request.name)])
            claim = json.loads(request.with_name(request.name.replace(".launch.json", ".claim.json")).read_text(encoding="utf-8"))
            info = runner.runner_info(root)
            self.assertEqual((claim["epoch"], info["epoch"], info["kura_version"]), (1, 1, __version__))

    def test_a_request_from_another_version_is_left_pending_and_does_not_keep_the_runner(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            payload = json.loads(request.read_text(encoding="utf-8"))
            request.write_text(json.dumps({**payload, "kura_version": "0.0.1"}), encoding="utf-8")
            self.assertEqual(runner.serve(root, spawn_child=lambda *_: self.fail("spawned"), sleep=lambda _: None), 0)
            self.assertEqual(runner.pending_requests(run_dir), [request])
            self.assertEqual(runner.foreign_requests(root), [f"example/{request.name} (Kura 0.0.1)"])

    def test_a_failing_follower_is_restarted_with_backoff_not_every_poll(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="running")
            starts, now = [], [0.0]

            def spawn(*_args):
                starts.append(now[0])
                return Child(1)

            def sleep(seconds):
                now[0] += seconds
                if now[0] > 60:
                    _status(run_dir, last_realization="realizations/r1.json", state="failed")

            runner.serve(root, spawn_child=spawn, sleep=sleep, clock=lambda: now[0])
            gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
            self.assertTrue(gaps and all(gap >= 10 for gap in gaps), gaps)


class HeldRunTests(unittest.TestCase):
    def test_a_run_held_by_a_surviving_follower_gets_no_second_follower(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="running")
            polls = [0]

            def sleep(_):
                polls[0] += 1
                if polls[0] > 3:
                    _launched(run_dir, request, state="completed", publication_state="completed")

            with _run_operation_lock(run_dir, "controller"):
                runner.serve(root, spawn_child=lambda *_: self.fail("spawned beside a live follower"), sleep=sleep)

    def test_follow_does_not_restart_a_runner_stopped_on_purpose(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.stop_runner(root)
            out = io.StringIO()
            self.assertEqual(runner.follow(root, run_dir, request, sleep=lambda _: None, out=out, ensure=lambda *a, **k: self.fail("restarted")), 2)
            self.assertIn("stopped on purpose", out.getvalue())


class WorkTests(unittest.TestCase):
    def test_a_request_taken_before_and_never_launched_is_recorded_not_launched(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            self.assertTrue(runner._first_attempt(request))  # an earlier follower died right here
            with patch("kura.run_commands.launch.launch_run") as launch:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            launch.assert_not_called()
            self.assertEqual(runner.request_outcome(request)["kind"], "not_launched")

    def test_a_refused_launch_is_recorded_for_the_follower_to_report(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            with patch("kura.run_commands.launch.launch_run", return_value=1) as launch:
                runner.work(root, "example", request.name)
            self.assertEqual(launch.call_args.kwargs["controlled_by"], {"request": request.name, "epoch": 1})
            self.assertEqual(runner.request_outcome(request)["kind"], "launch_failed")
            self.assertEqual(runner.follow(root, run_dir, request, sleep=lambda _: None, out=io.StringIO()), 1)

    def test_a_second_follower_for_the_same_run_steps_aside(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            with _run_operation_lock(run_dir, "controller"), patch("kura.run_commands.launch.launch_run") as launch:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            launch.assert_not_called()


class FollowTests(unittest.TestCase):
    def test_follow_returns_the_run_outcome_once_no_follower_holds_the_run(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="failed", publication_state="not-required")
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                self.assertEqual(runner.follow(root, run_dir, request, sleep=lambda _: None), 1)
            _launched(run_dir, request, state="completed", publication_state="completed")
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                self.assertEqual(runner.follow(root, run_dir, request, sleep=lambda _: None), 0)

    def test_publishing_and_pending_publication_are_not_finished(self) -> None:
        self.assertFalse(run_finished({"state": "publishing"}))
        self.assertFalse(run_finished({"state": "failed", "publication_state": "pending"}))
        for state in ("completed", "failed", "interrupted", "launch_failed", "recovery_required", "unknown"):
            self.assertTrue(run_finished({"state": state}), state)


class EpochTests(unittest.TestCase):
    def test_a_replaced_runner_follower_stops_instead_of_overwriting(self) -> None:
        with _workspace() as (_, run_dir):
            _status(run_dir, state="running", epoch=5)
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "3"}), self.assertRaises(StaleRunnerEpoch):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "failed"}))
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "5"}):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "failed"}))
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "failed")

    def test_a_command_line_writer_is_never_fenced_and_keeps_the_epoch(self) -> None:
        with _workspace() as (_, run_dir):
            # runner.json is gone, yet the status remembers epoch 7.
            _status(run_dir, state="running", epoch=7)
            env = {key: value for key, value in os.environ.items() if key != "KURA_RUNNER_EPOCH"}
            with patch.dict(os.environ, env, clear=True):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "interrupted"}))
            written = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((written["state"], written["epoch"]), ("interrupted", 7))

    def test_a_new_runner_epoch_never_goes_back(self) -> None:
        with _workspace() as (root, run_dir):
            _status(run_dir, state="running", epoch=9)
            self.assertEqual(runner.highest_epoch(root), 9)

    def test_a_follower_of_a_replaced_runner_exits(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="running", epoch=4)
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "2"}), \
                 patch("kura.executors.docker.reconcile_docker", side_effect=StaleRunnerEpoch("newer")):
                self.assertEqual(runner.work(root, "example", request.name), 0)


class ViewerTests(unittest.TestCase):
    def test_a_viewer_never_reconciles_a_runner_controlled_run(self) -> None:
        from kura.executors import observe_run

        with _workspace() as (_, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="running")
            with patch("kura.executors.observe.reconcile_docker") as reconcile, patch("kura.runner.ensure_runner") as wake:
                status = observe_run(run_dir)
            reconcile.assert_not_called()
            wake.assert_called_once()
            self.assertEqual(status["state"], "running")


class StopAndQueueTests(unittest.TestCase):
    def test_a_follower_carries_out_a_stop_request_once(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="running")
            runner.write_stop_request(run_dir)
            states = iter([{"state": "running"}, {"state": "failed", "publication_state": "not-required"}])
            with patch("kura.executors.docker.stop_docker") as stop, \
                 patch("kura.executors.docker.reconcile_docker", side_effect=lambda *_a, **_k: next(states)):
                self.assertEqual(runner.work(root, "example", request.name, sleep=lambda _: None), 0)
            stop.assert_called_once()
            self.assertTrue(runner.stop_done(run_dir))

    def test_a_stop_before_launch_cancels_the_request(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.write_stop_request(run_dir)
            runner.serve(root, spawn_child=lambda *_: self.fail("launched a stopped request"), sleep=lambda _: None)
            self.assertEqual(runner.request_outcome(request)["kind"], "not_launched")

    def test_local_training_waits_for_a_free_slot(self) -> None:
        with _workspace() as (root, first):
            second = root / "runs" / "second"
            (second / "realizations").mkdir(parents=True)
            _status(second, state="compiled")
            busy = runner.write_launch_request(first, executor="docker")
            runner.claim_request(busy, 1)
            _launched(first, busy, state="running")
            waiting = runner.write_launch_request(second, executor="docker")
            polls = [0]

            def sleep(_):
                polls[0] += 1
                if polls[0] == 3:
                    self.assertEqual(runner.pending_requests(second), [waiting])
                    self.assertTrue(runner.local_slots_full(root))
                    _launched(first, busy, state="completed", publication_state="completed")

            def spawn(workspace, run_id, request):
                if run_id == "second":
                    (second / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "controlled_by": {"request": request.name}}), encoding="utf-8")
                    _status(second, last_realization="realizations/r1.json", state="completed", publication_state="completed")
                return Child(0)

            with _run_operation_lock(first, "controller"):
                runner.serve(root, spawn_child=spawn, sleep=sleep)
            self.assertEqual(runner.pending_requests(second), [])
            self.assertGreaterEqual(polls[0], 3)


class OrderAndCancelTests(unittest.TestCase):
    def test_requests_are_taken_in_the_order_written_not_by_run_name(self) -> None:
        with _workspace() as (root, _):
            older_dir, newer_dir = root / "runs" / "zzz", root / "runs" / "aaa"
            for run_dir in (older_dir, newer_dir):
                (run_dir / "realizations").mkdir(parents=True)
                _status(run_dir, state="compiled")
            older = runner.write_launch_request(older_dir, executor="docker")
            time.sleep(0.01)
            runner.write_launch_request(newer_dir, executor="docker")
            claimed = []

            def spawn(workspace, run_id, request):
                claimed.append(run_id)
                run_dir = workspace / "runs" / run_id
                (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "controlled_by": {"request": request.name}}), encoding="utf-8")
                _status(run_dir, last_realization="realizations/r1.json", state="completed", publication_state="completed")
                return Child(0)

            runner.serve(root, spawn_child=spawn, sleep=lambda _: None)
            self.assertEqual(claimed[0], "zzz")
            self.assertTrue(older.with_name(older.name.replace(".launch.json", ".claim.json")).exists())

    def test_a_pending_request_is_cancelled_without_a_runner(self) -> None:
        from kura.run_commands.plan import _stop_through_runner

        with _workspace() as (_, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            self.assertEqual(_stop_through_runner(run_dir), 0)
            self.assertEqual(runner.request_outcome(request)["kind"], "not_launched")

    def test_stopping_a_finished_run_does_not_wait_for_a_follower(self) -> None:
        from kura.run_commands.plan import _stop_through_runner

        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="completed", publication_state="completed")
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                self.assertIsNone(_stop_through_runner(run_dir, timeout_sec=0.1))


class ProcessTests(unittest.TestCase):
    def test_a_detached_runner_starts_records_its_epoch_and_exits_when_idle(self) -> None:
        with _workspace() as (root, _):
            source = Path(__file__).resolve().parents[1] / "src"
            env = {**os.environ, "PYTHONPATH": str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")}
            with patch.dict(os.environ, env):
                process = runner._spawn_runner(root)
            process.wait(timeout=60)
            self.assertEqual(process.returncode, 0, (root / ".kura" / "runner" / "runner.log").read_text(encoding="utf-8"))
            self.assertEqual(runner.runner_info(root)["epoch"], 1)
            self.assertFalse(runner.runner_alive(root))
            self.assertIn("nothing left to control", (root / ".kura" / "runner" / "runner.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
