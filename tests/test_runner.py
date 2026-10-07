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
                 patch("kura.executors.docker.reconcile_docker", side_effect=lambda *_a, **_k: next(states)) as reconcile:
                self.assertEqual(runner.work(root, "example", request.name, sleep=lambda _: None), 0)
            stop.assert_called_once()
            # Polling never records an observation unless the container's state changed.
            self.assertEqual({call.kwargs.get("source") for call in reconcile.call_args_list}, {"automatic"})
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


def _runpod_request(run_dir: Path, **extra) -> Path:
    return runner.write_launch_request(run_dir, executor="runpod", extra={
        "options": {"max_lease": "3h", "hold_for": "0"}, "runpod_config": {"gpu_type_ids": ["NVIDIA A40"]},
        "billing_confirmed_at": "2026-10-07T00:00:00+00:00", **extra,
    })


def _runpod_launched(run_dir: Path, request: Path, **status) -> None:
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"},
                                                                  "controlled_by": {"request": request.name}}), encoding="utf-8")
    _status(run_dir, last_realization="realizations/r1.json", pod_id="pod-1", **status)


class RunPodFollowerTests(unittest.TestCase):
    def test_a_request_without_billing_confirmation_creates_nothing(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="runpod")
            with patch("kura.run_commands.launch._run_remote_locked") as remote:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            remote.assert_not_called()
            self.assertIn("billing confirmation", runner.request_outcome(request)["error"])

    def test_the_first_attempt_launches_with_the_confirmed_settings(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)

            def launched(*_args, **kwargs):
                _runpod_launched(run_dir, request, state="completed", publication_state="completed", pod_stopped_at="t")
                return 0

            with patch("kura.run_commands.launch._run_remote_locked", side_effect=launched) as remote:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            kwargs = remote.call_args.kwargs
            self.assertEqual((kwargs["yes"], kwargs["reattach"], kwargs["max_lease"]), (True, False, "3h"))
            self.assertEqual(kwargs["runpod_config_override"], {"gpu_type_ids": ["NVIDIA A40"]})
            self.assertEqual(kwargs["controlled_by"]["request"], request.name)

    def test_a_follower_without_a_create_intent_continues_the_confirmed_launch(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            runner._first_attempt(request)  # an earlier follower died during the capacity wait
            with patch("kura.run_commands.launch._run_remote_locked", return_value=0) as remote:
                runner.work(root, "example", request.name)
            remote.assert_called_once()
            self.assertIsNone(runner.request_outcome(request))

    def test_a_pod_whose_job_never_started_is_deleted(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")

            def stopped(run_dir_arg, _config):
                _status(run_dir_arg, **{**json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8")), "pod_stopped_at": "t"})
                return {}

            with patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop, \
                 patch("kura.run_commands.launch._run_remote_locked") as remote:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            stop.assert_called_once()
            remote.assert_not_called()
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")

    def test_a_finished_run_that_left_its_pod_running_gets_the_pod_deleted(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            # Recovered from a create intent, or collected before a crash: finished, Pod still up.
            _runpod_launched(run_dir, request, state="interrupted")
            self.assertTrue(runner.run_unfinished(run_dir))

            def stopped(run_dir_arg, _config):
                _status(run_dir_arg, **{**json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8")), "pod_stopped_at": "t"})
                return {}

            with patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            stop.assert_called_once()
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_job_whose_pid_file_exists_is_followed_not_deleted(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job-intent.json").write_text(json.dumps({"pid_path": "/tmp/kura-jobs/x.pid"}), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh.remote_job_pid", return_value="42"), \
                 patch("kura.executors.runpod.stop_runpod") as stop, \
                 patch("kura.run_commands.launch._run_remote_locked", return_value=0) as remote:
                runner.work(root, "example", request.name)
            stop.assert_not_called()
            self.assertTrue(remote.call_args.kwargs["reattach"])
            self.assertEqual(json.loads((run_dir / "realizations" / "r1.remote-job.json").read_text(encoding="utf-8"))["pid"], "42")

    def test_an_unreachable_pod_is_never_taken_for_a_job_that_did_not_start(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job-intent.json").write_text(json.dumps({"pid_path": "/tmp/kura-jobs/x.pid"}), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh.remote_job_pid", side_effect=ValueError("ssh timed out")), \
                 patch("kura.executors.runpod.stop_runpod") as stop:
                # Counted as a failed attempt and retried later, never taken as "the job did not start".
                self.assertEqual(runner.work(root, "example", request.name), 1)
            stop.assert_not_called()

    def test_repeated_collection_failures_hand_the_run_to_a_person(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
            codes = []
            with patch("kura.run_commands.launch._run_remote_locked", return_value=1):
                for _ in range(runner.COLLECTION_ATTEMPTS):
                    codes.append(runner.work(root, "example", request.name))
            self.assertEqual(codes, [1] * (runner.COLLECTION_ATTEMPTS - 1) + [0])
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((status["state"], status["recovery_required"]), ("recovery_required", True))
            self.assertTrue(run_finished(status))

    def test_a_stop_request_stops_the_pod(self) -> None:
        from kura.executors.common import StopRequested

        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
            with patch("kura.run_commands.launch._run_remote_locked", side_effect=StopRequested()), \
                 patch("kura.executors.runpod.reconcile_runpod", side_effect=lambda run_dir_arg, *_a, **_k: json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8"))), \
                 patch("kura.executors.runpod.stop_runpod", return_value={}) as stop:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            stop.assert_called_once()
            self.assertTrue(runner.stop_done(run_dir))

    def test_a_stop_that_ended_the_capacity_wait_is_acknowledged_not_counted_as_a_failure(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)

            def stop_arrives_during_the_wait(*_args, **_kwargs):
                # launch_runpod turns the stop into "capacity wait cancelled" and the launch returns 1.
                runner.write_stop_request(run_dir)
                return 1

            with patch("kura.run_commands.launch._run_remote_locked", side_effect=stop_arrives_during_the_wait):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertTrue(runner.stop_done(run_dir))
            self.assertFalse(request.with_name(request.name.replace(".launch.json", ".failures.json")).exists())

    def test_runpod_requests_do_not_wait_for_a_local_slot(self) -> None:
        with _workspace() as (root, first):
            busy = runner.write_launch_request(first, executor="docker")
            runner.claim_request(busy, 1)
            _launched(first, busy, state="running")
            second = root / "runs" / "second"
            (second / "realizations").mkdir(parents=True)
            _status(second, state="compiled")
            remote = _runpod_request(second)
            polls = [0]

            def sleep(_):
                polls[0] += 1
                if polls[0] > 2:
                    _launched(first, busy, state="completed", publication_state="completed")

            spawned = []

            def spawn(workspace, run_id, request):
                spawned.append(run_id)
                _runpod_launched(second, request, state="completed", publication_state="completed", pod_stopped_at="t")
                return Child(0)

            with _run_operation_lock(first, "controller"):
                runner.serve(root, spawn_child=spawn, sleep=sleep)
            self.assertEqual(spawned, ["second"])
            self.assertTrue(remote.with_name(remote.name.replace(".launch.json", ".claim.json")).exists())


class RunPodWriterTests(unittest.TestCase):
    def test_billing_is_confirmed_before_the_request_and_recorded_in_it(self) -> None:
        from kura.run_commands import launch

        with _workspace() as (root, run_dir):
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(launch, "launch_run", return_value=1) as check:
                    self.assertEqual(launch._launch_runpod_through_runner("example", follow=False, yes=False, options={"max_lease": "3h"}), 1)
                self.assertTrue(check.call_args.kwargs["check_only"])
                self.assertEqual(runner.launch_requests(run_dir), [])

                def confirmed(*_args, **kwargs):
                    kwargs["prepared"].update({"runpod_config": {"gpu_type_ids": ["NVIDIA A40"]}, "remote_image": "img@sha256:" + "0" * 64})
                    return 0

                with patch.object(launch, "launch_run", side_effect=confirmed), \
                     patch("kura.runner.ensure_runner", return_value=False), patch("kura.runner.await_claim", return_value=True):
                    self.assertEqual(launch._launch_runpod_through_runner("example", follow=False, yes=True, options={"max_lease": "3h", "image": None}), 0)
            finally:
                os.chdir(previous)
            [request] = runner.launch_requests(run_dir)
            payload = json.loads(request.read_text(encoding="utf-8"))
            self.assertEqual(payload["executor"], "runpod")
            self.assertTrue(payload["billing_confirmed_at"])
            self.assertEqual(payload["runpod_config"], {"gpu_type_ids": ["NVIDIA A40"]})
            self.assertEqual(payload["options"], {"max_lease": "3h"})


def _render_run(run_dir: Path, *, lora_dir: Path | None = None) -> None:
    import yaml

    from kura.comfyui_models import endpoint_fingerprint

    resolved = run_dir / "resolved"
    resolved.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "render", "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"}, "executor": {"name": "local"},
        "inputs": {"workflow": {"path": "workflows/test.json"}, "promptset": {"path": "promptsets/test.jsonl"}, "checkpoint": {}},
        "workflow_patches": {}, "render": {"output_dir": "samples/images", "timeout_sec": 5},
        "comfyui_endpoint_identity": endpoint_fingerprint({"KSampler": {}}),
        **({"comfyui": {"lora_dir": str(lora_dir)}} if lora_dir else {}),
    }
    (resolved / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (resolved / "workflow_used.json").write_text("{}", encoding="utf-8")
    (resolved / "promptset_used.jsonl").write_text('{"id":"p1","prompt":"hello","seeds":[1]}\n', encoding="utf-8")
    _status(run_dir, state="compiled")


class LocalRenderTests(unittest.TestCase):
    def test_an_interrupted_render_withdraws_only_its_own_prompt(self) -> None:
        from kura.render import launch_render

        with _workspace() as (root, run_dir):
            _render_run(run_dir)
            calls = []

            class Client:
                def __init__(self, endpoint, timeout):
                    pass

                def object_info(self):
                    return {"KSampler": {}}

                def queue(self, workflow):
                    return "prompt-1"

                def wait(self, prompt_id):
                    raise KeyboardInterrupt

                def cancel(self, prompt_id):
                    calls.append(prompt_id)

            with patch("kura.render.ComfyUIClient", Client), self.assertRaises(KeyboardInterrupt):
                launch_render(root, run_dir, controlled_by={"request": "r.launch.json"})
            self.assertEqual(calls, ["prompt-1"])
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "interrupted")
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual((realization["state"], realization["controlled_by"]), ("interrupted", {"request": "r.launch.json"}))

    def test_cancel_deletes_the_prompt_and_interrupts_only_that_prompt(self) -> None:
        from kura.render import ComfyUIClient

        for running, interrupts in (([[0, "prompt-1"]], True), ([[0, "someone-else"]], False)):
            sent = []

            def fake_json(path, payload=None, running=running):
                sent.append((path, payload))
                return {"queue_running": running, "queue_pending": []} if payload is None else {}

            client = ComfyUIClient("http://127.0.0.1:8188", 5)
            with patch.object(client, "_json", side_effect=fake_json):
                client.cancel("prompt-1")
            self.assertEqual(sent[0], ("/queue", {"delete": ["prompt-1"]}))
            self.assertEqual(("/interrupt", {"prompt_id": "prompt-1"}) in sent, interrupts)

    def test_a_manifest_that_is_not_a_mapping_removes_nothing(self) -> None:
        from kura.render import remove_leftover_stages

        with _workspace() as (root, run_dir):
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("- a\n- b\n", encoding="utf-8")
            self.assertEqual(remove_leftover_stages(root, run_dir), [])

    def test_an_unreachable_comfyui_is_refused_before_the_request(self) -> None:
        from kura.run_commands import launch

        with _workspace() as (root, run_dir):
            _render_run(run_dir)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(launch, "launch_render", side_effect=OSError("connection refused")), patch("sys.stderr", io.StringIO()):
                    self.assertEqual(launch._launch_render_through_runner("example", follow=False), 1)
            finally:
                os.chdir(previous)
            self.assertEqual(runner.launch_requests(run_dir), [])

    def test_a_follower_renders_once_with_its_request(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="render-local")
            runner.claim_request(request, 1)
            with patch("kura.render.launch_render", return_value=0) as render:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(render.call_args.kwargs["controlled_by"]["request"], request.name)
            # A second follower never renders again.
            with patch("kura.render.launch_render") as again:
                runner.work(root, "example", request.name)
            again.assert_not_called()

    def test_a_render_cut_short_is_recorded_and_its_staged_files_removed(self) -> None:
        with _workspace() as (root, run_dir):
            lora_dir = root / "comfy" / "loras"
            (lora_dir / "Kura_tmp").mkdir(parents=True)
            mine = lora_dir / "Kura_tmp" / "example-lora-1234abcd.safetensors"
            others = lora_dir / "Kura_tmp" / "other-run-lora-1234abcd.safetensors"
            mine.write_bytes(b"x")
            others.write_bytes(b"y")
            _render_run(run_dir, lora_dir=lora_dir)
            request = runner.write_launch_request(run_dir, executor="render-local")
            runner.claim_request(request, 1)
            runner._first_attempt(request)  # the follower that was rendering died
            _status(run_dir, state="running", last_step=3, total_steps=10)
            with patch("kura.render.launch_render") as render:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            render.assert_not_called()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "interrupted")
            self.assertFalse(mine.exists())
            self.assertTrue(others.exists())
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_stop_request_ends_the_render(self) -> None:
        from kura.executors.common import StopRequested

        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="render-local")
            runner.claim_request(request, 1)
            with patch("kura.render.launch_render", side_effect=StopRequested()):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertTrue(runner.stop_done(run_dir))

    def test_execute_sends_every_render_to_the_runner_with_runpod_settings_only_for_runpod(self) -> None:
        import yaml

        from kura.run_commands import launch

        with _workspace() as (root, run_dir):
            (run_dir / "resolved").mkdir()
            for name in ("runpod", "local"):
                (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({"type": "render", "executor": {"name": name}}), encoding="utf-8")
                with patch.object(launch, "_run_path", return_value=run_dir), \
                     patch.object(launch, "_launch_render_through_runner", return_value=0) as through_runner:
                    launch.execute_run("example", yes=True)
                through_runner.assert_called_once()
                runpod = through_runner.call_args.kwargs.get("runpod")
                if name == "runpod":
                    self.assertEqual((runpod["yes"], runpod["max_lease"]), (True, "12h"))
                else:
                    self.assertIsNone(runpod)


def _render_runpod_request(run_dir: Path, **extra) -> Path:
    return runner.write_launch_request(run_dir, executor="render-runpod", extra={
        "options": {"max_lease_sec": 3600}, "runpod_config": {"gpu_type_ids": ["A"]}, "remote_image": "img",
        "billing_confirmed_at": "2026-10-08T00:00:00+00:00", **extra,
    })


class RunPodRenderTests(unittest.TestCase):
    def test_a_request_without_billing_confirmation_creates_nothing(self) -> None:
        with _workspace() as (root, run_dir):
            request = runner.write_launch_request(run_dir, executor="render-runpod")
            with patch("kura.run_commands.render_runpod.launch_render_runpod") as render:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            render.assert_not_called()
            self.assertIn("billing confirmation", runner.request_outcome(request)["error"])

    def test_the_first_attempt_renders_once_with_the_confirmed_settings(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)

            def rendered(*_args, **kwargs):
                _runpod_launched(run_dir, request, state="completed", pod_stopped_at="t")
                return 0

            with patch("kura.run_commands.render_runpod.launch_render_runpod", side_effect=rendered) as render, \
                    patch("kura.executors.runpod.stop_runpod") as stop:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            kwargs = render.call_args.kwargs
            self.assertTrue(kwargs["yes"])
            self.assertEqual((kwargs["max_lease_sec"], kwargs["image"], kwargs["runpod_config_override"]), (3600, "img", {"gpu_type_ids": ["A"]}))
            self.assertEqual(kwargs["controlled_by"]["request"], request.name)
            stop.assert_not_called()  # the render deleted its own Pod
            self.assertIsNone(runner.request_outcome(request))
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_pod_the_render_left_running_is_deleted(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)

            def rendered(*_args, **kwargs):
                _runpod_launched(run_dir, request, state="failed")
                return 1

            with patch("kura.run_commands.render_runpod.launch_render_runpod", side_effect=rendered), \
                    patch("kura.executors.runpod.stop_runpod") as stop:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(stop.call_args.args[1], {"gpu_type_ids": ["A"]})

    def test_a_render_whose_follower_died_is_interrupted_and_its_pod_deleted_not_continued(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)
            runner._first_attempt(request)  # the follower that was rendering died
            _runpod_launched(run_dir, request, state="running", last_step=2, total_steps=5)

            def stopped(run_dir, config):
                _mutate = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                _status(run_dir, **{**_mutate, "pod_stopped_at": "t"})
                return {}

            with patch("kura.run_commands.render_runpod.launch_render_runpod") as render, \
                    patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop, patch("kura.render.write_realization") as realization:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            render.assert_not_called()
            stop.assert_called_once()
            self.assertEqual(realization.call_args.kwargs["executor"], "runpod")
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_stop_asked_of_a_dead_follower_is_acknowledged_once_the_pod_is_gone(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)
            runner._first_attempt(request)
            _runpod_launched(run_dir, request, state="running", pod_stopped_at="t")
            runner.write_stop_request(run_dir)
            with patch("kura.render.write_realization"):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertTrue(runner.stop_done(run_dir))

    def test_a_pod_that_could_not_be_deleted_is_tried_again(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)
            runner._first_attempt(request)
            _runpod_launched(run_dir, request, state="running")
            with patch("kura.executors.runpod.stop_runpod", side_effect=ValueError("api down")):
                self.assertEqual(runner.work(root, "example", request.name), 1)
            self.assertTrue(runner.run_unfinished(run_dir))

    def test_a_stop_request_is_acknowledged_after_the_render_ends(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)

            def rendered(*_args, **kwargs):
                _runpod_launched(run_dir, request, state="interrupted", pod_stopped_at="t")
                runner.write_stop_request(run_dir)
                return 130

            with patch("kura.run_commands.render_runpod.launch_render_runpod", side_effect=rendered):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertTrue(runner.stop_done(run_dir))

    def test_a_render_that_never_started_is_recorded_as_a_failed_launch(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)
            with patch("kura.run_commands.render_runpod.launch_render_runpod", return_value=1):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(runner.request_outcome(request)["kind"], "launch_failed")


class RunPodSessionLeaseTests(unittest.TestCase):
    def test_the_session_pod_keeps_its_lease_in_the_deadline_file(self) -> None:
        from kura.executors import runpod

        captured = {}

        def create(method, path, api_key, body=None, **_):
            captured["body"] = body
            return {"id": "pod-1", "desiredStatus": "RUNNING"}

        with _workspace() as (root, run_dir), patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), \
                patch.object(runpod, "_runpod_request", side_effect=create), \
                patch.object(runpod, "_confirm_runpod_launch", return_value={}):
            runpod.launch_runpod_session(run_dir=run_dir, image="img", config={"gpu_type_ids": ["A"]}, purpose="comfyui-render",
                                         yes=True, max_lease_sec=3600, controlled_by={"request": "r.launch.json"})
            script = captured["body"]["dockerStartCmd"][-1]
            self.assertIn(runpod.LEASE_DEADLINE_PATH, script)
            # Armed before SSH is installed, so a stalled setup is still bounded; and it parses under sh.
            self.assertLess(script.index(runpod.LEASE_DEADLINE_PATH), script.index("openssh-server"))
            import shutil
            import subprocess

            if shutil.which("sh"):
                self.assertEqual(subprocess.run(["sh", "-n", "-c", script], capture_output=True).returncode, 0)
            self.assertNotIn('sleep "$KURA_MAX_LEASE_SEC"', script)
            realization = json.loads(next(p for p in (run_dir / "realizations").glob("*.json") if "." not in p.stem).read_text(encoding="utf-8"))
            self.assertEqual((realization["lease_guard"], realization["controlled_by"]), ("deadline_file", {"request": "r.launch.json"}))

    def test_only_a_render_pod_with_a_deadline_file_takes_a_lease_change(self) -> None:
        from kura.run_commands.runpod_ssh import change_runpod_lease

        with _workspace() as (root, run_dir):
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"executor": "runpod", "purpose": "comfyui-render", "pod": {"id": "pod-1"}}), encoding="utf-8")
            _status(run_dir, last_realization="realizations/r1.json", pod_id="pod-1", state="running")
            with self.assertRaisesRegex(ValueError, "before Kura kept the deadline"):
                change_runpod_lease(run_dir, 3600, yes=True)
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"executor": "runpod", "purpose": "comfyui-render", "lease_guard": "deadline_file", "pod": {"id": "pod-1"}}), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", side_effect=ValueError("reached the Pod")):
                with self.assertRaisesRegex(ValueError, "reached the Pod"):
                    change_runpod_lease(run_dir, 3600, yes=True)


class RunPodRenderLaunchTests(unittest.TestCase):
    def _render(self, root: Path) -> Path:
        import yaml

        (root / "workspace.yaml").write_text("images:\n  comfyui: remote/comfy\nrunpod:\n  gpu_type_ids: [A]\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        resolved = run_dir / "resolved"
        resolved.mkdir(parents=True, exist_ok=True)
        (resolved / "workflow_used.json").write_text("{}", encoding="utf-8")
        (resolved / "comfyui_model_registry.json").write_text("{}", encoding="utf-8")
        (resolved / "manifest.lock.yaml").write_text(yaml.safe_dump({
            "type": "render", "generator": {"name": "comfyui"}, "executor": {"name": "runpod"},
            "comfyui_models": [], "comfyui_model_registry": {},
        }), encoding="utf-8")
        return run_dir

    def test_a_render_interrupted_before_its_cases_is_recorded_and_its_pod_deleted(self) -> None:
        from kura.run_commands import render_runpod

        with _workspace() as (root, _):
            run_dir = self._render(root)

            def session(**kwargs):
                _status(run_dir, state="running", pod_id="pod-1", last_realization="realizations/r1.json")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(render_runpod, "launch_runpod_session", side_effect=session), \
                        patch.object(render_runpod, "_runpod_ssh_details", side_effect=KeyboardInterrupt), \
                        patch.object(render_runpod, "stop_runpod") as stop, patch("sys.stderr", io.StringIO()):
                    code = render_runpod.launch_render_runpod("example", dry_run=False, yes=True, runpod_config_override={"gpu_type_ids": ["A"]})
            finally:
                os.chdir(previous)
            self.assertEqual(code, 130)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")
            stop.assert_called_once_with(run_dir, {"gpu_type_ids": ["A"]})

    def test_check_only_confirms_billing_and_creates_nothing(self) -> None:
        from kura.run_commands import render_runpod

        with _workspace() as (root, _):
            self._render(root)
            prepared: dict = {}
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(render_runpod, "confirm_runpod_billing") as confirm, \
                        patch.object(render_runpod, "launch_runpod_session") as session:
                    code = render_runpod.launch_render_runpod("example", dry_run=False, yes=True, max_lease_sec=7200, check_only=True, prepared=prepared)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            session.assert_not_called()
            self.assertEqual(confirm.call_args.kwargs["max_lease_sec"], 7200)
            self.assertEqual((prepared["remote_image"], prepared["runpod_config"]["gpu_type_ids"]), ("remote/comfy", ["A"]))

    def test_a_stop_during_pod_setup_deletes_the_pod(self) -> None:
        from kura.executors.common import StopRequested
        from kura.run_commands import render_runpod

        with _workspace() as (root, _):
            run_dir = self._render(root)

            def session(**kwargs):
                _status(run_dir, state="running", pod_id="pod-1", last_realization="realizations/r1.json")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(render_runpod, "launch_runpod_session", side_effect=session), \
                        patch.object(render_runpod, "_runpod_ssh_details", return_value={"ip": "h", "port": 22, "key": "k"}), \
                        patch.object(render_runpod, "_start_runpod_session_lease_guard"), \
                        patch.object(render_runpod, "_record_session_lease"), \
                        patch.object(render_runpod, "_sync_runpod_remote_stdout"), \
                        patch.object(render_runpod, "check_stop", side_effect=StopRequested()), \
                        patch.object(render_runpod.subprocess, "run") as remote, \
                        patch.object(render_runpod, "stop_runpod") as stop, patch("sys.stderr", io.StringIO()):
                    code = render_runpod.launch_render_runpod("example", dry_run=False, yes=True, runpod_config_override={"gpu_type_ids": ["A"]})
            finally:
                os.chdir(previous)
            self.assertEqual(code, 130)
            remote.assert_not_called()  # nothing was uploaded after the stop
            stop.assert_called_once()
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")

    def test_a_declined_confirmation_sends_no_notification(self) -> None:
        from kura.run_commands import render_runpod

        with _workspace() as (root, _):
            self._render(root)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(render_runpod, "confirm_runpod_billing", side_effect=ValueError("RunPod launch cancelled")), \
                        patch.object(render_runpod, "_notify") as notify, patch("sys.stderr", io.StringIO()):
                    code = render_runpod.launch_render_runpod("example", dry_run=False, check_only=True)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            notify.assert_not_called()
