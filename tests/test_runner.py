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


_DISK_WATCH = patch("kura.run_commands.plan.local_reserve_problem", return_value=(None, []))
_DOCKER_DATA = patch("kura.executors.docker.docker_data_location", return_value=(None, "not measured in tests"))


def setUpModule() -> None:
    # The disk watch reads this host's free space and asks Docker; its own tests are in test_disk_watch.
    _DISK_WATCH.start()
    _DOCKER_DATA.start()


def tearDownModule() -> None:
    _DISK_WATCH.stop()
    _DOCKER_DATA.stop()


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

    def test_a_follower_starts_without_the_runners_secrets_so_it_reads_the_files_again(self) -> None:
        # The runner loaded the secrets when it started; a token set or rotated later must reach
        # the next run, so each follower starts from the same secret-free environment as the runner.
        loaded = {"HF_TOKEN": "hf_token_loaded_at_runner_start", "KURA_RUNNER_EPOCH": "3"}
        with _workspace() as (root, run_dir), patch.dict(os.environ, loaded), \
                patch.object(runner.subprocess, "Popen") as popen:
            runner._spawn_child(root, run_dir.name, run_dir / "request.json")
        env = popen.call_args.kwargs.get("env")
        self.assertIsNotNone(env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["KURA_RUNNER_EPOCH"], "3")
        self.assertEqual(env, runner.runner_environment({**os.environ, **loaded}))


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
            self.assertEqual((written["state"], written["runner_epoch"]), ("interrupted", 7))
            self.assertNotIn("epoch", written)

    def test_status_names_the_runner_epoch_and_reads_the_old_key(self) -> None:
        with _workspace() as (_, run_dir):
            # A status written before the rename still fences a replaced follower.
            _status(run_dir, state="running", epoch=5)
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "3"}), self.assertRaises(StaleRunnerEpoch):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "failed"}))
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "5"}):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "failed"}))
            written = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((written["state"], written["runner_epoch"]), ("failed", 5))
            self.assertNotIn("epoch", written)
            # The new key fences in the same way.
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "4"}), self.assertRaises(StaleRunnerEpoch):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "interrupted"}))

    def test_a_render_of_a_replaced_follower_records_nothing(self) -> None:
        from kura import render

        with _workspace() as (_, run_dir):
            _status(run_dir, state="running", epoch=5, last_step=1)
            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "3"}), self.assertRaises(StaleRunnerEpoch):
                render.write_realization(run_dir, status_changes={"state": "failed"}, state="failed")
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")

    def test_a_render_whose_runner_was_replaced_mid_render_records_no_outcome(self) -> None:
        from kura.render import launch_render

        with _workspace() as (root, run_dir):
            _render_run(run_dir)

            class Client:
                def __init__(self, endpoint, timeout):
                    pass

                def object_info(self):
                    return {"KSampler": {}}

                def queue(self, workflow):
                    return "prompt-1"

                def wait(self, prompt_id):
                    # A newer runner takes the run over while this case renders.
                    current = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                    _status(run_dir, **{**current, "epoch": 9})
                    return [{"filename": "a.png"}]

                def download(self, image):
                    return b"png"

                def cancel(self, prompt_id):
                    pass

            with patch.dict(os.environ, {"KURA_RUNNER_EPOCH": "3"}), patch("kura.render.ComfyUIClient", Client), \
                    self.assertRaises(StaleRunnerEpoch):
                launch_render(root, run_dir, controlled_by={"request": "r.launch.json"})
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")
            self.assertEqual([path for path in (run_dir / "realizations").glob("*.json") if "." not in path.stem], [])

    def test_a_run_already_ended_by_someone_else_is_left_as_it_is(self) -> None:
        from kura.executors.common import end_run

        with _workspace() as (_, run_dir):
            _status(run_dir, state="completed", last_realization="realizations/r1.json")
            end_run(run_dir, "failed", reason="late", unless_finished=True)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "completed")
            self.assertEqual(list((run_dir / "realizations").glob("*.ended-*.json")), [])
            with self.assertRaises(ValueError):
                end_run(run_dir, "failed", reason="x", facts={"state": "completed"})

    def test_a_run_ended_by_a_follower_has_a_record_before_its_status(self) -> None:
        from kura.executors.common import end_run

        with _workspace() as (_, run_dir):
            _status(run_dir, state="running", last_realization="realizations/r1.json", exit_code=None)
            end_run(run_dir, "interrupted", reason="the Pod is gone", error="gone")
            recorded = json.loads(next((run_dir / "realizations").glob("r1.ended-*.json")).read_text(encoding="utf-8"))
            self.assertEqual((recorded["kind"], recorded["state"], recorded["reason"]), ("run_end", "interrupted", "the Pod is gone"))
            written = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((written["state"], written["ended"], written["error"]), ("interrupted", recorded["at"], "gone"))

    def test_a_new_runner_epoch_never_goes_back(self) -> None:
        with _workspace() as (root, run_dir):
            _status(run_dir, state="running", epoch=9)
            self.assertEqual(runner.highest_epoch(root), 9)
            _status(run_dir, state="running", runner_epoch=11)
            self.assertEqual(runner.highest_epoch(root), 11)

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
        from kura.executors import read_run_status as observe_run

        with _workspace() as (_, run_dir):
            request = runner.write_launch_request(run_dir, executor="docker")
            runner.claim_request(request, 1)
            _launched(run_dir, request, state="running")
            with patch("kura.executors.observe.reconcile_docker") as reconcile, patch("kura.runner.ensure_runner") as wake:
                status = observe_run(run_dir)
            reconcile.assert_not_called()
            wake.assert_called_once()
            self.assertEqual(status["state"], "running")

    def _cleared_backoff(self):
        from kura.executors import observe

        observe._last_wake.clear()
        return observe

    def test_a_view_wakes_the_runner_for_any_unfinished_controlled_run_and_only_reads(self) -> None:
        observe = self._cleared_backoff()
        for state, extra in (("running", {}), ("completed", {"pod_id": "pod-1"}), ("compiled", None)):
            with _workspace() as (_, run_dir):
                request = runner.write_launch_request(run_dir, executor="runpod")
                runner.claim_request(request, 1)
                if extra is None:
                    _status(run_dir, state=state)  # claimed, launch not recorded yet
                else:
                    _launched(run_dir, request, state=state, **extra)
                before = (run_dir / "status.json").read_text(encoding="utf-8")
                observe._last_wake.clear()
                with patch("kura.executors.observe.reconcile_docker") as docker, \
                        patch("kura.executors.observe.reconcile_runpod") as pod, patch("kura.runner.ensure_runner") as wake:
                    observe.read_run_status(run_dir)
                docker.assert_not_called()
                pod.assert_not_called()
                wake.assert_called_once()
                self.assertEqual((run_dir / "status.json").read_text(encoding="utf-8"), before)

    def test_a_view_of_a_run_outside_the_runner_neither_observes_nor_wakes(self) -> None:
        observe = self._cleared_backoff()
        with _workspace() as (_, run_dir):
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "docker"}), encoding="utf-8")
            _status(run_dir, state="running", last_realization="realizations/r1.json")
            with patch("kura.executors.observe.reconcile_docker") as docker, patch("kura.runner.ensure_runner") as wake:
                self.assertEqual(observe.read_run_status(run_dir)["state"], "running")
            docker.assert_not_called()
            wake.assert_not_called()

    def test_status_says_who_follows_the_run(self) -> None:
        import argparse
        import io

        from kura.cli import cmd_run_status

        self._cleared_backoff()
        with _workspace() as (root, run_dir):
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "docker"}), encoding="utf-8")
            _status(run_dir, state="running", last_realization="realizations/r1.json")
            for controlled in (False, True):
                if controlled:
                    request = runner.write_launch_request(run_dir, executor="docker")
                    runner.claim_request(request, 1)
                    _launched(run_dir, request, state="running")
                stdout = io.StringIO()
                with patch("kura.cli._run_path", return_value=run_dir), patch("kura.runner.ensure_runner", return_value=False), \
                        patch("kura.runner.stopped_on_purpose", return_value=True), patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
                summary = json.loads(stdout.getvalue())["summary"]
                self.assertTrue(summary["status_updated_at"])
                self.assertIsNone(summary["last_observed_at"])
                if controlled:
                    self.assertIn("kura runner start", summary["runner"])
                    self.assertNotIn("observe_now", summary)
                else:
                    self.assertIn("kura run reconcile example", summary["observe_now"])
                    self.assertNotIn("runner", summary)


    def test_status_offers_reconcile_for_every_unfinished_run_outside_the_runner(self) -> None:
        import argparse
        import io

        from kura.cli import cmd_run_status

        self._cleared_backoff()
        # Publishing, and a failed run whose training state is not yet published, are written only by Docker.
        cases = [(state, "runpod", {}) for state in ("queued", "staged", "launching", "running")]
        cases += [("publishing", "docker", {}), ("failed", "docker", {"publication_state": "pending"})]
        for state, executor, extra in cases:
            with self.subTest(state=state, **extra), _workspace() as (_, run_dir):
                (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": executor}), encoding="utf-8")
                _status(run_dir, state=state, last_realization="realizations/r1.json", **extra)
                stdout = io.StringIO()
                with patch("kura.cli._run_path", return_value=run_dir), patch("kura.runner.ensure_runner", return_value=False), patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
                summary = json.loads(stdout.getvalue())["summary"]
                self.assertIn("kura run reconcile example", summary["observe_now"])
        # A finished run gets no hint.
        with _workspace() as (_, run_dir):
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "docker"}), encoding="utf-8")
            _status(run_dir, state="failed", publication_state="published", last_realization="realizations/r1.json")
            stdout = io.StringIO()
            with patch("kura.cli._run_path", return_value=run_dir), patch("kura.runner.ensure_runner", return_value=False), patch("sys.stdout", stdout):
                self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
            self.assertNotIn("observe_now", json.loads(stdout.getvalue())["summary"])


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
                    self.assertIn("every local training slot is in use", runner.local_slot_wait(root))
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


class RunnerStateTests(unittest.TestCase):
    def test_status_says_why_no_runner_is_running(self) -> None:
        with _workspace() as (root, _):
            self.assertEqual(runner.runner_state(root)[0], "never_started")
            runner.serve(root, spawn_child=lambda *_: self.fail("spawned"), sleep=lambda _: None)
            state, detail = runner.runner_state(root)
            self.assertEqual(state, "idle")
            self.assertIn("nothing left to control", detail)
            self.assertIn("the next launch starts it", detail)
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                self.assertEqual(runner.runner_state(root)[0], "running")
            runner.stop_runner(root)
            self.assertEqual(runner.runner_state(root)[0], "stopped_on_purpose")
            # A SIGTERM with no stop mark (logout, shutdown) records `stopped` and is named as that.
            (root / ".kura" / "runner" / "stopped").unlink()
            info = runner.runner_info(root)
            (root / ".kura" / "runner" / "runner.json").write_text(json.dumps({**info, "ended": "stopped"}), encoding="utf-8")
            state, detail = runner.runner_state(root)
            self.assertEqual(state, "stopped_by_signal")
            self.assertIn("logout or shutdown", detail)
            runner.stop_runner(root)
            # A runner that ended without recording why (a crash, a kill, or an older Kura).
            (root / ".kura" / "runner" / "stopped").unlink()
            info = runner.runner_info(root)
            info.pop("ended")
            (root / ".kura" / "runner" / "runner.json").write_text(json.dumps(info), encoding="utf-8")
            state, detail = runner.runner_state(root)
            self.assertEqual(state, "ended_unexpectedly")
            self.assertIn(".kura/runner/runner.log", detail)


class InvalidWorkspaceTests(unittest.TestCase):
    def test_an_invalid_workspace_is_named_once_and_follow_says_so(self) -> None:
        with _workspace() as (root, run_dir):
            (root / "workspace.yaml").write_text("schema_version: 2\nrunner: {local_slots: 0}\n", encoding="utf-8")
            self.assertIn("workspace.yaml is not valid", runner.local_slot_wait(root))
            request = runner.write_launch_request(run_dir, executor="docker")
            logged: list[str] = []
            polls = [0]

            def sleep(_):
                polls[0] += 1
                if polls[0] == 4:
                    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")

            def spawn(workspace, run_id, request):
                _launched(run_dir, request, state="completed", publication_state="completed")
                return Child(0)

            with patch.object(runner, "_log", side_effect=logged.append):
                runner.serve(root, spawn_child=spawn, sleep=sleep)
            waits = [line for line in logged if "workspace.yaml is not valid" in line]
            self.assertEqual(len(waits), 1, logged)
            self.assertEqual(runner.pending_requests(run_dir), [])

    def test_each_waiting_request_is_named_and_named_again_after_its_wait_clears(self) -> None:
        with _workspace() as (root, run_dir):
            second = root / "runs" / "second"
            (second / "realizations").mkdir(parents=True)
            _status(second, state="compiled")
            (root / "workspace.yaml").write_text("schema_version: 2\nrunner: {local_slots: 0}\n", encoding="utf-8")
            first_request = runner.write_launch_request(run_dir, executor="docker")
            second_request = runner.write_launch_request(second, executor="docker")
            logged: list[str] = []
            polls = [0]
            invalid = "schema_version: 2\nrunner: {local_slots: 0}\n"

            def sleep(_):
                polls[0] += 1
                if polls[0] == 3:
                    # The wait clears while the other run holds the only slot, then the file breaks again.
                    (root / "workspace.yaml").write_text("schema_version: 2\nrunner: {local_slots: 9}\n", encoding="utf-8")
                if polls[0] == 4:
                    (root / "workspace.yaml").write_text(invalid, encoding="utf-8")
                if polls[0] == 6:
                    raise KeyboardInterrupt

            with patch.object(runner, "_log", side_effect=logged.append), \
                    patch.object(runner, "claim_request", return_value=False):
                with self.assertRaises(KeyboardInterrupt):
                    runner.serve(root, spawn_child=lambda *_: Child(None), sleep=sleep)
            waits = [line for line in logged if "workspace.yaml is not valid" in line]
            self.assertEqual(sum(first_request.name in line for line in waits), 2, logged)
            self.assertEqual(sum(second_request.name in line for line in waits), 2, logged)

    def test_follow_names_the_invalid_workspace_not_another_run(self) -> None:
        with _workspace() as (root, run_dir):
            (root / "workspace.yaml").write_text("schema_version: 2\nrunner: {local_slots: 0}\n", encoding="utf-8")
            request = runner.write_launch_request(run_dir, executor="docker")
            out = io.StringIO()
            calls = [0]

            def sleep(_):
                calls[0] += 1
                runner._record_request_outcome(request, ".not-launched.json", "not_launched", error="test end")

            with patch.object(runner, "runner_alive", return_value=True):
                runner.follow(root, run_dir, request, out=out, sleep=sleep)
            text = out.getvalue()
            self.assertIn("waiting to launch: workspace.yaml is not valid", text)
            self.assertNotIn("another local", text)


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
        "options": {"max_lease": "3h"}, "runpod_config": {"gpu_type_ids": ["NVIDIA A40"]},
        "billing_confirmed_at": "2026-10-07T00:00:00+00:00", **extra,
    })


def _status_as_recorded(run_dir: Path, *_args, **_kwargs) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


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
            self.assertEqual((kwargs["yes"], kwargs["reattach"], kwargs["max_lease"]), (True, False, 3 * 3600))
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

            def stopped(run_dir_arg):
                _status(run_dir_arg, **{**json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8")), "pod_stopped_at": "t"})
                return {}

            with patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop, \
                 patch("kura.run_commands.launch._run_remote_locked") as remote:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(stop.call_args.args[1:], ())
            remote.assert_not_called()
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")

    def test_a_finished_run_that_left_its_pod_running_gets_the_pod_deleted(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            # Recovered from a create intent, or collected before a crash: finished, Pod still up.
            _runpod_launched(run_dir, request, state="interrupted")
            self.assertTrue(runner.run_unfinished(run_dir))

            def stopped(run_dir_arg):
                _status(run_dir_arg, **{**json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8")), "pod_stopped_at": "t"})
                return {}

            with patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(stop.call_args.args[1:], ())
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_finished_run_keeps_how_it_ended_however_often_its_pod_delete_fails(self) -> None:
        # Nothing on a finished run's Pod is needed, so a failing delete is retried uncounted
        # and never turns the run into recovery_required.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="completed")
            attempts = 5
            with patch("kura.executors.runpod.stop_runpod", side_effect=ValueError("api down")), \
                 patch("kura.notifications.notify") as notify:
                codes = [runner.work(root, "example", request.name) for _ in range(attempts)]
            self.assertEqual(codes, [1] * attempts)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "completed")
            notify.assert_called_once()
            self.assertIn("could not be deleted", notify.call_args.kwargs["subject"])

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
                # Retried later, never taken as "the job did not start".
                self.assertEqual(runner.work(root, "example", request.name), 1)
            stop.assert_not_called()

    def _exited_job(self, run_dir: Path, request: Path) -> None:
        # The job exited on the Pod, so each failed attempt is a failed collection.
        _runpod_launched(run_dir, request, state="running", remote_state="completed", remote_exit_code=0)
        (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")

    def test_a_failed_collection_is_retried_while_the_pod_exists_and_needs_no_person(self) -> None:
        # The Pod is disposable: the runner keeps collecting it, notifies once, and never hands the run to a person.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            self._exited_job(run_dir, request)
            attempts = 5

            def collected(*_args, **_kwargs):
                _runpod_launched(run_dir, request, state="completed", exit_code=0, remote_exit_code=0,
                                 publication_state="completed", downloaded_run="downloads/x", pod_stopped_at="t")
                return 0

            with patch("kura.executors.runpod.reconcile_runpod", side_effect=_status_as_recorded), \
                 patch("kura.notifications.notify") as notify:
                with patch("kura.run_commands.launch._run_remote_locked", return_value=1):
                    codes = [runner.work(root, "example", request.name) for _ in range(attempts)]
                self.assertEqual(codes, [1] * attempts)
                self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")
                self.assertTrue(runner.run_unfinished(run_dir))
                notify.assert_called_once()
                self.assertIn("could not be collected", notify.call_args.kwargs["subject"])
                with patch("kura.run_commands.launch._run_remote_locked", side_effect=collected):
                    self.assertEqual(runner.work(root, "example", request.name), 0)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((status["state"], status["downloaded_run"]), ("completed", "downloads/x"))
            self.assertFalse(runner.run_unfinished(run_dir))
            notify.assert_called_once()

    def test_a_collection_that_fails_until_the_pod_is_gone_ends_interrupted(self) -> None:
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            self._exited_job(run_dir, request)
            with patch("kura.executors.runpod.reconcile_runpod", side_effect=_status_as_recorded), \
                 patch("kura.run_commands.launch._run_remote_locked", return_value=1), \
                 patch("kura.notifications.notify") as notify:
                codes = [runner.work(root, "example", request.name) for _ in range(5)]

                def gone(run_dir_arg, *_args, **_kwargs):
                    _status(run_dir_arg, **{**_status_as_recorded(run_dir_arg), "pod_missing_at": "t"})
                    return _status_as_recorded(run_dir_arg)

                with patch("kura.executors.runpod.reconcile_runpod", side_effect=gone):
                    codes.append(runner.work(root, "example", request.name))
            self.assertEqual(codes, [1] * 5 + [0])
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "interrupted")
            self.assertFalse(runner.run_unfinished(run_dir))
            # One notice that collection is failing, and one that the run ended.
            self.assertEqual([call.kwargs["subject"].split(":")[0] for call in notify.call_args_list],
                             ["Kura run's outputs could not be collected yet", "Kura run interrupted"])

    def test_failures_while_the_job_still_runs_are_retried_and_never_notified(self) -> None:
        # The API or SSH being unreachable while training continues is not a failed collection:
        # ending the run then would leave the Pod to delete itself with outputs nobody collected.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
            attempts = 5
            with patch("kura.run_commands.launch._run_remote_locked", return_value=1), \
                 patch("kura.executors.runpod.reconcile_runpod", side_effect=_status_as_recorded), \
                 patch("kura.notifications.notify") as notify:
                codes = [runner.work(root, "example", request.name) for _ in range(attempts)]
            self.assertEqual(codes, [1] * attempts)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")
            notify.assert_not_called()

    def test_a_refused_runpod_key_notifies_once_and_the_run_continues_once_the_key_works(self) -> None:
        # Without a key RunPod accepts, the runner cannot learn the Pod is gone, so it would follow forever unseen.
        from kura.executors.runpod import RunPodAPIError
        from kura.secrets import MissingSecret

        refusals = [RunPodAPIError("RunPod GraphQL failed (401): unauthorized", status_code=401),
                    RunPodAPIError("RunPod GraphQL failed (403): forbidden", status_code=403),
                    MissingSecret("RUNPOD_API_KEY", "needed to reconcile a RunPod run")]
        with _workspace() as (root, run_dir), patch.dict("os.environ", {"RUNPOD_API_KEY": "rpa_SECRETVALUE123"}):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
            with patch("kura.notifications.notify") as notify:
                with patch("kura.executors.runpod.reconcile_runpod", side_effect=refusals * 2), \
                     patch("kura.run_commands.launch._run_remote_locked") as remote:
                    codes = [runner.work(root, "example", request.name) for _ in range(6)]
                self.assertEqual(codes, [1] * 6)
                remote.assert_not_called()
                self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")
                self.assertTrue(runner.run_unfinished(run_dir))
                notify.assert_called_once()
                subject, body = notify.call_args.kwargs["subject"], notify.call_args.kwargs["body"]
                self.assertIn("RunPod key", subject)
                self.assertIn("kura secrets set RUNPOD_API_KEY", body)
                self.assertIn("next try reads it", body)
                self.assertIn("lease", body)
                self.assertNotIn("rpa_SECRETVALUE123", subject + body)

                # The key works again: the next attempt follows the job as usual.
                with patch("kura.executors.runpod.reconcile_runpod", side_effect=_status_as_recorded), \
                     patch("kura.run_commands.launch._run_remote_locked", return_value=1) as remote:
                    self.assertEqual(runner.work(root, "example", request.name), 1)
                remote.assert_called_once()
                self.assertTrue(remote.call_args.kwargs["reattach"])
                notify.assert_called_once()

    def test_a_refused_key_after_the_job_exited_sends_only_the_key_notice(self) -> None:
        from kura.executors.runpod import RunPodAPIError

        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            self._exited_job(run_dir, request)
            with patch("kura.executors.runpod.reconcile_runpod", side_effect=RunPodAPIError("unauthorized", status_code=401)), \
                 patch("kura.notifications.notify") as notify:
                self.assertEqual([runner.work(root, "example", request.name) for _ in range(3)], [1] * 3)
            notify.assert_called_once()
            self.assertIn("RunPod key", notify.call_args.kwargs["subject"])

    def test_a_runpod_failure_that_is_not_the_key_is_retried_without_the_key_notice(self) -> None:
        from kura.executors.runpod import RunPodAPIError
        from kura.secrets import MissingSecret

        for failure in (RunPodAPIError("RunPod GraphQL failed (500): oops", status_code=500),
                        RunPodAPIError("RunPod GraphQL failed: bad input", status_code=400),
                        MissingSecret("HF_TOKEN", "needed elsewhere"), ValueError("RunPod API is unreachable: timeout")):
            with self.subTest(failure=str(failure)), _workspace() as (root, run_dir):
                request = _runpod_request(run_dir, notify=["ntfy"])
                runner.claim_request(request, 1)
                _runpod_launched(run_dir, request, state="running")
                (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
                with patch("kura.executors.runpod.reconcile_runpod", side_effect=failure), \
                     patch("kura.notifications.notify") as notify:
                    self.assertEqual([runner.work(root, "example", request.name) for _ in range(3)], [1] * 3)
                notify.assert_not_called()

    def test_a_launch_that_fails_before_recording_a_pod_fails_at_once_as_on_docker(self) -> None:
        # No Pod exists, so nothing can be lost or collected: the request ends on its first failure.
        for executor in ("runpod", "docker"):
            with self.subTest(executor=executor), _workspace() as (root, run_dir):
                request = _runpod_request(run_dir) if executor == "runpod" else runner.write_launch_request(run_dir, executor="docker")
                runner.claim_request(request, 1)
                with patch("kura.run_commands.launch._run_remote_locked", return_value=1) as remote, \
                     patch("kura.run_commands.launch.launch_run", return_value=1) as local:
                    self.assertEqual(runner.work(root, "example", request.name), 0)
                self.assertEqual((remote if executor == "runpod" else local).call_count, 1)
                self.assertEqual(runner.request_outcome(request)["kind"], "launch_failed")
                self.assertFalse(runner.run_unfinished(run_dir))
                self.assertNotEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "recovery_required")

    def test_a_launch_cut_short_after_its_create_intent_waits_for_the_lookup_by_name(self) -> None:
        # A Pod may exist under the intent's name: the next follower settles it before anything ends.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            def cut_short(*_args, **_kwargs):
                (run_dir / "realizations" / "r1.create-intent.json").write_text(json.dumps({
                    "executor": "runpod", "controlled_by": {"request": request.name}}), encoding="utf-8")
                return 1

            with patch("kura.run_commands.launch._run_remote_locked", side_effect=cut_short):
                self.assertEqual(runner.work(root, "example", request.name), 1)
            self.assertIsNone(runner.request_outcome(request))
            self.assertTrue(runner.run_unfinished(run_dir))

    def test_a_pod_runpod_no_longer_runs_is_deleted_not_followed(self) -> None:
        # The follower's one look at RunPod decides: a Pod that exists but no longer runs ends the run as
        # recorded and is deleted, instead of a second check refusing the reattach and a retry deleting it.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")

            def exited(run_dir_arg, *_args, **_kwargs):
                _status(run_dir_arg, **{**_status_as_recorded(run_dir_arg), "state": "unknown"})
                return _status_as_recorded(run_dir_arg)

            def stopped(run_dir_arg):
                _status(run_dir_arg, **{**_status_as_recorded(run_dir_arg), "pod_stopped_at": "t"})
                return {}

            with patch("kura.executors.runpod.reconcile_runpod", side_effect=exited), \
                 patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop, \
                 patch("kura.run_commands.launch._run_remote_locked") as remote:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            remote.assert_not_called()
            stop.assert_called_once()
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "unknown")
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_capacity_wait_cut_short_continues_whether_or_not_it_wrote_a_create_intent(self) -> None:
        # The follower can die before or after a refused create in the wait wrote the create intent. Either
        # way no Pod exists (the lookup by name finds none), so the confirmed launch continues the wait.
        from kura.executors.common import can_start, unresolved_create_intents

        wait = {"started_at": "2026-10-07T00:00:00+00:00", "attempts": 3}
        for order in ("before the intent", "after the intent"):
            with self.subTest(order=order), _workspace() as (root, run_dir), patch.dict("os.environ", {"RUNPOD_API_KEY": "api-secret"}):
                request = _runpod_request(run_dir)
                runner.claim_request(request, 1)
                _status(run_dir, state="queued", host="runpod", capacity_wait=wait)
                if order == "after the intent":
                    (run_dir / "realizations" / "r1.create-intent.json").write_text(json.dumps({
                        "kind": "pod_create_intent", "executor": "runpod", "realization_id": "r1", "pod_name": "kura-example-r1",
                        "requested_at": "2026-10-07T00:01:00+00:00", "controlled_by": {"request": request.name}}), encoding="utf-8")
                with patch("kura.executors.runpod._discover_pods", return_value=[]), \
                     patch("kura.run_commands.launch._run_remote_locked", return_value=0) as remote:
                    runner.work(root, "example", request.name)
                remote.assert_called_once()
                self.assertFalse(remote.call_args.kwargs["reattach"])
                self.assertIsNone(runner.request_outcome(request))
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                self.assertEqual((status["state"], status["capacity_wait"]), ("queued", wait))
                self.assertTrue(can_start(status))
                self.assertEqual(unresolved_create_intents(run_dir), [])
                self.assertTrue(runner.run_unfinished(run_dir))

    def test_a_continued_wait_whose_launch_fails_ends_the_request_as_any_failed_launch(self) -> None:
        with _workspace() as (root, run_dir), patch.dict("os.environ", {"RUNPOD_API_KEY": "api-secret"}):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _status(run_dir, state="queued", host="runpod", capacity_wait={"attempts": 1})
            (run_dir / "realizations" / "r1.create-intent.json").write_text(json.dumps({
                "kind": "pod_create_intent", "executor": "runpod", "realization_id": "r1", "pod_name": "kura-example-r1",
                "requested_at": "2026-10-07T00:01:00+00:00", "controlled_by": {"request": request.name}}), encoding="utf-8")
            with patch("kura.executors.runpod._discover_pods", return_value=[]), \
                 patch("kura.run_commands.launch._run_remote_locked", return_value=1):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(runner.request_outcome(request)["kind"], "launch_failed")
            # The run no longer says it waits for a GPU nobody looks for.
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "launch_failed")

    def test_a_collected_run_whose_pod_delete_failed_keeps_how_it_ended(self) -> None:
        # The controller collected and its stop failed: the follower retries the delete, never the collection.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)

            def collected_but_not_stopped(*_args, **_kwargs):
                _runpod_launched(run_dir, request, state="completed", exit_code=0, remote_exit_code=0, downloaded_run="downloads/x")
                return 0

            with patch("kura.run_commands.launch._run_remote_locked", side_effect=collected_but_not_stopped):
                self.assertEqual(runner.work(root, "example", request.name), 1)
            attempts = 5
            with patch("kura.executors.runpod.stop_runpod", side_effect=ValueError("api down")):
                codes = [runner.work(root, "example", request.name) for _ in range(attempts)]
            self.assertEqual(codes, [1] * attempts)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((status["state"], status["exit_code"]), ("completed", 0))

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
            self.assertEqual(stop.call_args.args[1:], ())
            self.assertTrue(runner.stop_done(run_dir))

    def test_a_stop_whose_delete_fails_is_retried_with_backoff(self) -> None:
        from kura.executors.common import StopRequested

        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir)
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"pid": "42"}), encoding="utf-8")
            with patch("kura.run_commands.launch._run_remote_locked", side_effect=StopRequested()), \
                 patch("kura.executors.runpod.reconcile_runpod", side_effect=lambda run_dir_arg, *_a, **_k: json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8"))), \
                 patch("kura.executors.runpod.stop_runpod", side_effect=ValueError("RunPod API returned 500")):
                # Non-zero, so the runner waits its backoff before the next try instead of polling every 2 s.
                self.assertEqual(runner.work(root, "example", request.name), 1)
            self.assertFalse(runner.stop_done(run_dir))

    def test_a_pod_whose_job_never_started_is_deleted_however_often_the_delete_fails(self) -> None:
        # Deleting a Pod with nothing to collect is retry-safe, so a failing delete never hands the run to a person.
        with _workspace() as (root, run_dir):
            request = _runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            _runpod_launched(run_dir, request, state="running")
            attempts = 5
            with patch("kura.executors.runpod.stop_runpod", side_effect=ValueError("api down")), \
                 patch("kura.notifications.notify") as notify:
                codes = [runner.work(root, "example", request.name) for _ in range(attempts)]
            self.assertEqual(codes, [1] * attempts)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")
            self.assertTrue(runner.run_unfinished(run_dir))
            self.assertEqual(json.loads(runner._sibling(request, ".delete-failures.json").read_text(encoding="utf-8"))["count"], attempts)
            notify.assert_called_once()
            self.assertIn("could not be deleted", notify.call_args.kwargs["subject"])

            def stopped(run_dir_arg):
                _status(run_dir_arg, **{**json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8")), "pod_stopped_at": "t"})
                return {}

            with patch("kura.executors.runpod.stop_runpod", side_effect=stopped), \
                 patch("kura.notifications.notify"):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")
            self.assertFalse(runner.run_unfinished(run_dir))

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
                    self.assertEqual((runpod["yes"], runpod["max_lease"]), (True, launch.DEFAULT_MAX_LEASE_SEC))
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
            self.assertEqual(stop.call_args.args[1:], ())

    def test_a_render_pod_whose_delete_fails_is_retried_and_notified_once(self) -> None:
        # As for a training run: a refused key, or any failed delete, never loops unseen.
        from kura.executors.runpod import RunPodAPIError

        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir, notify=["ntfy"])
            runner.claim_request(request, 1)
            runner._first_attempt(request)
            _runpod_launched(run_dir, request, state="running", last_step=2, total_steps=5)
            with patch("kura.executors.runpod.stop_runpod", side_effect=RunPodAPIError("unauthorized", status_code=401)), \
                    patch("kura.notifications.notify") as notify:
                self.assertEqual([runner.work(root, "example", request.name) for _ in range(3)], [1] * 3)
            notify.assert_called_once()
            self.assertIn("could not be deleted", notify.call_args.kwargs["subject"])
            self.assertIn("kura secrets set RUNPOD_API_KEY", notify.call_args.kwargs["body"])

    def test_a_render_whose_follower_died_is_interrupted_and_its_pod_deleted_not_continued(self) -> None:
        with _workspace() as (root, run_dir):
            request = _render_runpod_request(run_dir)
            runner.claim_request(request, 1)
            runner._first_attempt(request)  # the follower that was rendering died
            _runpod_launched(run_dir, request, state="running", last_step=2, total_steps=5)

            def stopped(run_dir):
                _mutate = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                _status(run_dir, **{**_mutate, "pod_stopped_at": "t"})
                return {}

            with patch("kura.run_commands.render_runpod.launch_render_runpod") as render, \
                    patch("kura.executors.runpod.stop_runpod", side_effect=stopped) as stop, \
                    patch("kura.render.write_realization", wraps=__import__("kura.render", fromlist=["x"]).write_realization) as realization:
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
                # From a file: a script passed as one argument meets the Windows argument conversion.
                script_path = root / "start.sh"
                script_path.write_text(script, encoding="utf-8", newline="\n")
                checked = subprocess.run(["sh", "-n", str(script_path)], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertNotIn('sleep "$KURA_MAX_LEASE_SEC"', script)
            realization = json.loads(next(p for p in (run_dir / "realizations").glob("*.json") if "." not in p.stem).read_text(encoding="utf-8"))
            self.assertEqual(realization["controlled_by"], {"request": "r.launch.json"})

    def test_a_render_pod_takes_a_lease_change(self) -> None:
        from kura.run_commands.runpod_ssh import change_runpod_lease

        with _workspace() as (root, run_dir):
            # Older render realizations also say `lease_guard: deadline_file`; neither form is refused.
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"executor": "runpod", "purpose": "comfyui-render", "pod": {"id": "pod-1"}}), encoding="utf-8")
            _status(run_dir, last_realization="realizations/r1.json", pod_id="pod-1", state="running")
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
            stop.assert_called_once()
            self.assertEqual(stop.call_args.args[0].resolve(), run_dir.resolve())

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
                        patch.object(render_runpod, "record_pod_lease_deadline"), \
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
