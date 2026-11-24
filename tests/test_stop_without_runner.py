"""`kura run stop` with no runner alive stops a runner-controlled run itself, on Docker and RunPod alike."""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from kura import runner
from kura.executors.common import _run_operation_lock
from kura.run_commands.plan import stop_run


@contextmanager
def _runner_launched(executor: str, realization: dict, extra: dict | None = None, **status):
    """A run whose runner-controlled launch is recorded as running, and no runner holds the lock."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        request = runner.write_launch_request(run_dir, executor=executor, extra=extra)
        runner.claim_request(request, 1)
        runner._first_attempt(request)
        (run_dir / "realizations" / "r1.json").write_text(
            json.dumps({"id": "r1", **realization, "controlled_by": {"request": request.name, "epoch": 1}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"run_id": "example", "state": "running", "last_realization": "realizations/r1.json", **status}),
                                             encoding="utf-8")
        assert runner.run_unfinished(run_dir) and not runner.runner_alive(root)
        yield root, run_dir


def _stop(run_dir: Path) -> tuple[int, str]:
    stderr = io.StringIO()
    with patch("kura.run_commands.plan._run_path", return_value=run_dir), \
         patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
        code = stop_run("example", yes=True)
    return code, stderr.getvalue()


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def _stop_records(run_dir: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob("r1.stop-*.json")]


class StopWithoutRunnerTests(unittest.TestCase):
    def _assert_stopped_and_left_alone(self, root: Path, run_dir: Path, executor: str) -> None:
        status = _status(run_dir)
        self.assertEqual((status["state"], status["exit_code"]), ("interrupted", None))
        [record] = _stop_records(run_dir)
        self.assertEqual((record["executor"], record["outcome"]), (executor, "stopped"))
        # A runner started later finds nothing to settle, so it never contradicts the stop.
        self.assertFalse(runner.run_unfinished(run_dir))
        runner.serve(root, spawn_child=lambda *_: self.fail("a later runner followed a stopped run"), sleep=lambda _: None)
        self.assertEqual(_status(run_dir)["state"], "interrupted")

    def test_a_runpod_run_has_its_pod_deleted_by_the_cli(self) -> None:
        with _runner_launched("runpod", {"executor": "runpod", "pod": {"id": "pod-1"}}, pod_id="pod-1") as (root, run_dir):
            with patch("kura.executors.runpod.runpod_api_key", return_value="key"), \
                 patch("kura.executors.runpod._runpod_pods_named", return_value=[]), \
                 patch("kura.executors.runpod._runpod_request", return_value={}) as api:
                code, stderr = _stop(run_dir)
            self.assertEqual(code, 0, stderr)
            self.assertIn(("DELETE", "/pods/pod-1"), [call.args[:2] for call in api.call_args_list])
            self.assertTrue(_status(run_dir)["pod_stopped_at"])
            self._assert_stopped_and_left_alone(root, run_dir, "runpod")

    def test_a_docker_run_has_its_container_stopped_by_the_cli(self) -> None:
        after_stop = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        commands = []

        def docker(command, *args, **kwargs):
            commands.append(command[:2])
            if command[:2] == ["docker", "stop"]:
                return subprocess.CompletedProcess(command, 0, "kura-c\n", "")
            if command[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(command, 0, json.dumps({"Running": False, "ExitCode": 137, "FinishedAt": after_stop}), "")
            raise AssertionError(f"unexpected command {command}")

        with _runner_launched("docker", {"executor": "docker", "container": {"name": "kura-c"}}, container_name="kura-c") as (root, run_dir):
            with patch("kura.executors.docker.subprocess.run", side_effect=docker):
                code, stderr = _stop(run_dir)
            self.assertEqual(code, 0, stderr)
            self.assertIn(["docker", "stop"], commands)
            self._assert_stopped_and_left_alone(root, run_dir, "docker")

    def test_a_local_render_cut_short_is_left_for_the_runner_to_record(self) -> None:
        # A local render runs inside the runner's follower; with both gone nothing is left to stop or bill,
        # the CLI records nothing, and a runner started later records the render as interrupted.
        with _runner_launched("render-local", {}) as (root, run_dir):
            (run_dir / "realizations" / "r1.json").unlink()
            (run_dir / "status.json").write_text(json.dumps({"run_id": "example", "state": "running"}), encoding="utf-8")
            code, stderr = _stop(run_dir)
            self.assertEqual(code, 1)
            self.assertIn("`kura runner start`", stderr)
            self.assertEqual((_status(run_dir)["state"], _stop_records(run_dir)), ("running", []))
            [request] = runner.launch_requests(run_dir)
            with patch("kura.render.remove_leftover_stages", return_value=[]), patch("kura.runner._notify_text"):
                self.assertEqual(runner.work(root, "example", request.name), 0)
            self.assertEqual(_status(run_dir)["state"], "interrupted")
            self.assertFalse(runner.run_unfinished(run_dir))

    def test_a_runpod_launch_not_recorded_yet_is_never_continued_after_the_stop(self) -> None:
        # A claimed RunPod request whose follower died before recording a launch (a capacity wait between
        # rounds): the CLI has no Pod to delete, and the next runner records it as not launched, never creating one.
        confirmed = {"options": {"max_lease": "3h"}, "runpod_config": {"gpu_type_ids": ["NVIDIA A40"]},
                     "billing_confirmed_at": "2026-10-22T00:00:00+00:00"}
        with _runner_launched("runpod", {}, confirmed) as (root, run_dir):
            (run_dir / "realizations" / "r1.json").unlink()
            (run_dir / "status.json").write_text(json.dumps({"run_id": "example", "state": "queued"}), encoding="utf-8")
            code, stderr = _stop(run_dir)
            self.assertEqual(code, 1)
            self.assertIn("`kura runner start` records the stop", stderr)
            [request] = runner.launch_requests(run_dir)
            with patch("kura.runner._remote") as remote:
                self.assertEqual(runner.work(root, "example", request.name), 0)
            remote.assert_not_called()
            self.assertEqual(runner.request_outcome(request)["kind"], "not_launched")
            self.assertFalse(runner.run_unfinished(run_dir))


RUNS = (("docker", {"executor": "docker", "container": {"name": "kura-c"}}, {"container_name": "kura-c"}),
        ("runpod", {"executor": "runpod", "pod": {"id": "pod-1"}}, {"pod_id": "pod-1"}))


@contextmanager
def _effects(during=lambda: None, *, fail: bool = False):
    """Fake `docker stop` and the RunPod API; yields the list of stops or deletes the CLI itself sent.
    `during` runs while one is in flight; `fail` makes it fail as Docker or RunPod would."""
    sent = []
    after_stop = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def docker(command, *args, **kwargs):
        if command[:2] == ["docker", "stop"]:
            sent.append("docker stop")
            during()
            return subprocess.CompletedProcess(command, 1 if fail else 0, "", "Cannot connect to the Docker daemon" if fail else "")
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({"Running": False, "ExitCode": 137, "FinishedAt": after_stop}), "")
        raise AssertionError(f"unexpected command {command}")

    def api(method, path, *args, **kwargs):
        if method == "DELETE":
            sent.append(f"DELETE {path}")
            during()
        if fail:
            raise ValueError("RunPod API is unreachable: down")
        return {}

    with patch("kura.executors.docker.subprocess.run", side_effect=docker), \
         patch("kura.executors.runpod.runpod_api_key", return_value="key"), \
         patch("kura.executors.runpod._runpod_pods_named", return_value=[]), \
         patch("kura.executors.runpod._runpod_request", side_effect=api):
        yield sent


class OneStopActorTests(unittest.TestCase):
    """Whoever holds the run's controller lock carries out a stop: a follower, or the CLI for the time
    it acts. The same holds for a container and for a Pod."""

    def test_a_follower_that_took_the_run_after_the_check_carries_out_the_stop(self) -> None:
        for executor, realization, status in RUNS:
            with self.subTest(executor), _runner_launched(executor, realization, **status) as (_root, run_dir):
                [request] = runner.launch_requests(run_dir)

                def follower_stops(_seconds):
                    runner._record_request_outcome(request, ".stop-done.json", "stop_done")

                # The first check saw no follower; one took the controller lock before the CLI acted.
                present = runner.follower_present
                checks = []
                with _run_operation_lock(run_dir, "controller"), \
                     patch.object(runner, "follower_present", side_effect=lambda *args: bool(checks.append(1) or len(checks) > 1) and present(*args)), \
                     patch("time.sleep", side_effect=follower_stops), _effects() as sent:
                    code, stderr = _stop(run_dir)
                self.assertEqual(sent, [])
                self.assertEqual(code, 0, stderr)
                self.assertIn("the runner's follower stops the run", stderr)
                self.assertTrue(runner._sibling(request, ".stop.json").exists())
                self.assertEqual(_stop_records(run_dir), [])

    def test_no_follower_can_take_the_run_while_the_cli_stops_it(self) -> None:
        for executor, realization, status in RUNS:
            with self.subTest(executor), _runner_launched(executor, realization, **status) as (root, run_dir):
                [request] = runner.launch_requests(run_dir)
                seen = []

                def a_follower_starts():
                    seen.append(runner._held(run_dir))
                    # A follower spawned now steps aside, as it does for any other holder; it observes nothing.
                    with patch("kura.executors.docker.reconcile_docker", side_effect=AssertionError("a follower reconciled during the stop")), \
                         patch("kura.executors.runpod.reconcile_runpod", side_effect=AssertionError("a follower reconciled during the stop")):
                        seen.append(runner.work(root, "example", request.name))

                with _effects(a_follower_starts) as sent:
                    code, stderr = _stop(run_dir)
                self.assertEqual(code, 0, stderr)
                self.assertEqual(len(sent), 1)
                self.assertEqual(seen, [True, 0])
                self.assertEqual(_status(run_dir)["state"], "interrupted")
                self.assertFalse(runner._held(run_dir))

    def test_a_stop_a_follower_carried_out_before_the_cli_took_the_lock_is_not_carried_out_again(self) -> None:
        for executor, realization, status in RUNS:
            with self.subTest(executor), _runner_launched(executor, realization, **status) as (root, run_dir):
                [request] = runner.launch_requests(run_dir)

                def a_follower_came_and_went(*_args):
                    # Between the check and the lock a follower took the run, stopped it, and left.
                    runner.write_stop_request(run_dir)
                    runner._record_request_outcome(request, ".stop-done.json", "stop_done")
                    return False

                with patch.object(runner, "follower_present", side_effect=a_follower_came_and_went), _effects() as sent:
                    code, stderr = _stop(run_dir)
                self.assertEqual((code, sent), (0, []), stderr)
                self.assertEqual(_stop_records(run_dir), [])

    def test_a_stop_waiting_on_another_stop_command_that_gave_up_carries_the_stop_out_itself(self) -> None:
        # The controller lock is held by another `kura run stop`, not a follower. When it leaves without
        # stopping the run and no runner is alive, nobody is left to read the request, so this command acts.
        for executor, realization, status in RUNS:
            with self.subTest(executor), _runner_launched(executor, realization, **status) as (_root, run_dir):
                other = _run_operation_lock(run_dir, "controller")
                other.__enter__()
                waits = []

                def the_other_command_gives_up(_seconds):
                    waits.append(1)
                    if len(waits) > 1:
                        raise AssertionError("still waiting for a follower that does not exist")
                    other.__exit__(None, None, None)

                try:
                    with patch("time.sleep", side_effect=the_other_command_gives_up), _effects() as sent:
                        code, stderr = _stop(run_dir)
                finally:
                    if not waits:
                        other.__exit__(None, None, None)
                self.assertEqual((code, len(sent)), (0, 1), stderr)
                self.assertEqual(_status(run_dir)["state"], "interrupted")

    def test_a_run_without_a_launch_request_held_by_another_stop_command_is_refused_cleanly(self) -> None:
        for executor, realization, status in RUNS:
            with self.subTest(executor), _runner_launched(executor, realization, **status) as (_root, run_dir):
                for path in runner.requests_dir(run_dir).iterdir():
                    path.unlink()
                with _run_operation_lock(run_dir, "controller"), _effects() as sent:
                    code, stderr = _stop(run_dir)
                self.assertEqual((code, sent), (1, []))
                self.assertIn("cannot stop run: another controller operation is already active", stderr)

    def test_a_failed_direct_stop_says_the_stop_stays_requested_only_when_a_request_is_on_disk(self) -> None:
        for executor, realization, status in RUNS:
            with self.subTest(executor, request="on disk"), _runner_launched(executor, realization, **status) as (_root, run_dir):
                with _effects(fail=True) as sent:
                    code, stderr = _stop(run_dir)
                self.assertEqual((code, len(sent)), (1, 1))
                self.assertIn("cannot stop run: ", stderr)
                self.assertTrue(runner._sibling(runner.latest_request(run_dir), ".stop.json").exists())
                self.assertIn("The stop stays requested", stderr)
                self.assertIn("`kura runner start` carries it out", stderr)
            with self.subTest(executor, request="none"), _runner_launched(executor, realization, **status) as (_root, run_dir):
                # A run no runner controls: the stop leaves no request, so nothing is promised.
                for path in runner.requests_dir(run_dir).iterdir():
                    path.unlink()
                with _effects(fail=True) as sent:
                    code, stderr = _stop(run_dir)
                self.assertEqual((code, len(sent)), (1, 1))
                self.assertIn("cannot stop run: ", stderr)
                self.assertNotIn("stays requested", stderr)
                self.assertNotIn("kura runner start", stderr)


if __name__ == "__main__":
    unittest.main()
