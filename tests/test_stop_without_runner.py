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


if __name__ == "__main__":
    unittest.main()
