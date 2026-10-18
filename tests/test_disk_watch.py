"""The job runner's disk watch stops a local run before a store it writes to falls below the reserve
(docs/adr/local-disk-protection.md), recording the stop and its reason before `docker stop`."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from kura import runner
from kura.run_commands.plan import local_min_free_gib, local_reserve_problem
from kura.storage import StorageStatus

GIB = 1024**3


def _status_of(free_gib: float, *, confidence: str = "exact", backing: str = "C:") -> StorageStatus:
    return StorageStatus(
        path="x", probe="x", backing_id=backing, backing_kind="native", linux_free_bytes=int(free_gib * GIB),
        linux_total_bytes=1000 * GIB, host_free_bytes=None, effective_free_bytes=int(free_gib * GIB),
        confidence=confidence, mount={},
    )


@contextlib.contextmanager
def _launched_run(*, safety: dict | None = None, workspace_config: dict | None = None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "workspace.yaml").write_text(yaml.safe_dump({"schema_version": 2, **(workspace_config or {})}), encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({"type": "train", **({"safety": safety} if safety else {})}), encoding="utf-8")
        request = runner.write_launch_request(run_dir, executor="docker")
        runner.claim_request(request, 1)
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "controlled_by": {"request": request.name, "epoch": 1}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
        yield root, run_dir, request


class DiskWatchTests(unittest.TestCase):
    def _work(self, root: Path, request: Path, free_gib: float, *, confidence: str = "exact", states=None, clock=None, sleep=None):
        states = iter(states or [{"state": "running"}, {"state": "running"}, {"state": "interrupted", "publication_state": "not-required"}])
        seen_at_stop: list[dict] = []

        def stop(run_dir, **kwargs):
            # The intent and its reason are on disk before the container is stopped.
            seen_at_stop.append(json.loads(runner._sibling(request, ".stop.json").read_text(encoding="utf-8")))
            seen_at_stop[-1]["passed"] = kwargs
            return {}

        # runner.work sets KURA_RUNNER_EPOCH for its process; keep it out of later tests.
        with patch.dict(os.environ), patch("kura.run_commands.plan.probe_storage", return_value=_status_of(free_gib, confidence=confidence)) as probe, \
                patch("kura.executors.docker.stop_docker", side_effect=stop) as stopped, \
                patch("kura.executors.docker.reconcile_docker", side_effect=lambda *_a, **_k: next(states)):
            code = runner.work(root, "example", request.name, sleep=sleep or (lambda _: None), **({"clock": clock} if clock else {}))
        return code, stopped, seen_at_stop, probe

    def test_a_store_below_the_reserve_stops_the_run_after_recording_why(self) -> None:
        with _launched_run() as (root, run_dir, request):
            code, stopped, seen, _ = self._work(root, request, 5)
        self.assertEqual(code, 0)
        stopped.assert_called_once()
        [request_at_stop] = seen
        reason = request_at_stop["reason"]
        self.assertIn("5 GiB effective free on C:", reason)
        self.assertIn("requires at least 20 GiB", reason)
        self.assertIn("kura run resume example", reason)
        # The stop record is written with the reason.
        self.assertEqual(request_at_stop["passed"], {"reason": reason})

    def test_a_store_above_the_reserve_is_left_running(self) -> None:
        with _launched_run() as (root, _run_dir, request):
            _, stopped, _, _ = self._work(root, request, 25, states=[{"state": "running"}, {"state": "completed", "publication_state": "completed"}])
        stopped.assert_not_called()

    def test_an_unmeasurable_backing_stops_the_run_unless_the_run_accepts_the_risk(self) -> None:
        with _launched_run() as (root, _run_dir, request):
            _, stopped, seen, _ = self._work(root, request, 900, confidence="unknown")
        stopped.assert_called_once()
        self.assertIn("unknown physical backing free space", seen[0]["reason"])
        with _launched_run(safety={"allow_storage_risk": True}) as (root, _run_dir, request):
            _, stopped, _, _ = self._work(root, request, 900, confidence="unknown",
                                          states=[{"state": "running"}, {"state": "completed", "publication_state": "completed"}])
        stopped.assert_not_called()

    def test_the_workspace_reserve_setting_applies(self) -> None:
        with _launched_run(workspace_config={"docker": {"min_free_gb": 30}}) as (root, _run_dir, request):
            _, stopped, seen, _ = self._work(root, request, 25)
        stopped.assert_called_once()
        self.assertIn("requires at least 30 GiB", seen[0]["reason"])

    def test_the_watch_probes_about_every_30_seconds_not_at_every_poll(self) -> None:
        now = [0.0]
        polls = [{"state": "running"}] * 7 + [{"state": "completed", "publication_state": "completed"}]
        with _launched_run() as (root, _run_dir, request):
            _, stopped, _, probe = self._work(root, request, 25, states=polls, clock=lambda: now[0],
                                              sleep=lambda _: now.__setitem__(0, now[0] + 10))
        stopped.assert_not_called()
        # Seven running polls 10 s apart: checked at 0, 30, and 60 s, each time the workspace and the HF cache.
        self.assertEqual([call.kwargs["role"] for call in probe.call_args_list], ["workspace", "hf_cache"] * 3)

    def test_a_user_stop_already_asked_for_keeps_its_own_record(self) -> None:
        with _launched_run() as (root, run_dir, request):
            runner.write_stop_request(run_dir)
            _, stopped, seen, _ = self._work(root, request, 5)
        stopped.assert_called_once()
        self.assertNotIn("reason", seen[0])


class ReserveOwnerTests(unittest.TestCase):
    def test_the_reserve_defaults_to_20_gib(self) -> None:
        self.assertEqual(local_min_free_gib({}), 20)

    def test_every_store_a_run_writes_is_watched_once_per_backing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root.parent / (root.name + "-cache")
            config = {"docker": {"hf_cache": str(outside), "mounts": [{"source": "data", "target": "/data", "mode": "ro"}, {"source": "extra", "target": "/extra", "mode": "rw"}]}}
            probed: list[str] = []

            def probe(path, _config=None, *, role=None):
                probed.append(role)
                return _status_of(5 if role == "mount:/extra" else 50, backing="D:" if role == "mount:/extra" else "C:")

            with patch("kura.run_commands.plan.probe_storage", side_effect=probe):
                problem = local_reserve_problem(root, {}, config)
        self.assertEqual(probed, ["workspace", "hf_cache", "mount:/extra"])
        self.assertIn(f"{root / 'extra'} has only 5 GiB effective free on D:", problem)

    def test_the_watch_says_when_the_wsl_virtual_disk_is_the_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch("kura.storage.is_wsl", return_value=True))
            stack.enter_context(patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4", "target": "/", "source": "/dev/sdd"}))
            stack.enter_context(patch("kura.storage._auto_wsl_host_drive", return_value="D:"))
            stack.enter_context(patch("kura.storage._windows_drive_free_bytes", return_value=1600 * GIB))
            stack.enter_context(patch("kura.storage.shutil.disk_usage", return_value=Mock(total=1000 * GIB, used=990 * GIB, free=10 * GIB)))
            stack.enter_context(patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu-24.04"}))
            problem = local_reserve_problem(Path(directory).resolve(), {}, {})
        self.assertIn("the WSL virtual disk is the limit", problem)


class StopReasonRecordTests(unittest.TestCase):
    def test_the_reason_reaches_the_stop_record_and_the_interrupted_status(self) -> None:
        from tests.test_docker_create_intent import SPEC, _docker, _run
        from kura.executors.docker import launch_docker, stop_docker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            with _docker(lambda command: subprocess.CompletedProcess(command, 0, "container-1\n", "")):
                _, realization_id = launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[])
            after_stop = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            stopped = json.dumps({"Running": False, "ExitCode": 137, "FinishedAt": after_stop})
            with _docker([subprocess.CompletedProcess([], 0, "container-1\n", ""), subprocess.CompletedProcess([], 0, stopped, "")]):
                status = stop_docker(run_dir, reason="below the reserve")
            [stop] = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob(f"{realization_id}.stop-*.json")]
        self.assertEqual(stop["reason"], "below the reserve")
        self.assertEqual((status["state"], status["stop_reason"]), ("interrupted", "below the reserve"))

    def test_following_the_run_says_why_it_stopped(self) -> None:
        import io

        from kura.fsio import file_lock

        with _launched_run() as (root, run_dir, request):
            (run_dir / "status.json").write_text(json.dumps({"state": "interrupted", "last_realization": "realizations/r1.json",
                                                            "stop_reason": "stopped by the disk watch: low"}), encoding="utf-8")
            out = io.StringIO()
            with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
                self.assertNotEqual(runner.follow(root, run_dir, request, sleep=lambda _: None, out=out), 0)
        self.assertIn("the run was stopped by the disk watch: low", out.getvalue())

    def test_the_notice_says_why_the_run_stopped(self) -> None:
        with _launched_run() as (_root, run_dir, request):
            payload = json.loads(request.read_text(encoding="utf-8"))
            payload["notify"] = ["ntfy"]
            request.write_text(json.dumps(payload), encoding="utf-8")
            with patch("kura.notifications.notify") as notify:
                runner._notify_finished(run_dir, request, {"state": "interrupted", "stop_reason": "below the reserve"})
        self.assertIn("below the reserve", notify.call_args.kwargs["body"])


if __name__ == "__main__":
    unittest.main()
