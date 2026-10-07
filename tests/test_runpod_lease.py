"""Changing a running Pod's lease, and the warning when training would outlast it."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.run_commands import runpod_ssh
from kura.run_commands.runpod_ssh import change_runpod_lease, lease_shortfall, record_lease_deadline


@contextmanager
def _run(**status):
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1", "cost_per_h": 0.27}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1", "last_realization": "realizations/r1.json", **status}), encoding="utf-8")
        yield run_dir


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@contextmanager
def _pod(current: str):
    commands = []

    def run(command, *args, **kwargs):
        script = command[-1]
        commands.append(script)
        if script.startswith("date +%s"):
            return subprocess.CompletedProcess(command, 0, f"{int(time.time())}\n{current}\n", "")
        # The Pod computes the new deadline from its own clock.
        added = int(script.split("+ ", 1)[1].split(" ", 1)[0])
        return subprocess.CompletedProcess(command, 0, f"{int(time.time()) + added}\n", "")

    with patch.object(runpod_ssh, "_runpod_ssh_details", return_value={"ip": "h", "port": 22, "key": "k"}), \
         patch.object(runpod_ssh.subprocess, "run", side_effect=run), patch("sys.stderr", io.StringIO()):
        yield commands


class LeaseChangeTests(unittest.TestCase):
    def test_a_confirmed_change_moves_the_deadline_on_the_pod_and_records_it(self) -> None:
        with _run() as run_dir, _pod(str(int(time.time()) + 600)) as commands:
            self.assertEqual(change_runpod_lease(run_dir, 18 * 3600, yes=False, input_stream=Tty("y\n")), 0)
            self.assertIn("mv /tmp/kura-lease-deadline.tmp /tmp/kura-lease-deadline", commands[-1])
            [record] = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob("r1.lease-*.json")]
            self.assertEqual(record["kind"], "lease")
            self.assertAlmostEqual(record["deadline_epoch"], int(time.time()) + 18 * 3600, delta=5)
            self.assertIn("previous_deadline_epoch", record)

    def test_without_a_terminal_or_yes_the_lease_is_not_changed(self) -> None:
        with _run() as run_dir, _pod(str(int(time.time()) + 600)) as commands:
            with self.assertRaisesRegex(ValueError, "confirm in a terminal"):
                change_runpod_lease(run_dir, 3600, yes=False, input_stream=io.StringIO())
            self.assertTrue(all(command.startswith("date +%s") for command in commands))

    def test_a_pod_from_before_changeable_leases_is_refused(self) -> None:
        with _run() as run_dir, _pod(""), self.assertRaisesRegex(ValueError, "cannot be changed"):
            change_runpod_lease(run_dir, 3600, yes=True)

    def test_a_render_pod_is_refused_because_its_creation_timer_would_still_end_it(self) -> None:
        with _run() as run_dir:
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "purpose": "comfyui-render", "pod": {"id": "pod-1"}}), encoding="utf-8")
            with patch.object(runpod_ssh, "_runpod_ssh_details") as ssh, self.assertRaisesRegex(ValueError, "render Pod"):
                change_runpod_lease(run_dir, 3600, yes=True)
            ssh.assert_not_called()

    def test_a_stopped_pod_is_refused(self) -> None:
        with _run(pod_stopped_at="t") as run_dir, self.assertRaisesRegex(ValueError, "already stopped"):
            change_runpod_lease(run_dir, 3600, yes=True)


class LeaseWarningTests(unittest.TestCase):
    def test_training_that_would_outlast_the_lease_is_reported(self) -> None:
        with _run(last_step=100, total_steps=1100, seconds_per_iter=36.0) as run_dir:
            now = time.time()
            record_lease_deadline(run_dir, int(now + 4 * 3600), reason="armed")
            shortfall = lease_shortfall(run_dir, "r1", now=now)
            self.assertAlmostEqual(shortfall["training_left_sec"], 1000 * 36.0)
            record_lease_deadline(run_dir, int(now + 12 * 3600), reason="changed by kura run lease")
            self.assertIsNone(lease_shortfall(run_dir, "r1", now=now))

    def test_the_warning_is_given_once_per_deadline(self) -> None:
        with _run(last_step=1, total_steps=1000, seconds_per_iter=60.0) as run_dir:
            record_lease_deadline(run_dir, int(time.time()) + 3600, reason="armed")
            stderr = io.StringIO()
            with patch("sys.stderr", stderr), patch("kura.notifications.notify") as notify:
                runpod_ssh._warn_if_lease_short(run_dir, "r1", "example", ["ntfy"])
                runpod_ssh._warn_if_lease_short(run_dir, "r1", "example", ["ntfy"])
            self.assertEqual(stderr.getvalue().count("kura run lease example"), 1)
            notify.assert_called_once()



@unittest.skipIf(os.name == "nt", "the Pod runs the guard under Linux sh")
class LeaseGuardShellTests(unittest.TestCase):
    def _guard(self, directory: Path, lease_sec: int) -> str:
        deadline = str(directory / "deadline")
        deleted = directory / "deleted"
        guard = runpod_ssh._runpod_lease_guard_shell(max_lease_sec=lease_sec, pod_id="pod-1", log_path=str(directory / "stdout.log"))
        guard = guard.replace(runpod_ssh.LEASE_DEADLINE_PATH, deadline).replace("sleep 30", "sleep 0.1")
        # Stand in for the RunPod API call, after the real function is defined.
        return guard.replace("kura_lease_initial=", f"kura_pod_self_delete() {{ echo deleted > {deleted}; }}\nkura_lease_initial=", 1)

    def test_the_guard_deletes_the_pod_when_the_deadline_passes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            subprocess.run(["bash", "-c", self._guard(directory, 1)], check=True, timeout=10)
            deadline = time.monotonic() + 8
            while not (directory / "deleted").exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertTrue((directory / "deleted").exists())
            self.assertIn("the maximum lease ended", (directory / "stdout.log").read_text(encoding="utf-8"))

    def test_a_moved_deadline_keeps_the_pod(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            subprocess.run(["bash", "-c", self._guard(directory, 2)], check=True, timeout=10)
            (directory / "deadline").write_text(str(int(time.time()) + 3600), encoding="utf-8")
            time.sleep(4)
            self.assertFalse((directory / "deleted").exists())
            # A broken deadline file falls back to the deadline the guard was armed with.
            (directory / "deadline").write_text("garbage", encoding="utf-8")
            deadline = time.monotonic() + 5
            while not (directory / "deleted").exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertTrue((directory / "deleted").exists())

    def test_an_absurd_deadline_falls_back_to_the_armed_one(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            subprocess.run(["bash", "-c", self._guard(directory, 2)], check=True, timeout=10)
            (directory / "deadline").write_text("9" * 30, encoding="utf-8")
            deadline = time.monotonic() + 6
            while not (directory / "deleted").exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertTrue((directory / "deleted").exists())

    def test_a_second_guard_never_moves_an_existing_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "deadline").write_text("123", encoding="utf-8")
            guard = self._guard(directory, 3600).split("\n(\n", 1)[0]  # only the arming lines
            subprocess.run(["bash", "-c", guard], check=True, timeout=10)
            self.assertEqual((directory / "deadline").read_text(encoding="utf-8").strip(), "123")


class LeaseRecordTests(unittest.TestCase):
    def test_a_warning_never_hides_a_later_extension(self) -> None:
        with _run(last_step=1, total_steps=1000, seconds_per_iter=60.0) as run_dir:
            now = time.time()
            record_lease_deadline(run_dir, int(now) + 3600, reason="armed")
            with patch("sys.stderr", io.StringIO()), patch("kura.notifications.notify"):
                runpod_ssh._warn_if_lease_short(run_dir, "r1", "example", None)
            record_lease_deadline(run_dir, int(now) + 40 * 3600, reason="changed by kura run lease")
            self.assertEqual(runpod_ssh.latest_lease_deadline(run_dir, "r1"), int(now) + 40 * 3600)
            self.assertIsNone(lease_shortfall(run_dir, "r1", now=now))

    def test_an_absurd_lease_is_refused_before_touching_the_pod(self) -> None:
        with _run() as run_dir, patch.object(runpod_ssh, "_runpod_ssh_details") as ssh, self.assertRaisesRegex(ValueError, "refused"):
            change_runpod_lease(run_dir, 10**12, yes=True)
        ssh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
