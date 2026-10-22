"""A RunPod run's GPU cost: one rule (`kura.executors.runpod.pod_gpu_cost`) computed from recorded
facts (the price RunPod gave when it created the Pod, the launch, and the recorded stop), read by
`kura run status`, the execute completion lines, and the monitor alike. Nothing new is recorded."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kura.executors.runpod as runpod
import kura.monitor as monitor
from kura.cli import cmd_run_status
from kura.run_commands.experiment import format_run_completion

# The 0.7.0 acceptance run on an RTX A5000: Pod launched 08:02:21.74, deleted 08:09:14.51, $0.27/h.
TAIL = ("at $0.27/h (approximate: the GPU's hourly price times the time from the Pod's launch; storage and network "
        "are not included; earlier launches of this run are not counted)")
COST = f"about $0.03 for 6m 52s, {TAIL}"


def _runpod_run(root: Path, pod: dict | None = None, **status_changes) -> Path:
    run_dir = root / "runs" / "example"
    (run_dir / "realizations").mkdir(parents=True)
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    (run_dir / "run.yaml").write_text("id: example\ntype: train\nintent: check the cost\n", encoding="utf-8")
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({
        "id": "r1", "executor": "runpod", "launched_at": "2026-10-19T08:02:21.737876+09:00",
        "pod": pod if pod is not None else {"id": "pod1", "desired_status": "RUNNING", "last_started_at": None, "cost_per_h": 0.27},
    }), encoding="utf-8")
    status = {
        "state": "completed", "exit_code": 0, "host": "runpod", "pod_id": "pod1",
        "started": "2026-10-19T08:02:21.737876+09:00", "ended": "2026-10-19T08:07:54.686666+09:00",
        "last_realization": "realizations/r1.json", "pod_stopped_at": "2026-10-19T08:09:14.510840+09:00", "outputs": [],
        **status_changes,
    }
    (run_dir / "status.json").write_text(json.dumps({key: value for key, value in status.items() if value is not None}), encoding="utf-8")
    return run_dir


class PodCostTests(unittest.TestCase):
    def _status_summary(self, root: Path) -> dict:
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
        finally:
            os.chdir(previous)
        return json.loads(stdout.getvalue())["summary"]

    def test_status_completion_and_monitor_say_the_same_cost_from_one_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir = _runpod_run(root)
            # An observed container start after the launch does not shorten the billed time.
            (run_dir / "realizations" / "r1.observed-1.json").write_text(json.dumps({"last_started_at": "2026-10-19T08:05:00+09:00"}), encoding="utf-8")
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            owner = runpod.pod_gpu_cost
            with patch.object(runpod, "pod_gpu_cost", wraps=owner) as through_runpod, \
                    patch.object(monitor, "pod_gpu_cost", wraps=owner) as through_monitor:
                summary = self._status_summary(root)
                completion = format_run_completion(root, run_dir, status)
                pod = monitor.collect_run_summary(root, "example").executor_info.pod
            self.assertEqual((through_runpod.call_count, through_monitor.call_count), (2, 1))
        self.assertEqual(summary["pod_gpu_cost"], COST)
        self.assertIn(f"pod cost   {COST}", completion)
        self.assertAlmostEqual(pod.cost_used, 0.27 * (6 * 60 + 52.772964) / 3600, places=6)

    def test_a_cost_is_final_only_when_the_pod_stop_is_recorded(self) -> None:
        cases = (
            # The run ended but its Pod's stop is not recorded (a delete that failed or has not run yet).
            ({"pod_stopped_at": None}, " so far; the Pod may still bill, "),
            ({"pod_stopped_at": None, "state": "recovery_required"}, " so far; the Pod may still bill, "),
            ({"pod_stopped_at": None, "pod_missing_at": "2026-10-19T20:07:54+09:00"},
             "at most $3.26 for at most 12h 05m, until the Pod was found gone; its stop was not recorded, "),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                _runpod_run(root, **changes)
                self.assertIn(expected, self._status_summary(root)["pod_gpu_cost"])

    def test_a_relaunch_waiting_for_its_pod_does_not_bill_the_deleted_one(self) -> None:
        # The second launch's create intent cleared pod_id; status still names the first realization.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _runpod_run(root, state="launching", pod_id=None, pod_stopped_at=None, ended=None)
            self.assertNotIn("pod_gpu_cost", self._status_summary(root))

    def test_a_missing_fact_is_named_and_a_run_without_a_pod_says_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _runpod_run(root, pod={"id": "pod1"})
            self.assertEqual(self._status_summary(root)["pod_gpu_cost"], "unknown: the Pod's hourly price was not recorded")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir = _runpod_run(root, pod={}, pod_id=None, host="docker", pod_stopped_at=None)
            self.assertNotIn("pod_gpu_cost", self._status_summary(root))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertNotIn("pod cost", format_run_completion(root, run_dir, status))


if __name__ == "__main__":
    unittest.main()
