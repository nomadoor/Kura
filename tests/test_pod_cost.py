"""A RunPod run's GPU cost: one rule (`kura.monitor._pod_gpu_cost`) computed from recorded facts (the
price RunPod gave for the Pod, its start, and its recorded stop), read by `kura run status`, the
execute completion lines, and the monitor alike. Nothing new is recorded."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kura.monitor as monitor
from kura.cli import cmd_run_status
from kura.run_commands.experiment import format_run_completion

# The 0.7.0 acceptance run on an RTX A5000: Pod created 08:02:21, deleted 08:09:14, $0.27/h.
COST = "about $0.03 for 6m 52s at $0.27/h (approximate: the GPU's hourly price times the Pod's recorded time; storage and network are not included)"


def _runpod_run(root: Path, **status_changes) -> Path:
    run_dir = root / "runs" / "example"
    (run_dir / "realizations").mkdir(parents=True)
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    (run_dir / "run.yaml").write_text("id: example\ntype: train\nintent: check the cost\n", encoding="utf-8")
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({
        "id": "r1", "executor": "runpod", "launched_at": "2026-10-19T08:02:21.737876+09:00",
        "pod": {"id": "pod1", "desired_status": "RUNNING", "last_started_at": None, "cost_per_h": 0.27},
    }), encoding="utf-8")
    status = {
        "state": "completed", "exit_code": 0, "host": "runpod", "pod_id": "pod1",
        "started": "2026-10-19T08:02:21.737876+09:00", "ended": "2026-10-19T08:07:54.686666+09:00",
        "last_realization": "realizations/r1.json", "pod_stopped_at": "2026-10-19T08:09:14.510840+09:00", "outputs": [],
        **status_changes,
    }
    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
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
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            with patch.object(monitor, "_pod_gpu_cost", wraps=monitor._pod_gpu_cost) as owner:
                summary = self._status_summary(root)
                completion = format_run_completion(root, run_dir, status)
                pod = monitor.collect_run_summary(root, "example").executor_info.pod
            self.assertEqual(owner.call_count, 3)
        self.assertEqual(summary["pod_gpu_cost"], COST)
        self.assertIn(f"pod cost   {COST}", completion)
        self.assertAlmostEqual(pod.cost_used, 0.27 * (6 * 60 + 52.772964) / 3600, places=6)

    def test_an_unfinished_run_says_so_far_and_a_run_without_a_pod_says_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _runpod_run(root, state="running", ended=None, pod_stopped_at=None)
            self.assertIn(" so far at $0.27/h", self._status_summary(root)["pod_gpu_cost"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir = _runpod_run(root, pod_id=None, host="docker")
            self.assertNotIn("pod_gpu_cost", self._status_summary(root))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertNotIn("pod cost", format_run_completion(root, run_dir, status))


if __name__ == "__main__":
    unittest.main()
