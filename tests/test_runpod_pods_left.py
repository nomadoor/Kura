"""Which of a run's recorded Pods may still bill: one answer for deleting, following, listing, and pruning."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura import runner
from kura.executors import runpod


@contextmanager
def _run(status: dict, *, duplicates: list[str] | None = None):
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "workspace.yaml").write_text("schema_version: 2\nrunpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
        run_dir = Path(directory) / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        realization = {"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}, **({"duplicate_pod_ids": duplicates} if duplicates else {})}
        (run_dir / "realizations" / "r1.json").write_text(json.dumps(realization), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"last_realization": "realizations/r1.json", "pod_id": "pod-1", **status}), encoding="utf-8")
        yield run_dir


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


class PodsLeftTests(unittest.TestCase):
    def test_a_duplicate_outlives_a_missing_pod(self) -> None:
        # The recorded Pod deleted itself; the duplicate a recovered launch found still runs.
        with _run({"state": "interrupted", "pod_missing_at": "t"}, duplicates=["pod-2"]) as run_dir:
            self.assertEqual(runpod.pods_left(run_dir, _status(run_dir)), ["pod-2"])
            self.assertIsNotNone(runner.pod_may_still_bill(run_dir, _status(run_dir)))
            self.assertTrue(runner._pod_left_running(run_dir, _status(run_dir)))

    def test_a_pod_an_older_kura_held_for_a_person_is_deleted_like_any_other(self) -> None:
        # Kura no longer keeps a Pod for a person, and reads no old held form: its Pod still bills, so it goes.
        held = {"state": "recovery_required", "downloaded_run": None}
        with _run(held) as run_dir:
            self.assertIsNotNone(runner.pod_may_still_bill(run_dir, _status(run_dir)))
            self.assertTrue(runner._pod_left_running(run_dir, _status(run_dir)))

    def test_a_stop_deletes_the_duplicate_of_a_missing_pod(self) -> None:
        with _run({"state": "interrupted", "pod_missing_at": "t"}, duplicates=["pod-2"]) as run_dir:
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), patch.object(runpod, "_runpod_request", return_value={}) as request:
                self.assertTrue(runner._delete_pod(run_dir, why="test"))
            self.assertIn(("DELETE", "/pods/pod-2"), [call.args[:2] for call in request.call_args_list])
            self.assertEqual(runpod.pods_left(run_dir, _status(run_dir)), [])

    def test_the_monitor_says_a_pod_is_up_only_while_one_may_bill(self) -> None:
        from kura.monitor import _executor_info
        from kura.tui import _remote_execution_phase
        from types import SimpleNamespace

        def shown(run_dir: Path) -> tuple[bool, str]:
            status = _status(run_dir)
            realization = json.loads((run_dir / "realizations" / "r1.json").read_text(encoding="utf-8"))
            info = _executor_info("runpod", {}, status, realization, run_dir)
            summary = SimpleNamespace(executor_info=info, capacity_wait=None, is_stale=False, state=status.get("state"),
                                      progress=SimpleNamespace(step=None, total=None))
            return info.pod_stopped, _remote_execution_phase(summary)

        # RunPod reported the Pod gone: it is not provisioning, and it bills no more.
        with _run({"state": "interrupted", "pod_missing_at": "t"}) as run_dir:
            stopped, phase = shown(run_dir)
            self.assertTrue(stopped)
            self.assertNotIn("provisioning", phase)
        # A duplicate outlives it: a Pod is still up.
        with _run({"state": "interrupted", "pod_missing_at": "t"}, duplicates=["pod-2"]) as run_dir:
            self.assertFalse(shown(run_dir)[0])

    def test_a_new_launch_waits_while_a_recovered_duplicate_is_left(self) -> None:
        with _run({"state": "interrupted", "pod_missing_at": "t"}, duplicates=["pod-2"]) as run_dir:
            realization = json.loads((run_dir / "realizations" / "r1.json").read_text(encoding="utf-8"))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({**realization, "recovered_from_intent": "r1.create-intent.json"}), encoding="utf-8")
            self.assertEqual(runpod.unstopped_recovered_pod(run_dir), "pod-2")

    def test_nothing_is_left_once_a_stop_deleted_them_all(self) -> None:
        with _run({"state": "interrupted", "pod_stopped_at": "t"}, duplicates=["pod-2"]) as run_dir:
            self.assertEqual(runpod.pods_left(run_dir, _status(run_dir)), [])
            self.assertIsNone(runner.pod_may_still_bill(run_dir, _status(run_dir)))


if __name__ == "__main__":
    unittest.main()
