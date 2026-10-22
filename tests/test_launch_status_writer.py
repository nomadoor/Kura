"""Status follows a launch's realization through one writer, under the status lock, on every
executor and for every launch outcome: a Pod or container started, a failed launch, a cancelled
capacity wait, and a create intent settled after a crash."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.status_projection import shadow_differences
from tests.test_runpod_create_intent import CONFIG, CAPACITY_REFUSED, _crashed_launch, _status

SRC = Path(__file__).resolve().parents[1] / "src" / "kura" / "executors"
# Facts an earlier realization left in status; none of them is the new launch's.
STALE = {"capacity_wait": {"started_at": "2026-10-01T00:00:00+09:00"}, "last_observation": "realizations/old.observed-1.json",
         "last_step": 40, "total_steps": 100, "ai_toolkit_models": [], "container_id": "c-old", "container_name": "old", "pod_id": "pod-old",
         # How the earlier launch's Pod stopped; kept, it would make the new Pod look stopped to billing checks.
         "pod_stopped_at": "2026-10-01T01:00:00+09:00", "pod_missing_at": "2026-10-01T01:00:00+09:00", "stop_reason": "user"}


def _session_run(root: Path) -> Path:
    run_dir = root / "runs" / "render"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled", **STALE}), encoding="utf-8")
    return run_dir


class OneLaunchWriterTests(unittest.TestCase):
    def test_no_executor_writes_status_outside_the_status_lock(self) -> None:
        for name in ("runpod.py", "docker.py"):
            with self.subTest(executor=name):
                self.assertIsNone(re.search(r"\b_write_status\(", (SRC / name).read_text(encoding="utf-8")))

    def test_a_session_launch_points_status_at_its_pod_and_drops_the_earlier_launchs_facts(self) -> None:
        from kura.executors.runpod import launch_runpod_session

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _session_run(Path(directory))
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-9", "desiredStatus": "RUNNING"}), \
                    patch("kura.executors.runpod.time.sleep"):
                launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render", yes=True, max_lease_sec=3600)
            status = _status(run_dir)
            self.assertEqual(status["pod_id"], "pod-9")
            for key in ("capacity_wait", "last_observation", "last_step", "total_steps", "ai_toolkit_models", "container_id", "container_name",
                        "pod_stopped_at", "pod_missing_at", "stop_reason"):
                self.assertNotIn(key, status)
            from kura import runner

            self.assertTrue(runner._pod_left_running(run_dir, status))
            self.assertEqual(shadow_differences(run_dir, status), {})

    def test_a_failed_session_launch_says_what_its_realization_says(self) -> None:
        from kura.executors.runpod import RunPodAPIError, launch_runpod_session

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _session_run(Path(directory))
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError(CAPACITY_REFUSED, status_code=400)), \
                    patch("kura.executors.runpod.time.sleep"):
                with self.assertRaises(ValueError):
                    launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render", yes=True, max_lease_sec=3600)
            status = _status(run_dir)
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual((status["state"], status["ended"]), ("launch_failed", realization["attempted_at"]))
            for key in ("capacity_wait", "pod_id", "container_id", "last_step", "pod_stopped_at", "stop_reason"):
                self.assertNotIn(key, status)
            self.assertEqual(shadow_differences(run_dir, status), {})

    def test_a_settled_create_intent_ends_when_its_realization_says(self) -> None:
        from kura.executors.runpod import resolve_runpod_create_intents

        for pods in ([], [{"id": "pod-a", "name": "example", "runtime": {"uptimeInSeconds": 90}}]):
            with self.subTest(found=bool(pods)), _crashed_launch(previous_pod="pod-old") as run_dir:
                with patch("kura.executors.runpod._runpod_pods_named", return_value=pods):
                    resolve_runpod_create_intents(run_dir)
                status = _status(run_dir)
                self.assertEqual(shadow_differences(run_dir, status), {})
                self.assertEqual(status.get("pod_id"), "pod-a" if pods else None)


if __name__ == "__main__":
    unittest.main()
