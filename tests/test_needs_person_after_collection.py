"""A RunPod run recorded as needing a person after collection keeps that state, loses its Pod, and is retried until the Pod is gone."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura import runner
from kura.run_commands.runpod_ssh import DOWNLOAD_NEEDS_PERSON

NEEDS_PERSON = {"state": "recovery_required", "exit_code": 0, "execution_state": "completed", "publication_state": "blocked",
                "recovery_required": True, "publication_error": "required training-state artifact is not published",
                "downloaded_run": "downloads/example", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}


class NeedsPersonTests(unittest.TestCase):
    def test_deleting_the_pod_keeps_the_recorded_state(self) -> None:
        from kura.executors import runpod

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps(NEEDS_PERSON), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), patch.object(runpod, "_runpod_request", return_value={}):
                status = runpod.stop_runpod(run_dir, {"gpu_type_ids": ["A"]})
            self.assertEqual((status["state"], status["exit_code"]), ("recovery_required", 0))
            self.assertTrue(status["pod_stopped_at"])

    def test_a_collected_run_needing_a_person_still_gets_its_pod_deleted_by_the_runner(self) -> None:
        # The controller's own stop failed; the Pod holds nothing the snapshot lacks, so a follower deletes it.
        self.assertTrue(runner._pod_left_running(NEEDS_PERSON))
        held = {**NEEDS_PERSON, "downloaded_run": None}  # collection never finished: a person decides about the Pod
        self.assertFalse(runner._pod_left_running(held))

    def test_the_controller_deletes_the_pod_and_says_the_run_needs_attention(self) -> None:
        from tests.test_cli import _run_remote_in_process

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                def collected(*_args, **_kwargs):
                    (run_dir / "status.json").write_text(json.dumps(NEEDS_PERSON), encoding="utf-8")
                    return DOWNLOAD_NEEDS_PERSON

                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", side_effect=collected), \
                     patch("kura.run_commands.launch._notify") as notify, \
                     patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop:
                    code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, max_lease="3h", yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            stop.assert_called_once()
            self.assertIn("needs attention", notify.call_args.kwargs["subject"])


if __name__ == "__main__":
    unittest.main()
