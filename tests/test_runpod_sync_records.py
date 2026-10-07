"""RunPod sync facts are recorded before status shows them (run-records ADR, decision 5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kura.executors.common import _mutate_run_status
from kura.run_commands import runpod_ssh


def _run(directory: str, **status) -> Path:
    run_dir = Path(directory) / "runs" / "example"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json", **status}), encoding="utf-8")
    return run_dir


class RemoteLogCursorTests(unittest.TestCase):
    def test_a_run_synced_before_the_cursor_file_continues_from_its_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(directory, remote_log_bytes=120)
            self.assertEqual(runpod_ssh._remote_log_offset(run_dir), 120)
            (run_dir / runpod_ssh.REMOTE_LOG_CURSOR).write_text(json.dumps({"remote_bytes": 300}), encoding="utf-8")
            self.assertEqual(runpod_ssh._remote_log_offset(run_dir), 300)

    def test_a_damaged_cursor_starts_over_rather_than_failing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(directory)
            (run_dir / runpod_ssh.REMOTE_LOG_CURSOR).write_text("{", encoding="utf-8")
            self.assertEqual(runpod_ssh._remote_log_offset(run_dir), 0)


class SyncErrorRecordTests(unittest.TestCase):
    def test_only_a_change_of_sync_error_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(directory)
            for error in ("scp failed", "scp failed", None, None):
                _mutate_run_status(run_dir, lambda status, error=error: runpod_ssh._record_sync_error(run_dir, status, "checkpoint_sync_error", error))
            kinds = sorted(json.loads(path.read_text(encoding="utf-8"))["kind"] for path in (run_dir / "realizations").glob("r1.sync-*.json"))
            self.assertEqual(kinds, ["sync_error", "sync_recovered"])
            self.assertNotIn("checkpoint_sync_error", json.loads((run_dir / "status.json").read_text(encoding="utf-8")))


class MirroredOutputRecordTests(unittest.TestCase):
    def test_a_copied_checkpoint_is_recorded_and_a_skipped_one_is_not_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(directory)
            copied = {"name": "a.safetensors", "path": "outputs/a.safetensors", "step": 100, "size": 1, "remote_path": "/r/a", "remote_mtime_ns": 5, "skipped": False}
            runpod_ssh._record_pulled_outputs(run_dir, [copied], emit_event=False)
            runpod_ssh._record_pulled_outputs(run_dir, [{**copied, "skipped": True}], emit_event=False)
            lines = runpod_ssh.mirrored_outputs_path(run_dir, "r1").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual((json.loads(lines[0])["kind"], json.loads(lines[0])["remote_mtime_ns"]), ("output_mirrored", 5))
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["mirrored_outputs"][0]["step"], 100)


if __name__ == "__main__":
    unittest.main()
