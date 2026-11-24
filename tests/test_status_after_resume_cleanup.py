"""After `kura cleanup --resume-data`, a run's status stops listing the training states that went.

The reclaim record cleanup appends states the removal; `read_run_status`, which the monitor and
`kura run status` read through, lists what the records leave. Cleanup never writes status.json.
The same holds for a Docker run and for a RunPod run collected from its downloads snapshot."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from kura.cli import build_parser, cmd_run_status
from kura.executors import read_run_status
from kura.executors.common import _mutate_run_status
from kura.monitor import collect_run_summary
from tests.test_collected_state_identity import _artifacts, _collected_run
from tests.test_reclaim import _cleanup, _run, _state, _workspace, _write
from tests.test_run_end import _download
from tests.user_config import isolate_user_config


def _listed(run_dir: Path) -> list[str]:
    """The run's recoverable states as its viewers show them; `kura run status` prints the same list."""
    listed = [item["artifact_id"] for item in read_run_status(run_dir).get("recoverable_training_states", [])]
    out = io.StringIO()
    previous = Path.cwd()
    os.chdir(run_dir.parent.parent)
    try:
        with contextlib.redirect_stdout(out):
            code = cmd_run_status(build_parser().parse_args(["run", "status", run_dir.name]))
    finally:
        os.chdir(previous)
    assert code == 0
    printed = [item["artifact_id"] for item in json.loads(out.getvalue()).get("recoverable_training_states", [])]
    assert printed == listed, (printed, listed)
    return listed


def _summary(root: Path, run_id: str):
    return collect_run_summary(root, run_id)


class StatusAfterResumeCleanupTests(unittest.TestCase):
    def test_a_docker_run_lists_only_the_states_cleanup_left(self) -> None:
        with _workspace() as root:
            source = _run(root, "source", ended="2026-10-01T00:00:00+00:00")
            kept = _state(root, "state-step-00000010-aaaaaaaaaaaa", "source")
            freed = _state(root, "state-step-00000020-bbbbbbbbbbbb", "source")
            # A continuation references the earlier state, so cleanup keeps it.
            _run(root, "continued", state="compiled", continuation=kept.name)
            status = json.loads((source / "status.json").read_text(encoding="utf-8"))
            status["recoverable_training_states"] = [
                {"artifact_id": kept.name, "manifest_sha256": "a" * 64, "observed_step": 10, "restoration_level": "exact_resume"},
                {"artifact_id": freed.name, "manifest_sha256": "b" * 64, "observed_step": 20, "restoration_level": "exact_resume"},
            ]
            _write(source / "status.json", json.dumps(status))
            self.assertEqual(_listed(source), [kept.name, freed.name])
            self.assertEqual(_summary(root, "source").recoverable_state_step, 20)
            written = (source / "status.json").read_bytes()

            self.assertEqual(_cleanup(["--resume-data", "source", "--yes"])[0], 0)

            self.assertFalse(freed.exists())
            self.assertEqual(_listed(source), [kept.name])
            self.assertEqual(_listed(source), [item["id"] for item in _artifacts(root)])
            self.assertEqual(_summary(root, "source").recoverable_state_step, 10)
            # The reclaim record states the removal; status.json is not rewritten for it.
            self.assertEqual((source / "status.json").read_bytes(), written)

    def test_a_runpod_run_lists_none_after_cleanup_and_collecting_again_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir, snapshot_state = _collected_run(root)
            self.assertEqual(_download(root), 0)
            [artifact] = _artifacts(root)
            self.assertEqual(_listed(run_dir), [artifact["id"]])
            self.assertEqual(_summary(root, "source").recoverable_state_step, 200)
            # The collected run's Pod was deleted; cleanup offers nothing of a run whose Pod may still bill.
            _mutate_run_status(run_dir, lambda status: status.update({"pod_stopped_at": "2026-01-01T00:10:00+00:00"}))
            written = (run_dir / "status.json").read_bytes()

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, _, err = _cleanup(["--resume-data", "source", "--yes"])
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0, err)

            self.assertEqual(_artifacts(root), [])
            self.assertFalse(snapshot_state.exists())
            self.assertEqual(_listed(run_dir), [])
            self.assertIsNone(_summary(root, "source").recoverable_state_step)
            self.assertEqual((run_dir / "status.json").read_bytes(), written)

            # Collecting the run again from its snapshot (`materialize_downloaded_status`) finds no state,
            # as the user chose: the run stays completed and nothing comes back into the list.
            self.assertEqual(_download(root), 0)
            status = read_run_status(run_dir)
            self.assertEqual(status["state"], "completed")
            self.assertNotIn("training_state_sync_error", status)
            self.assertEqual(_listed(run_dir), [])
            self.assertEqual(_artifacts(root), [])
            self.assertIsNone(_summary(root, "source").recoverable_state_step)


def setUpModule() -> None:
    isolate_user_config()


if __name__ == "__main__":
    unittest.main()
