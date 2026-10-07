"""Status projected from records alone (run-records ADR, decision 5), in shadow mode."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.executors.common import _mutate_run_status
from kura.status_projection import project_status, shadow_differences


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.directory.name) / "runs" / "example"
        (self.run_dir / "logs").mkdir(parents=True)
        self.realizations = self.run_dir / "realizations"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_a_run_without_launches_is_draft_or_compiled(self) -> None:
        self.assertEqual(project_status(self.run_dir), {"state": "draft"})
        _write(self.run_dir / "resolved" / "manifest.lock.yaml", {})
        self.assertEqual(project_status(self.run_dir), {"state": "compiled"})

    def test_an_intent_without_a_realization_is_launching_and_unconfirmed_is_interrupted(self) -> None:
        _write(self.realizations / "20261008-000000-000001.create-intent.json", {"kind": "pod_create_intent"})
        self.assertEqual(project_status(self.run_dir)["state"], "launching")
        _write(self.realizations / "20261008-000000-000001.create-unconfirmed.json", {"at": "t1"})
        self.assertEqual(project_status(self.run_dir), {"state": "interrupted", "ended": "t1", "exit_code": None})

    def test_a_runpod_pod_stopped_mid_run_is_interrupted(self) -> None:
        rid = "20261008-000000-000001"
        _write(self.realizations / f"{rid}.json", {"executor": "runpod", "state": "running", "launched_at": "t0", "pod": {"id": "pod-1"}})
        _write(self.realizations / f"{rid}.stop-1.json", {"executor": "runpod", "outcome": "stopped", "stopped_at": "t2", "requested_at": "t1"})
        projected = project_status(self.run_dir)
        self.assertEqual((projected["state"], projected["ended"], projected["pod_stopped_at"], projected["pod_id"]), ("interrupted", "t2", "t2", "pod-1"))

    def test_a_downloaded_runpod_run_ends_with_the_pods_exit_record(self) -> None:
        rid = "20261008-000000-000001"
        _write(self.realizations / f"{rid}.json", {"executor": "runpod", "state": "running", "launched_at": "t0", "pod": {"id": "pod-1"}})
        _write(self.run_dir / "downloads" / "example" / "realizations" / "remote-exit-1.json", {"exit_code": 0, "timestamp": "t3"})
        self.assertEqual({key: project_status(self.run_dir)[key] for key in ("state", "exit_code", "ended")}, {"state": "completed", "exit_code": 0, "ended": "t3"})

    def test_a_docker_run_is_completed_only_once_published(self) -> None:
        rid = "20261008-000000-000001"
        _write(self.realizations / f"{rid}.json", {"executor": "docker", "state": "running", "launched_at": "t0", "container": {"id": "c1", "name": "kura-x"}})
        _write(self.realizations / f"{rid}.observed-1.json", {"state": "completed", "exit_code": 0, "ended": "t2", "observed_at": "t2"})
        with patch("kura.artifact_publication.output_contract", return_value={"outputs": []}):
            self.assertEqual(project_status(self.run_dir)["state"], "publishing")
            _write(self.realizations / f"{rid}.publication.json", {"kind": "publication"})
            self.assertEqual(project_status(self.run_dir)["state"], "completed")

    def test_a_follower_ending_a_run_is_projected_from_its_record(self) -> None:
        rid = "20261008-000000-000001"
        _write(self.realizations / f"{rid}.json", {"executor": "runpod", "state": "running", "launched_at": "t0", "pod": {"id": "pod-1"}})
        _write(self.realizations / f"{rid}.ended-1.json", {"state": "interrupted", "at": "t5"})
        self.assertEqual(project_status(self.run_dir)["state"], "interrupted")

    def test_a_local_render_runs_from_its_start_event_until_its_realization(self) -> None:
        (self.run_dir / "logs" / "events.jsonl").write_text(json.dumps({"event": "render_started", "timestamp": "t0"}) + "\n", encoding="utf-8")
        self.assertEqual(project_status(self.run_dir)["state"], "running")
        _write(self.realizations / "20261008-000000-000001.json", {"generator": "comfyui", "state": "completed", "timestamp": "t9"})
        self.assertEqual({key: project_status(self.run_dir)[key] for key in ("state", "ended", "exit_code")}, {"state": "completed", "ended": "t9", "exit_code": 0})


class ShadowTests(unittest.TestCase):
    def test_a_status_write_its_records_do_not_support_is_logged_and_still_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            env = {key: value for key, value in os.environ.items() if key != "KURA_STATUS_SHADOW"}
            with patch.dict(os.environ, env, clear=True):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "running"}))
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")
            logged = json.loads((run_dir / "logs" / "status-shadow.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(logged["differences"]["state"], {"written": "running", "projected": "draft"})
            self.assertEqual(shadow_differences(run_dir, {"state": "draft"}), {})

    def test_shadow_mode_can_be_turned_off_and_never_breaks_a_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            with patch.dict(os.environ, {"KURA_STATUS_SHADOW": "0"}):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "running"}))
            self.assertFalse((run_dir / "logs" / "status-shadow.jsonl").exists())
            with patch("kura.status_projection.shadow_differences", side_effect=RuntimeError("broken")):
                _mutate_run_status(run_dir, lambda status: status.update({"state": "failed"}))
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "failed")


if __name__ == "__main__":
    unittest.main()
