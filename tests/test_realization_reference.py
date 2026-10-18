"""How status names a run's realization, and how a launch status is settled from its records."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from kura.records import names_realization, realization_reference

RID = "20261004-101010-000001"


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


class RealizationReferenceTests(unittest.TestCase):
    def test_a_reference_is_posix_and_names_its_realization_in_either_spelling(self) -> None:
        self.assertEqual(realization_reference(RID), f"realizations/{RID}.json")
        self.assertTrue(names_realization(f"realizations/{RID}.json", RID))
        # A status an earlier Kura wrote on Windows spells the reference with a backslash.
        self.assertTrue(names_realization(f"realizations\\{RID}.json", RID))
        self.assertFalse(names_realization(f"realizations/{RID}.json", "20261004-101010-000002"))
        self.assertFalse(names_realization(None, RID))

    def test_a_publication_recorded_under_a_windows_status_is_reused(self) -> None:
        from kura.artifact_publication import existing_publication

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "realizations").mkdir()
            manifest = f"realizations/{RID}.publication.json"
            (run_dir / manifest).write_text(json.dumps({"files": [{"path": "outputs/a.safetensors"}]}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({
                "last_realization": f"realizations\\{RID}.json", "publication_state": "completed", "publication_manifest": manifest,
            }), encoding="utf-8")
            self.assertEqual(existing_publication(run_dir, RID), (manifest, ["outputs/a.safetensors"]))

    def test_a_missing_container_observed_under_a_windows_status_is_its_own(self) -> None:
        from kura.reclaim import last_observation_container_missing

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "realizations").mkdir()
            observation = f"realizations/{RID}.observation-1.json"
            (run_dir / observation).write_text(json.dumps({"realization_id": RID, "container_missing": True}), encoding="utf-8")
            status = {"last_realization": f"realizations\\{RID}.json", "last_observation": observation}
            self.assertTrue(last_observation_container_missing(run_dir, status))


def _docker_draft(run_dir: Path) -> dict:
    return {"id": RID, "executor": "docker", "host": platform.node(),
            "container": {"id": None, "name": f"kura-{run_dir.name}-{RID}", "labels": {}}}


class SettleFromRealizationTests(unittest.TestCase):
    """Settling a launch status did not follow writes what the launch itself would have written."""

    STALE = {"state": "launching", "host": None, "last_step": 40, "total_steps": 100, "seconds_per_iter": 1.5,
             "current_run_step": 40, "current_run_total_steps": 100, "last_observation": "realizations/old.observation.json",
             "capacity_wait": {"attempts": 2}}

    def _run(self, root: Path, name: str) -> Path:
        run_dir = root / name
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps(self.STALE), encoding="utf-8")
        return run_dir

    def test_docker_settling_equals_the_launch_writer(self) -> None:
        from kura.executors import docker
        from kura.executors.common import settle_status_from_realization

        with tempfile.TemporaryDirectory() as directory:
            launched, settled = self._run(Path(directory), "a"), self._run(Path(directory), "b")
            with patch.object(docker, "append_run_event"):
                docker._record_container_started(launched, _docker_draft(launched), "c-1", launched_at="2026-10-04T10:10:11+09:00")
            realization = json.loads((launched / "realizations" / f"{RID}.json").read_text(encoding="utf-8"))
            (settled / "realizations" / f"{RID}.json").write_text(json.dumps(realization), encoding="utf-8")
            settle_status_from_realization(settled, settled / "realizations" / f"{RID}.json")
            expected = {key: value for key, value in _status(launched).items()}
            self.assertEqual(_status(settled), expected)
            self.assertEqual(expected["host"], platform.node())
            self.assertEqual(expected["last_realization"], f"realizations/{RID}.json")
            for key in ("last_step", "total_steps", "seconds_per_iter", "current_run_step", "current_run_total_steps", "last_observation", "capacity_wait"):
                self.assertNotIn(key, expected)

    def test_runpod_settling_sets_the_host_and_clears_the_previous_progress(self) -> None:
        from kura.executors.common import settle_status_from_realization

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory), "a")
            realization = {"id": RID, "executor": "runpod", "state": "running", "launched_at": "2026-10-04T10:10:11+09:00",
                           "pod": {"id": "pod-1"}, "ai_toolkit_models": {"state": "not_recorded"}}
            (run_dir / "realizations" / f"{RID}.json").write_text(json.dumps(realization), encoding="utf-8")
            settle_status_from_realization(run_dir, run_dir / "realizations" / f"{RID}.json")
            status = _status(run_dir)
            for key in ("kind", "schema_version", "runner_epoch"):
                status.pop(key, None)
            self.assertEqual(status, {
                "state": "running", "started": "2026-10-04T10:10:11+09:00", "ended": None, "exit_code": None, "host": "runpod",
                "last_realization": f"realizations/{RID}.json", "pod_id": "pod-1", "ai_toolkit_models": {"state": "not_recorded"},
            })


class ReconcileWithoutStatusTests(unittest.TestCase):
    def test_reconcile_refuses_a_run_whose_status_is_missing_before_looking_for_anything(self) -> None:
        from kura.cli import cmd_run_reconcile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\nrunpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "realizations" / f"{RID}.create-intent.json").write_text(
                json.dumps({"realization_id": RID, "executor": "runpod", "pod_name": f"kura-example-{RID}"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                stderr = io.StringIO()
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                        patch("kura.executors.runpod._runpod_pods_named") as listed, redirect_stderr(stderr):
                    code = cmd_run_reconcile(argparse.Namespace(run_id="example"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            listed.assert_not_called()
            message = stderr.getvalue()
            self.assertIn("status.json", message)
            self.assertIn("kura run reconcile example", message)
            self.assertNotIn("Errno", message)
            self.assertFalse((run_dir / "status.json").exists())
            self.assertEqual(sorted(path.name for path in (run_dir / "realizations").iterdir()), [f"{RID}.create-intent.json"])


if __name__ == "__main__":
    unittest.main()
