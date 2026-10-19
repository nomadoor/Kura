"""A training state is published once, with the run's identity, whichever path brings it home.

On RunPod a state may be pulled while the Pod trains and then arrive again in the final
snapshot; on Docker it is on local disk. Every path publishes it in the local run's context
(its realization, executor, and image), so one save is one artifact."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.training_artifacts import publish_completed_training_states, publish_training_state_candidate, select_training_state
from tests.test_resume_steps import sd_scripts_run
from tests.test_run_end import _download, _runpod_run
import tests.test_state_save_parity as save_parity

IMAGE = {"reference": "example/kura-sd-scripts@sha256:" + "1" * 64, "pinning": {"strength": "content-hash"}}


def _collected_run(root: Path) -> tuple[Path, Path]:
    run_dir = _runpod_run(root, exit_code=0)
    run = sd_scripts_run()
    run["id"] = "source"
    (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({"training_state_managed": True}), encoding="utf-8")
    (run_dir / "realizations" / "launch.json").write_text(json.dumps({"id": "launch", "executor": "runpod", "image_identity": IMAGE}), encoding="utf-8")
    # A snapshot carries the Pod's copy of resolved/, but no status.json or launch record.
    shutil.copytree(run_dir / "resolved", run_dir / "downloads" / "source" / "resolved")
    snapshot_state = save_parity.StatePublicationParityTests._write_state(None, run_dir / "downloads" / "source" / "outputs" / "source-step0200-state", 200)
    return run_dir, snapshot_state


def _artifacts(root: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "artifacts" / "training-state").glob("*/manifest.json"))]


class CollectedStateIdentityTests(unittest.TestCase):
    def test_a_state_pulled_mid_run_is_not_published_again_by_the_final_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, snapshot_state = _collected_run(root)
            pulled = root / "pulled"
            shutil.copytree(snapshot_state, pulled)
            mid_run = publish_training_state_candidate(root, run_dir, pulled, 200)
            self.assertIsNotNone(mid_run)
            _download(root)
            artifacts = _artifacts(root)
            self.assertEqual([item["id"] for item in artifacts], [mid_run["id"]])
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual([item["artifact_id"] for item in status["recoverable_training_states"]], [mid_run["id"]])

    def test_a_state_only_in_the_final_snapshot_carries_the_runs_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, _ = _collected_run(root)
            _download(root)
            [artifact] = _artifacts(root)
            self.assertEqual(artifact["source_realization"], "realizations/launch.json")
            self.assertEqual(artifact["runtime_identity"]["actual_executor"], "runpod")
            self.assertEqual(artifact["runtime_identity"]["actual_image_identity"], IMAGE)

    def test_docker_and_runpod_publish_one_save_under_one_id(self) -> None:
        # The same state, published from the run's own outputs (Docker) and from a snapshot
        # directory (RunPod), is the same artifact.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, snapshot_state = _collected_run(root)
            shutil.copytree(snapshot_state, run_dir / "outputs" / snapshot_state.name)
            local = publish_completed_training_states(root, run_dir)
            collected = publish_completed_training_states(root, run_dir, outputs_root=snapshot_state.parent)
            self.assertEqual([item["id"] for item in local], [item["id"] for item in collected])
            self.assertEqual(len(_artifacts(root)), 1)

    def test_resume_selection_prefers_the_copy_with_the_runs_identity_over_an_older_identityless_one(self) -> None:
        # Kura before this fix published RunPod snapshot states a second time without identity;
        # those records stay, and selection reads past them whichever id sorts higher.
        from kura.training_artifacts import publish_training_state, training_state_at_step

        orders = set()
        for variant in range(16):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir, snapshot_state = _collected_run(root)
                (snapshot_state / "extra.txt").write_text(str(variant), encoding="utf-8")
                with_identity = publish_training_state_candidate(root, run_dir, snapshot_state, 200)
                identityless = publish_training_state(
                    root, source_run="source", source_realization=None, backend="sd-scripts", observed_step=200,
                    candidate=snapshot_state, native_format=with_identity["format"],
                    restoration_contract=with_identity["restoration_contract"],
                    runtime_identity={}, compatibility=with_identity["compatibility"],
                )
                orders.add(identityless["id"] > with_identity["id"])
                self.assertEqual(select_training_state(root, "source")["id"], with_identity["id"])
                self.assertEqual(training_state_at_step(root, "source", 200)["id"], with_identity["id"])
        self.assertEqual(orders, {True, False})

if __name__ == "__main__":
    unittest.main()
