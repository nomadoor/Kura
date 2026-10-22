"""After any failed Pod create, the Pod is looked up by name before anything is created again.

The same input runs through both create loops (training `launch_runpod`, render
`launch_runpod_session`); both ask `_pod_made_by_failed_create`.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from kura.executors import runpod
from kura.executors.runpod import RunPodAPIError, unresolved_create_intents

# Two GPU candidates, so a loop that treats a failure as refused tries a second create.
CONFIG = {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40", "NVIDIA L40S"], "gpu_type_priority": "custom", "cloud_types": ["COMMUNITY"]}
# A GraphQL `errors` reply whose message says nothing about whether the Pod was made.
UNKNOWN = RunPodAPIError("RunPod GraphQL failed: Something went wrong. Please try again later.", status_code=400)
CAPACITY = RunPodAPIError("RunPod GraphQL failed: There are no longer any instances available with the requested specifications", status_code=400)


def _training_run(root: Path) -> Path:
    run_dir = root / "runs" / "train"
    (run_dir / "transfer").mkdir(parents=True)
    (run_dir / "transfer" / "stage.json").write_text(json.dumps(
        {"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled", "last_stage": "transfer/stage.json"}), encoding="utf-8")
    return run_dir


def _render_run(root: Path) -> Path:
    run_dir = root / "runs" / "render"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
    return run_dir


def _launch_training(run_dir: Path) -> str | None:
    return runpod.launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                                image="registry/image:tag", config=CONFIG, yes=True)


def _launch_render(run_dir: Path) -> str | None:
    return runpod.launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=CONFIG, purpose="comfyui-render",
                                        dry_run=False, yes=True, max_lease_sec=3600)


LOOPS = (("training", _training_run, _launch_training), ("render", _render_run, _launch_render))


class FailedCreateTests(unittest.TestCase):
    def _launch(self, make_run, launch, *, creates, listing):
        """Run one launch; returns (run_dir, create mock, listing mock, the launch's result or error)."""
        stack = ExitStack()
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.addCleanup(stack.close)
        run_dir = make_run(root)
        stack.enter_context(patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False))
        stack.enter_context(patch("kura.executors.runpod.time.sleep"))
        stack.enter_context(patch("kura.executors.runpod.runpod_gpu_availability",
                                  return_value={"status": "unavailable", "reason": "not measured", "candidates": []}))
        request = stack.enter_context(patch("kura.executors.runpod._runpod_request", side_effect=creates))
        listed = stack.enter_context(patch("kura.executors.runpod._runpod_pods_named", side_effect=listing))
        try:
            outcome: object = launch(run_dir)
        except ValueError as exc:
            outcome = exc
        return run_dir, request, listed, outcome

    def test_a_failed_create_whose_pod_is_found_by_name_adopts_it_and_creates_no_second_pod(self) -> None:
        for label, make_run, launch in LOOPS:
            with self.subTest(label):
                run_dir, request, _listed, realization_id = self._launch(
                    make_run, launch, creates=[UNKNOWN],
                    listing=lambda _key, name, **_k: [{"id": "pod-1", "name": name, "desiredStatus": "RUNNING"}])
                self.assertIsInstance(realization_id, str)
                self.assertEqual(request.call_count, 1)
                realization = json.loads((run_dir / "realizations" / f"{realization_id}.json").read_text(encoding="utf-8"))
                self.assertEqual(realization["pod"]["id"], "pod-1")
                self.assertEqual(realization["request"]["gpuTypeIds"], ["NVIDIA A40"])
                self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_a_failed_create_with_no_pod_by_that_name_tries_the_next_candidate(self) -> None:
        for label, make_run, launch in LOOPS:
            # Whatever the reply: a GraphQL error, a server error, or a lost connection.
            for failure in (UNKNOWN, RunPodAPIError("RunPod GraphQL failed (503): down", status_code=503), ValueError("RunPod API is unreachable: timed out")):
                with self.subTest(label, failure=str(failure)):
                    run_dir, request, listed, realization_id = self._launch(
                        make_run, launch, creates=[failure, {"id": "pod-2", "name": "created", "desiredStatus": "RUNNING"}],
                        listing=lambda *_a, **_k: [])
                    self.assertIsInstance(realization_id, str)
                    self.assertEqual(request.call_count, 2)
                    self.assertEqual(request.call_args_list[1].args[3]["gpuTypeIds"], ["NVIDIA L40S"])
                    # Looked up under the name the create used.
                    self.assertGreaterEqual(listed.call_count, 1)
                    self.assertEqual({call.args[1] for call in listed.call_args_list}, {request.call_args_list[0].args[3]["name"]})
                realization = json.loads((run_dir / "realizations" / f"{realization_id}.json").read_text(encoding="utf-8"))
                self.assertEqual(realization["pod"]["id"], "pod-2")

    def test_a_failed_create_whose_lookup_fails_hands_the_run_over_without_creating_again(self) -> None:
        for label, make_run, launch in LOOPS:
            for listing in (ValueError("RunPod API is unreachable: down"),
                            # More than one Pod by the name: which one is this launch's is not Kura's to guess.
                            lambda _key, name, **_k: [{"id": "pod-1", "name": name}, {"id": "pod-2", "name": name}]):
                with self.subTest(label, listing=listing):
                    run_dir, request, _listed, outcome = self._launch(make_run, launch, creates=[UNKNOWN], listing=listing)
                    self.assertIsInstance(outcome, ValueError)
                    self.assertIn("kura run reconcile", str(outcome))
                    self.assertEqual(request.call_count, 1)
                    self.assertEqual(len(unresolved_create_intents(run_dir)), 1)
                    self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")

    def test_a_capacity_refusal_tries_the_next_candidate_without_a_lookup(self) -> None:
        for label, make_run, launch in LOOPS:
            with self.subTest(label):
                _run_dir, request, listed, realization_id = self._launch(
                    make_run, launch, creates=[CAPACITY, {"id": "pod-2", "name": "created", "desiredStatus": "RUNNING"}],
                    listing=AssertionError("a capacity refusal needs no lookup"))
                self.assertIsInstance(realization_id, str)
                self.assertEqual(request.call_count, 2)
                listed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
