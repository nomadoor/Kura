"""A failed Pod create counts as refused only for known refusals; anything else may have created a Pod and is looked up by name."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.executors import runpod
from kura.executors.runpod import RunPodAPIError, _create_outcome_uncertain


def _graphql_reply(errors: list[dict]):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"errors": errors, "data": None}).encode("utf-8")

    return Response()


def _graphql_error(errors: list[dict]) -> RunPodAPIError:
    with patch("kura.executors.runpod.urlopen", return_value=_graphql_reply(errors)):
        try:
            runpod._runpod_graphql("mutation {}", {}, "api-secret")
        except RunPodAPIError as exc:
            return exc
    raise AssertionError("no error raised")


class CreateRefusalTests(unittest.TestCase):
    def test_an_unknown_graphql_error_on_a_create_is_uncertain(self) -> None:
        self.assertTrue(_create_outcome_uncertain(_graphql_error([{"message": "Something went wrong. Please try again later."}])))

    def test_known_refusals_are_not_uncertain(self) -> None:
        capacity = _graphql_error([{"message": "There are no longer any instances available with the requested specifications"}])
        validation = _graphql_error([{"message": 'Variable "$input" got invalid value', "extensions": {"code": "GRAPHQL_VALIDATION_FAILED"}}])
        for exc in (capacity, validation, RunPodAPIError("RunPod GraphQL failed (401): unauthorized", status_code=401),
                    RunPodAPIError("RunPod GraphQL failed (429): slow down", status_code=429)):
            with self.subTest(str(exc)):
                self.assertFalse(_create_outcome_uncertain(exc))

    def test_server_errors_and_lost_replies_stay_uncertain(self) -> None:
        for exc in (RunPodAPIError("RunPod GraphQL failed (502): bad gateway", status_code=502),
                    # A server error page that happens to read like a capacity refusal is still a server error.
                    RunPodAPIError("RunPod GraphQL failed (503): requested GPU unavailable", status_code=503),
                    # One rejected error beside one that is not says nothing about the rest.
                    _graphql_error([{"message": "bad", "extensions": {"code": "BAD_USER_INPUT"}}, {"message": "Something went wrong"}]),
                    ValueError("RunPod API is unreachable: timed out")):
            with self.subTest(str(exc)):
                self.assertTrue(_create_outcome_uncertain(exc))

    def test_a_create_answered_with_an_unknown_graphql_error_finds_its_pod_instead_of_creating_again(self) -> None:
        unknown = _graphql_error([{"message": "Something went wrong. Please try again later."}])
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "render"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            config = {"gpu_type_ids": ["NVIDIA A40", "NVIDIA L40S"], "gpu_type_priority": "custom", "cloud_types": ["COMMUNITY"]}
            pod = {"id": "pod-1", "name": "found", "desiredStatus": "RUNNING"}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.executors.runpod._runpod_request", side_effect=unknown) as request, \
                    patch("kura.executors.runpod._runpod_pods_named", side_effect=lambda _key, name, **_k: [{**pod, "name": name}]), \
                    patch("kura.executors.runpod.time.sleep"), \
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "not measured", "candidates": []}):
                runpod.launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=config, purpose="comfyui-render",
                                             dry_run=False, yes=True, max_lease_sec=3600)
            # One create only: the second GPU candidate is never tried while the first may have made a Pod.
            self.assertEqual(request.call_count, 1)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["pod_id"], "pod-1")


if __name__ == "__main__":
    unittest.main()
