"""The plan says how a trainer-resolved model is downloaded, as it differs by executor."""

from __future__ import annotations

import unittest
from pathlib import Path

from kura.paths import HF_CACHE_CONTAINER_PATH
from kura.run_commands.plan import _model_download_preflight_report, _plan_requirements


class BackendResolvedModelTests(unittest.TestCase):
    def test_runpod_downloads_on_every_run_and_local_reuses_the_cache(self) -> None:
        run = {"type": "train", "backend": {"name": "ai-toolkit"}, "model": {"base": "stabilityai/sdxl"}}
        cache = Path("/mnt/e/hf-cache")
        local = _model_download_preflight_report(run, {}, _plan_requirements(run, {}), executor="docker", local_cache=cache)[0]["fact"]
        remote = _model_download_preflight_report(run, {}, _plan_requirements(run, {}), executor="runpod", local_cache=cache)[0]["fact"]
        self.assertIn("stabilityai/sdxl", local)
        self.assertIn("reusing files an earlier run left there", local)
        self.assertIn("on every run, while the Pod bills", remote)

    def test_the_acquisition_line_names_where_the_trainer_downloads(self) -> None:
        run = {"type": "train", "backend": {"name": "ai-toolkit"}, "model": {"base": "org/model"}}
        cache = Path("/mnt/e/hf-cache")
        local = _model_download_preflight_report(run, {}, _plan_requirements(run, {}), executor="docker", local_cache=cache)[0]["fact"]
        remote = _model_download_preflight_report(run, {}, _plan_requirements(run, {}), executor="runpod", local_cache=cache)[0]["fact"]
        self.assertIn(f"into the Hugging Face cache at {cache} (diffusers repositories in hub/, single files in ai-toolkit-models/)", local)
        self.assertIn(f"into the Pod's Hugging Face cache at {HF_CACHE_CONTAINER_PATH} (diffusers repositories in hub/", remote)
        for fact in (local, remote):
            self.assertIn("estimate (often 10–30 GB)", fact)
            self.assertIn("not counted", fact)


class UnnamedModelTests(unittest.TestCase):
    def test_a_run_that_names_no_model_files_yet_says_its_downloads_are_unknown(self) -> None:
        # A Musubi draft with no model_bundle, model_downloads, or model_paths: nothing is sized yet.
        run = {"type": "train", "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2", "model_version": "klein-base-4b"}}}
        for executor in ("docker", "runpod"):
            with self.subTest(executor=executor):
                records = _model_download_preflight_report(run, {"bytes": 0, "items": []}, _plan_requirements(run, {"bytes": 0}), executor=executor, local_cache=Path("/c"))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["severity"], "info")
                self.assertEqual(records[0]["fact"], "model downloads: unknown until the run names its model files "
                                                     "(compile names a missing one); nothing is counted for them yet")
                self.assertNotIn("0 B", records[0]["fact"])
        # A run that names only local files downloads nothing, which the plan still says as 0 B.
        local = {"type": "train", "backend": {"name": "musubi-tuner", "config": {"architecture": "zimage", "model_paths": {"dit": "/m/d.safetensors"}}}}
        fact = _model_download_preflight_report(local, {"bytes": 0, "items": []}, _plan_requirements(local, {"bytes": 0}), executor="docker", local_cache=Path("/c"))[0]["fact"]
        self.assertEqual(fact, "estimated model downloads write 0 B")


if __name__ == "__main__":
    unittest.main()
