"""Writes Kura does not measure (a trainer's own model downloads, a run-scoped disk cache) are
shown as not counted with their scale and never refused: locally the disk watch covers them
(docs/adr/local-disk-protection.md); on RunPod a full Pod disk fails the run and the lease bounds the bill."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.run_commands import plan
from kura.run_envelope import validate_train_run_fields


def _ai_toolkit_run(**safety: object) -> dict:
    run = {"backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl"}}, "model": {"base": "stabilityai/stable-diffusion-xl-base-1.0"}}
    if safety:
        run["safety"] = dict(safety)
    return run


class UnmeasuredModelDownloadTests(unittest.TestCase):
    def test_a_trainer_download_is_a_warning_with_its_scale_and_passes(self) -> None:
        cache = Path("/mnt/e/hf-cache")
        for executor in ("docker", "runpod"):
            with self.subTest(executor=executor):
                report = plan._model_download_preflight_report(_ai_toolkit_run(), {"bytes": 0}, plan._plan_requirements(_ai_toolkit_run(), {"bytes": 0}), executor=executor, local_cache=cache)
                self.assertEqual([item["severity"] for item in report], ["warning"])
                fact = report[0]["fact"]
                self.assertIn("stabilityai/stable-diffusion-xl-base-1.0", fact)
                self.assertIn("diffusers repositories in hub/, single files in ai-toolkit-models/", fact)
                self.assertIn("(often 10–30 GB)", fact)
                self.assertIn("not counted", fact)
                if executor == "docker":
                    self.assertIn(str(cache), fact)
                    self.assertIn("the disk watch stops the run", fact)
                else:
                    self.assertIn("each Pod starts with an empty cache", fact)
                    self.assertIn("runpod.container_disk_gb", fact)

    def test_a_trainer_download_adds_nothing_to_either_disk_check_and_does_not_refuse(self) -> None:
        run = _ai_toolkit_run(allow_storage_risk=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None):
                runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 1}, {"bytes": 0})
            with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                local = plan._local_launch_disk_preflight(
                    Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                    enforce_model_download_safety=False, download_estimate={"bytes": 0},
                )
        self.assertEqual(runpod["measured_write_bytes"], 0)
        self.assertEqual(local["paths"]["hf_cache"]["estimated_write_bytes"], 0)

    def test_a_run_whose_downloads_kura_measures_names_no_trainer_download(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}}}
        self.assertIsNone(plan._trainer_downloads(plan._plan_requirements(run, {}), executor="docker", local_cache=Path("/c")))

    def test_a_local_path_base_is_no_trainer_download(self) -> None:
        # The trainer reads a local model.base from disk; only a selector it resolves itself downloads.
        run = _ai_toolkit_run()
        run["model"]["base"] = "/models/sdxl"
        for executor in ("docker", "runpod"):
            with self.subTest(executor=executor):
                self.assertIsNone(plan._trainer_downloads(plan._plan_requirements(run, {}), executor=executor, local_cache=Path("/c")))
                report = plan._model_download_preflight_report(run, {"bytes": 0}, plan._plan_requirements(run, {"bytes": 0}), executor=executor, local_cache=Path("/c"))
                self.assertFalse(any("downloads" in item["fact"] and "itself" in item["fact"] for item in report))
                self.assertNotIn("warning", {item["severity"] for item in report})

    def test_a_preflight_reads_the_model_requirements_once(self) -> None:
        import kura.model_requirements as model_requirements

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(plan, "model_requirements", wraps=model_requirements.model_requirements) as owner:
            records = plan.collect_run_preflight(_ai_toolkit_run(), Path(directory), config={}, executor="docker", download_estimate={"bytes": 0})
        self.assertIn("model-acquisition", {item["check"] for item in records})
        owner.assert_called_once()

    def test_the_declared_size_is_retired(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety.model_download_estimate_gb is retired"):
            validate_train_run_fields({"safety": {"model_download_estimate_gb": 14}})


class UnmeasuredDiskCacheTests(unittest.TestCase):
    def test_a_disk_cache_is_information_with_its_scale(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}}}
        report = plan._disk_cache_preflight_report(run)
        self.assertEqual(report[0]["severity"], "info")
        self.assertIn("not counted in the disk estimate (image caches are a few MB per item", report[0]["fact"])


if __name__ == "__main__":
    unittest.main()
