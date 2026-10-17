"""Writes Kura cannot measure (a trainer's own model downloads, a run-scoped disk cache) are
shown as not counted with their scale, and counted in both disk checks once the author declares
their size; the free-space floor covers them until then, so none is refused."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.run_commands import plan
from kura.run_envelope import validate_train_run_fields


def _ai_toolkit_run(**safety: float) -> dict:
    run = {"backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl"}}, "model": {"base": "stabilityai/stable-diffusion-xl-base-1.0"}}
    if safety:
        run["safety"] = dict(safety)
    return run


class UnmeasuredModelDownloadTests(unittest.TestCase):
    def test_an_undeclared_trainer_download_is_a_warning_with_its_scale_and_passes(self) -> None:
        cache = Path("/mnt/e/hf-cache")
        for executor in ("docker", "runpod"):
            with self.subTest(executor=executor):
                report = plan._model_download_preflight_report(_ai_toolkit_run(), {"bytes": 0}, executor=executor, local_cache=cache)
                self.assertEqual([item["severity"] for item in report], ["warning"])
                fact = report[0]["fact"]
                self.assertIn("stabilityai/stable-diffusion-xl-base-1.0", fact)
                self.assertIn("diffusers repositories in hub/, single files in ai-toolkit-models/", fact)
                self.assertIn("not counted in the disk estimate (often 10–30 GB)", fact)
                self.assertIn("set safety.model_download_estimate_gb to count it", fact)
                self.assertIn(str(cache) if executor == "docker" else "each Pod starts with an empty cache", fact)

    def test_a_declared_trainer_download_is_information(self) -> None:
        report = plan._model_download_preflight_report(_ai_toolkit_run(model_download_estimate_gb=14), {"bytes": 0}, executor="docker", local_cache=Path("/c"))
        self.assertEqual(report[0]["severity"], "info")
        self.assertIn("14.0 GiB (safety.model_download_estimate_gb)", report[0]["fact"])

    def test_local_and_runpod_disk_checks_count_the_declared_download_through_the_one_helper(self) -> None:
        run = _ai_toolkit_run(model_download_estimate_gb=12, allow_storage_risk=True)
        declared = 12 * 1024**3
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                    patch.object(plan, "_estimate_checkpoint_write_bytes", return_value={"bytes": 0}), \
                    patch.object(plan, "_disk_cache_estimate", return_value={"bytes": 0}), \
                    patch.object(plan, "_unmeasured_write", wraps=plan._unmeasured_write) as runpod_owner:
                runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 50}, {"bytes": 0})
            with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                    patch.object(plan, "_estimate_checkpoint_write_bytes", return_value={"bytes": 0}), \
                    patch.object(plan, "_disk_cache_estimate", return_value={"bytes": 0}), \
                    patch.object(plan, "_unmeasured_write", wraps=plan._unmeasured_write) as local_owner:
                local = plan._local_launch_disk_preflight(
                    Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                    enforce_model_download_safety=False, download_estimate={"bytes": 0},
                )
        runpod_owner.assert_called()
        local_owner.assert_called()
        self.assertEqual(runpod["estimated_write_bytes"], declared)
        self.assertEqual(runpod["estimates"]["trainer_downloads"]["status"], "declared-estimate")
        self.assertEqual(local["paths"]["hf_cache"]["estimated_write_bytes"], declared)
        self.assertEqual(local["paths"]["workspace"]["estimated_write_bytes"], 0)

    def test_an_undeclared_trainer_download_adds_nothing_and_does_not_refuse(self) -> None:
        with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                patch.object(plan, "_estimate_checkpoint_write_bytes", return_value={"bytes": 0}):
            runpod = plan._runpod_launch_disk_preflight(_ai_toolkit_run(), {"container_disk_gb": 1}, {"bytes": 0})
        self.assertEqual(runpod["estimated_write_bytes"], 0)
        self.assertEqual(runpod["estimates"]["trainer_downloads"]["status"], "unknown")

    def test_a_run_whose_downloads_kura_measures_ignores_no_declared_size_it_could_count(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}}, "safety": {"model_download_estimate_gb": 5}}
        self.assertEqual(plan._trainer_download_estimate(run, executor="docker", local_cache=Path("/c"))["bytes"], 0)

    def test_the_size_is_a_safety_key(self) -> None:
        validate_train_run_fields({"safety": {"model_download_estimate_gb": 14}})
        with self.assertRaisesRegex(ValueError, "safety.model_download_estimate_gb must be a positive number"):
            validate_train_run_fields({"safety": {"model_download_estimate_gb": "14"}})


class UnmeasuredDiskCacheTests(unittest.TestCase):
    def test_an_undeclared_disk_cache_is_information_with_its_scale(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}}}
        with patch.object(plan, "_unmeasured_write", wraps=plan._unmeasured_write) as owner:
            report = plan._disk_cache_preflight_report(run)
        owner.assert_called()
        self.assertEqual(report[0]["severity"], "info")
        self.assertIn("not counted in the disk estimate (image caches are a few MB per item)", report[0]["fact"])
        self.assertIn("set safety.disk_cache_estimate_gb for large or video datasets", report[0]["fact"])


if __name__ == "__main__":
    unittest.main()
