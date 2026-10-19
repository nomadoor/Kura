"""One rule for a run-scoped disk cache on every backend: the adapter says whether the trainer
writes one, and the plan shows it, not counted (docs/adr/local-disk-protection.md: the disk watch covers it)."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.backends.registry import validate_backend_config
from kura.run_commands import plan
from kura.run_envelope import validate_train_run_fields, without_retired_train_run_keys


def _runs_that_write_a_cache() -> dict[str, dict]:
    return {
        "sd-scripts": {"backend": {"name": "sd-scripts", "config": {"architecture": "sd15", "cache_latents_to_disk": True}}},
        "musubi-tuner": {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan"}}},
        "ai-toolkit authored": {"backend": {"name": "ai-toolkit", "config": {
            "model_arch": "sd1", "native_config": {"train": {"cache_text_embeddings": True}},
        }}},
        "ai-toolkit baseline": {"backend": {"name": "ai-toolkit", "config": {"model_arch": "ltx2.3"}}},
    }


class DiskCacheTests(unittest.TestCase):
    def test_every_backend_that_writes_a_cache_says_it_is_not_counted(self) -> None:
        for case, run in _runs_that_write_a_cache().items():
            with self.subTest(case=case):
                report = plan._disk_cache_preflight_report(run)
                self.assertEqual(report[0]["severity"], "info")
                self.assertIn("not counted in the disk estimate", report[0]["fact"])
                self.assertIn("image caches are a few MB per item", report[0]["fact"])

    def test_a_run_that_writes_no_cache_reports_none(self) -> None:
        for run in (
            {"backend": {"name": "sd-scripts", "config": {"architecture": "sd15"}}},
            {"backend": {"name": "ai-toolkit", "config": {"model_arch": "sd1"}}},
            {"backend": {"name": "musubi-tuner", "config": {"command": ["true"]}}},
            {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan", "precache": False}}},
            # An explicit command is the author's own; Kura cannot tell what it caches.
            {"backend": {"name": "sd-scripts", "config": {"command": ["true"], "cache_latents_to_disk": True}}},
        ):
            with self.subTest(backend=run["backend"]["name"]):
                self.assertEqual(plan._disk_cache_preflight_report(run), [])

    def test_ai_toolkit_reads_the_authored_text_embedding_setting_over_its_baseline(self) -> None:
        from kura.backends.ai_toolkit import ai_toolkit_disk_cache

        # minimax_h3's baseline caches latents and text embeddings; authoring the embeddings off leaves the latents.
        run = {"backend": {"name": "ai-toolkit", "config": {"model_arch": "minimax_h3", "native_config": {"train": {"cache_text_embeddings": False}}}}}
        self.assertEqual(ai_toolkit_disk_cache(run), "AI-Toolkit writes an on-disk cache (cache_latents_to_disk)")

    def test_neither_disk_check_counts_a_cache(self) -> None:
        run = _runs_that_write_a_cache()["musubi-tuner"]
        run["safety"] = {"allow_storage_risk": True}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None):
                runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 50}, {"bytes": 0})
            with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                local = plan._local_launch_disk_preflight(
                    Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                    download_estimate={"bytes": 0},
                )
        self.assertEqual(runpod["estimated_write_bytes"], 0)
        self.assertEqual(local["paths"]["workspace"]["estimated_write_bytes"], 0)

    def test_the_declared_size_is_retired_on_every_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety.disk_cache_estimate_gb is retired"):
            validate_train_run_fields({"safety": {"disk_cache_estimate_gb": 1.5}})
        retired = {"backend": {"name": "sd-scripts", "config": {"architecture": "sd15", "disk_cache_estimate_gb": 1}}}
        with self.assertRaisesRegex(ValueError, "disk watch"):
            validate_backend_config(retired)
        copied = without_retired_train_run_keys({**retired, "safety": {"disk_cache_estimate_gb": 4}})
        self.assertNotIn("disk_cache_estimate_gb", copied["backend"]["config"])
        self.assertEqual(copied["safety"], {})

    def test_a_compiled_run_holding_a_retired_backend_key_is_still_read(self) -> None:
        for name, config in (
            ("musubi-tuner", {"architecture": "wan", "allow_a40_uncheckpointed_9b": True}),
            ("sd-scripts", {"architecture": "sd15", "disk_cache_estimate_gb": 1}),
        ):
            with self.subTest(backend=name):
                run = {"backend": {"name": name, "config": config}}
                validate_backend_config(run, frozen=True)
                with self.assertRaises(ValueError):
                    validate_backend_config(run)


if __name__ == "__main__":
    unittest.main()
