"""How many steps a run trains is decided once, and every checkpoint and disk estimate reads it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

import kura.monitor as monitor
import kura.run_commands.plan as plan
from kura.run_envelope import common_recipe
from kura.training_artifacts import trained_steps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_resume_steps import as_resume, musubi_run  # noqa: E402


def _resume(*, save_every: int, additional: int) -> dict[str, Any]:
    run = musubi_run()
    run["backend"]["config"]["save_every_n_steps"] = save_every
    return as_resume(run, source_step=1000, additional=additional)


def _monitor_expected(run: dict[str, Any], save_every: int) -> int | None:
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "backend-display.lock.json").write_text(
            json.dumps({"checkpoint": {"save_every_n_steps": save_every}}), encoding="utf-8",
        )
        return monitor._checkpoint_expected(run, run_dir)


def _estimates(run: dict[str, Any], save_every: int) -> dict[str, Any]:
    """The run through all five places that count its checkpoints."""
    allowed = {**run, "safety": {"allow_many_checkpoints": True}}
    try:
        plan._checkpoint_safety_preflight(run)
        refused = None
    except ValueError as exc:
        refused = str(exc)
    return {
        "disk_warnings": plan._disk_warnings(run, {"save_every_n_steps": save_every}),
        "safety_refusal": refused,
        "preflight": [record["fact"] for record in plan._checkpoint_preflight_report(allowed)],
        "write_count": plan._estimate_checkpoint_write_bytes(allowed)["count"],
        "monitor_expected": _monitor_expected(run, save_every),
    }


class TrainedStepsOwnerTests(unittest.TestCase):
    def test_a_run_trains_the_resume_added_steps_else_the_recipe_steps(self) -> None:
        from kura.training_artifacts import trained_steps

        self.assertEqual(trained_steps(musubi_run()), 1000)
        self.assertEqual(trained_steps(as_resume(musubi_run(), additional=50)), 50)
        self.assertEqual(trained_steps(as_resume(musubi_run(), additional=5000)), 5000)
        no_steps = musubi_run()
        del no_steps["recipe"]["steps"]
        self.assertIsNone(trained_steps(no_steps))
        broken = as_resume(musubi_run())
        broken["continuation"]["target_step"] = 1
        with self.assertRaises(ValueError):
            trained_steps(broken)


class ResumeEstimateTests(unittest.TestCase):
    def test_a_short_resume_is_estimated_from_the_steps_it_adds(self) -> None:
        # 1000-step recipe resumed +50 with a cadence of 10: 5 checkpoints, not 100.
        estimates = _estimates(_resume(save_every=10, additional=50), 10)
        self.assertEqual(estimates["disk_warnings"], [])
        self.assertIsNone(estimates["safety_refusal"])
        self.assertEqual(estimates["preflight"], ["checkpoint cadence implies about 5 checkpoint(s)"])
        self.assertEqual(estimates["write_count"], 5)
        self.assertEqual(estimates["monitor_expected"], 5)

    def test_a_resume_longer_than_its_recipe_trips_the_checkpoint_guard(self) -> None:
        # 1000-step recipe resumed +5000 with a cadence of 500: 10 checkpoints, not 2.
        estimates = _estimates(_resume(save_every=500, additional=5000), 500)
        self.assertEqual(len(estimates["disk_warnings"]), 1)
        self.assertIn("about 10 checkpoints", estimates["disk_warnings"][0])
        self.assertIn("about 10 checkpoints", estimates["safety_refusal"] or "")
        self.assertEqual(estimates["preflight"], ["checkpoint cadence implies about 10 checkpoint(s)"])
        self.assertEqual(estimates["write_count"], 10)
        self.assertEqual(estimates["monitor_expected"], 10)

    def test_the_monitor_shows_no_expectation_for_an_invalid_continuation(self) -> None:
        broken = _resume(save_every=10, additional=50)
        broken["continuation"]["target_step"] = 1
        self.assertIsNone(_monitor_expected(broken, 10))

    def test_every_estimate_reads_the_one_owner(self) -> None:
        from kura.training_artifacts import trained_steps

        run = _resume(save_every=10, additional=50)
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        cases = {
            "_disk_warnings": (plan, lambda: plan._disk_warnings(run, {"save_every_n_steps": 10})),
            "_checkpoint_safety_preflight": (plan, lambda: plan._checkpoint_safety_preflight(run)),
            "_estimate_checkpoint_write_bytes": (plan, lambda: plan._estimate_checkpoint_write_bytes(allowed)),
            "_checkpoint_expected": (monitor, lambda: _monitor_expected(run, 10)),
        }
        for name, (module, call) in cases.items():
            with self.subTest(place=name), patch.object(module, "trained_steps", wraps=trained_steps) as owner:
                call()
                owner.assert_called()
        # The preflight report counts once for its guard and once for its own line.
        with patch.object(plan, "_checkpoint_safety_preflight"), patch.object(plan, "trained_steps", wraps=trained_steps) as owner:
            plan._checkpoint_preflight_report(run)
            owner.assert_called_once()


class LaunchDiskPreflightTests(unittest.TestCase):
    def test_local_and_runpod_disk_preflights_count_the_resume_added_steps(self) -> None:
        # 1000-step source resumed +50 with a cadence of 10: 5 checkpoints of 1 GiB on both executors.
        run = _resume(save_every=10, additional=50)
        run["safety"] = {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 1, "allow_storage_risk": True}
        expected = {"bytes": 5 * 1024**3, "count": 5, "per_checkpoint_gib": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(plan, "trained_steps", wraps=trained_steps) as owner:
            with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None):
                runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 50}, {"bytes": 0})
            with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                local = plan._local_launch_disk_preflight(
                    Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                    enforce_model_download_safety=False, download_estimate={"bytes": 0},
                )
            self.assertEqual(owner.call_count, 2)
        self.assertEqual(runpod["estimates"]["checkpoints"], expected)
        self.assertEqual(local["estimates"]["checkpoints"], expected)
        cache = int(runpod["estimates"]["disk_cache"].get("bytes") or 0)
        self.assertEqual(runpod["estimated_write_bytes"], expected["bytes"] + cache)
        self.assertEqual(local["paths"]["workspace"]["estimated_write_bytes"], expected["bytes"] + cache)

    def test_a_long_resume_warns_about_the_samples_it_adds(self) -> None:
        # 1000-step recipe with a sample cadence of 100: 10 batches fresh, about 50 when resumed +5000.
        fresh = musubi_run()
        fresh["sampling"] = {"cadence_steps": 100}
        self.assertEqual(plan._disk_warnings(fresh, {}), [])
        warnings = plan._disk_warnings(as_resume(fresh, source_step=1000, additional=5000), {})
        self.assertEqual(warnings, ["sampling cadence may create about 50 sample batches"])


class FreshRunPlanTests(unittest.TestCase):
    def test_a_fresh_run_plans_byte_identically_to_the_recipe_steps(self) -> None:
        def old_rule(run: dict[str, Any]) -> int | None:
            return plan._as_positive_int(common_recipe(run).get("steps"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (root / "datasets" / "tiny").mkdir(parents=True)
            run_dir = root / "runs" / "fresh"
            run_dir.mkdir(parents=True)
            run = {
                "id": "fresh",
                "type": "train",
                "model": {"base": "repo/model"},
                "datasets": [{"id": "tiny"}],
                "recipe": {"steps": 1000, "seed": 1},
                "compute": {"executor": "runpod"},
                "safety": {"allow_many_checkpoints": True},
                "sampling": {"cadence_steps": 10},
                "backend": {"name": "musubi-tuner", "config": {
                    "architecture": "flux2", "model_bundle": "none", "save_every_n_steps": 50,
                    "model_downloads": {"dit": {"repo": "repo/model", "filename": "weights.safetensors"}},
                }},
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 200}):
                    current = plan.format_run_plan(plan.plan_run("fresh"))
                    with patch.object(plan, "trained_steps", side_effect=old_rule):
                        before = plan.format_run_plan(plan.plan_run("fresh"))
            finally:
                os.chdir(previous)
        self.assertEqual(current, before)
        self.assertIn("about 20 checkpoint(s)", current)


if __name__ == "__main__":
    unittest.main()
