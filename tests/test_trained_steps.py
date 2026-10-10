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
import kura.training_artifacts as training_artifacts
from kura.run_envelope import common_recipe
from kura.training_artifacts import trained_steps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_resume_steps import ai_toolkit_run, as_resume, musubi_run, sd_scripts_run  # noqa: E402


def _resume(*, save_every: int, additional: int) -> dict[str, Any]:
    run = musubi_run()
    run["backend"]["config"]["save_every_n_steps"] = save_every
    return as_resume(run, source_step=1000, additional=additional)


def _monitor_expected(run: dict[str, Any], save_every: int | dict[str, Any]) -> int | None:
    checkpoint = save_every if isinstance(save_every, dict) else {"save_every_n_steps": save_every}
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "backend-display.lock.json").write_text(
            json.dumps({"checkpoint": checkpoint}), encoding="utf-8",
        )
        return monitor._checkpoint_expected(run, run_dir)


def _estimates(run: dict[str, Any], save_every: int) -> dict[str, Any]:
    """The run through all five places that count its checkpoints."""
    allowed = {**run, "safety": {"allow_many_checkpoints": True}}
    refusals = [record["fact"] for record in plan._checkpoint_preflight_report(run) if record["severity"] == "error"]
    refused = refusals[0] if refusals else None
    return {
        "disk_warnings": plan._disk_warnings(run, {"save_every_n_steps": save_every}),
        "safety_refusal": refused,
        "preflight": [record["fact"] for record in plan._checkpoint_preflight_report(allowed)],
        "write_count": plan._estimate_checkpoint_write_bytes(allowed)["count"],
        "monitor_expected": _monitor_expected(run, save_every),
    }


class TrainedStepsOwnerTests(unittest.TestCase):
    def test_a_run_trains_the_resume_added_steps_else_the_recipe_steps(self) -> None:
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
        self.assertEqual(estimates["preflight"], ["checkpoint cadence implies about 5 checkpoint(s) over the 50 steps this run trains"])
        self.assertEqual(estimates["write_count"], 5)
        self.assertEqual(estimates["monitor_expected"], 5)

    def test_a_resume_longer_than_its_recipe_trips_the_checkpoint_guard(self) -> None:
        # 1000-step recipe resumed +5000 with a cadence of 500: 10 checkpoints, not 2.
        estimates = _estimates(_resume(save_every=500, additional=5000), 500)
        self.assertEqual(len(estimates["disk_warnings"]), 1)
        self.assertIn("about 10 checkpoints over the 5000 steps this run trains;", estimates["disk_warnings"][0])
        self.assertIn("about 10 checkpoints without pruning over the 5000 steps this run trains", estimates["safety_refusal"] or "")
        self.assertEqual(estimates["preflight"], ["checkpoint cadence implies about 10 checkpoint(s) over the 5000 steps this run trains"])
        self.assertEqual(estimates["write_count"], 10)
        self.assertEqual(estimates["monitor_expected"], 10)

    def test_the_monitor_shows_no_expectation_for_an_invalid_continuation(self) -> None:
        broken = _resume(save_every=10, additional=50)
        broken["continuation"]["target_step"] = 1
        self.assertIsNone(_monitor_expected(broken, 10))

    def test_the_plan_fails_on_an_invalid_continuation_as_every_estimate_does(self) -> None:
        broken = _resume(save_every=10, additional=50)
        broken["continuation"]["target_step"] = 1
        for allowed in (False, True):
            run = {**broken, "safety": {"allow_many_checkpoints": allowed}}
            with self.subTest(allow_many_checkpoints=allowed), self.assertRaisesRegex(ValueError, "continuation.target_step does not match"):
                plan._checkpoint_preflight_report(run)

    def test_every_estimate_reads_the_one_owner(self) -> None:
        run = _resume(save_every=10, additional=50)
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        cases = {
            "_disk_warnings": (plan, lambda: plan._disk_warnings(run, {"save_every_n_steps": 10})),
            "_estimate_checkpoint_write_bytes": (plan, lambda: plan._estimate_checkpoint_write_bytes(allowed)),
            "_checkpoint_expected": (monitor, lambda: _monitor_expected(run, 10)),
        }
        for name, (module, call) in cases.items():
            with self.subTest(place=name), patch.object(module, "trained_steps", wraps=trained_steps) as owner:
                call()
                owner.assert_called()
        # The preflight report reads the owner once, for its guard and its own line.
        with patch.object(plan, "trained_steps", wraps=trained_steps) as owner:
            plan._checkpoint_preflight_report(run)
            owner.assert_called_once()


class LaunchDiskPreflightTests(unittest.TestCase):
    def test_local_and_runpod_disk_preflights_count_the_resume_added_steps(self) -> None:
        # 1000-step source resumed +50 with a cadence of 10: 5 checkpoints of 1 GiB on both executors.
        run = _resume(save_every=10, additional=50)
        run["safety"] = {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 1, "allow_storage_risk": True}
        expected = {"bytes": 5 * 1024**3, "count": 5, "per_checkpoint_gib": 1}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                    patch.object(plan, "trained_steps", wraps=trained_steps) as runpod_owner:
                runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 50}, {"bytes": 0})
            with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                    patch.object(plan, "trained_steps", wraps=trained_steps) as local_owner:
                local = plan._local_launch_disk_preflight(
                    Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                    enforce_model_download_safety=False, download_estimate={"bytes": 0},
                )
        runpod_owner.assert_called_once_with(run)
        local_owner.assert_called_once_with(run)
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
        self.assertEqual(warnings, ["sampling cadence may create about 50 sample batches over the 5000 steps this run trains"])


class ExpectedCheckpointsOwnerTests(unittest.TestCase):
    """How many checkpoints a run leaves, and whether a retention policy makes that unknown, is decided once."""

    def test_only_a_positive_retention_value_prunes(self) -> None:
        owner = training_artifacts.expected_checkpoints
        self.assertEqual(owner({"save_every_n_steps": 50}, 1000), 20)
        self.assertEqual(owner({"save_every_n_steps": 5000}, 1000), 1)
        self.assertIsNone(owner({"save_every_n_steps": 50}, None))
        self.assertIsNone(owner({}, 1000))
        for key in ("prune_before_step", "keep_last", "retention_window_steps"):
            with self.subTest(retention=key):
                self.assertIsNone(owner({"save_every_n_steps": 50, key: 3}, 1000))
                self.assertIsNone(owner({"save_every_n_steps": 50, key: "3"}, 1000))
                # AI-Toolkit keeps every save at max_step_saves_to_keep 0, a non-positive
                # Musubi prune threshold prunes nothing, and sd-scripts refuses 0.
                self.assertEqual(owner({"save_every_n_steps": 50, key: 0}, 1000), 20)
                self.assertEqual(owner({"save_every_n_steps": 50, key: None}, 1000), 20)
        # A trainer that prunes by its own default when the run sets no keep-last says so.
        self.assertIsNone(owner({"save_every_n_steps": 50, "unset_keep_last": "trainer_default"}, 1000))
        self.assertIsNone(owner({"save_every_n_steps": 50, "keep_last": None, "unset_keep_last": "trainer_default"}, 1000))
        self.assertEqual(owner({"save_every_n_steps": 50, "keep_last": 0, "unset_keep_last": "trainer_default"}, 1000), 20)

    def test_every_count_reads_the_one_owner(self) -> None:
        run = _resume(save_every=10, additional=50)
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        cases = {
            "_disk_warnings": (plan, lambda: plan._disk_warnings(run, {"save_every_n_steps": 10})),
            "_checkpoint_count_safety": (plan, lambda: plan._checkpoint_count_safety(run, trained_steps(run))),
            "_checkpoint_preflight_report": (plan, lambda: plan._checkpoint_preflight_report(allowed)),
            "_estimate_checkpoint_write_bytes": (plan, lambda: plan._estimate_checkpoint_write_bytes(allowed)),
            "_checkpoint_expected": (monitor, lambda: _monitor_expected(run, 10)),
        }
        for name, (module, call) in cases.items():
            with self.subTest(place=name), patch.object(
                module, "expected_checkpoints", wraps=training_artifacts.expected_checkpoints,
            ) as owner:
                call()
                owner.assert_called()

    def test_keep_last_zero_counts_every_save_in_plan_and_monitor_alike(self) -> None:
        # AI-Toolkit max_step_saves_to_keep 0 slices [:-0] and removes nothing: 1000 steps every 50 is 20 kept.
        run = ai_toolkit_run()
        run["backend"]["config"]["native_config"] = {"save": {"save_every": 50, "max_step_saves_to_keep": 0}}
        checkpoint = plan._adapter_display(run)["checkpoint"]
        self.assertEqual(checkpoint, {"save_every_n_steps": 50, "keep_last": 0, "unset_keep_last": "trainer_default"})
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        self.assertIn("about 20 checkpoints", plan._disk_warnings(run, checkpoint)[0])
        refusals = [record["fact"] for record in plan._checkpoint_preflight_report(run) if record["severity"] == "error"]
        self.assertIn("about 20 checkpoints without pruning", refusals[0])
        self.assertEqual(plan._estimate_checkpoint_write_bytes(allowed)["count"], 20)
        self.assertEqual(_monitor_expected(run, checkpoint), 20)
        # A positive keep-last leaves fewer: no count, no warning, no refusal, in both.
        run["backend"]["config"]["native_config"]["save"]["max_step_saves_to_keep"] = 3
        checkpoint = plan._adapter_display(run)["checkpoint"]
        self.assertEqual(plan._disk_warnings(run, checkpoint), [])
        self.assertEqual([record for record in plan._checkpoint_preflight_report(run) if record["severity"] == "error"], [])
        self.assertEqual(plan._estimate_checkpoint_write_bytes(allowed)["count"], 0)
        self.assertIsNone(_monitor_expected(run, checkpoint))


def _counts(run: dict[str, Any]) -> dict[str, Any]:
    """A fresh run through the real backend display and all five places that count its checkpoints."""
    checkpoint = plan._adapter_display(run)["checkpoint"]
    allowed = {**run, "safety": {"allow_many_checkpoints": True}}
    return {
        "disk_warnings": plan._disk_warnings(run, checkpoint),
        "refusals": [record["fact"] for record in plan._checkpoint_preflight_report(run) if record["severity"] == "error"],
        "preflight": [record["fact"] for record in plan._checkpoint_preflight_report(allowed)],
        "write_count": plan._estimate_checkpoint_write_bytes(allowed)["count"],
        "monitor_expected": _monitor_expected(run, checkpoint),
    }


class CheckpointRetentionTests(unittest.TestCase):
    """Each backend's display says whether its trainer prunes step checkpoints, and every count agrees."""

    def _assert_pruned(self, run: dict[str, Any]) -> None:
        self.assertEqual(_counts(run), {"disk_warnings": [], "refusals": [], "preflight": [], "write_count": 0, "monitor_expected": None})

    def _assert_counted(self, run: dict[str, Any], expected: int) -> None:
        counts = _counts(run)
        self.assertIn(f"about {expected} checkpoints", counts["disk_warnings"][0])
        self.assertIn(f"about {expected} checkpoints without pruning", counts["refusals"][0])
        self.assertEqual(counts["preflight"], [f"checkpoint cadence implies about {expected} checkpoint(s)"])
        self.assertEqual(counts["write_count"], expected)
        self.assertEqual(counts["monitor_expected"], expected)

    def test_ai_toolkit_with_no_keep_last_is_pruned_by_the_trainer_default(self) -> None:
        # The pinned trainer keeps max_step_saves_to_keep (default 5) when the run sets none.
        for config in ({"save_every_n_steps": 50}, {"native_config": {"save": {"save_every": 50}}}):
            run = ai_toolkit_run()
            run["backend"]["config"].update(config)
            with self.subTest(config=config):
                self._assert_pruned(run)

    def test_ai_toolkit_keep_last_zero_keeps_every_save(self) -> None:
        for config in (
            {"save_every_n_steps": 50, "save_last_n_steps": 0},
            {"native_config": {"save": {"save_every": 50, "max_step_saves_to_keep": 0}}},
        ):
            run = ai_toolkit_run()
            run["backend"]["config"].update(config)
            with self.subTest(config=config):
                self._assert_counted(run, 20)

    def test_musubi_epoch_retention_does_not_prune_step_checkpoints(self) -> None:
        # Musubi prunes step checkpoints only by save_last_n_steps (which Kura owns), not by epochs.
        run = musubi_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "extra_args": ["--save_last_n_epochs", "2"]})
        self.assertNotIn("keep_last", plan._adapter_display(run)["checkpoint"])
        self._assert_counted(run, 20)

    def test_a_fresh_run_with_retention_states_no_count(self) -> None:
        sd = sd_scripts_run()
        sd["backend"]["config"].update({"save_every_n_steps": 50, "save_last_n_steps": 100})
        musubi = musubi_run()
        musubi["backend"]["config"].update({"save_every_n_steps": 50, "prune_checkpoints_before_step": 1000})
        for run in (sd, musubi):
            with self.subTest(backend=run["backend"]["name"]):
                self._assert_pruned(run)

    def test_the_guard_points_to_each_backends_own_retention(self) -> None:
        run = sd_scripts_run()
        run["backend"]["config"]["save_every_n_steps"] = 50
        counts = _counts(run)
        for text in (counts["refusals"][0], counts["disk_warnings"][0]):
            self.assertNotIn("prune_checkpoints_before_step", text)
            self.assertIn("kura run capabilities sd-scripts", text)


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
