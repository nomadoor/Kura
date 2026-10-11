"""How many steps a run trains is decided once, and every checkpoint and disk estimate reads it."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
import subprocess
import sys
import tempfile
from dataclasses import replace
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

import kura.monitor as monitor
import kura.run_commands.plan as plan
import kura.training_artifacts as training_artifacts
from kura.backends.ai_toolkit import compile_ai_toolkit
from kura.backends.musubi_command import command_musubi_tuner
from kura.backends.sd_scripts import command_sd_scripts
from kura.run_envelope import common_recipe
from kura.training_artifacts import trained_steps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only  # noqa: E402
from tests.test_resume_steps import ai_toolkit_run, as_resume, musubi_run, publish_source, sd_scripts_run  # noqa: E402


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
        self.assertEqual(estimates["write_count"], 6)
        self.assertEqual(estimates["monitor_expected"], 5)

    def test_a_resume_longer_than_its_recipe_trips_the_checkpoint_guard(self) -> None:
        # 1000-step recipe resumed +5000 with a cadence of 500: 10 checkpoints, not 2.
        estimates = _estimates(_resume(save_every=500, additional=5000), 500)
        self.assertEqual(len(estimates["disk_warnings"]), 1)
        self.assertIn("about 10 checkpoints over the 5000 steps this run trains;", estimates["disk_warnings"][0])
        self.assertIn("about 10 checkpoints without pruning over the 5000 steps this run trains", estimates["safety_refusal"] or "")
        self.assertEqual(estimates["preflight"], ["checkpoint cadence implies about 10 checkpoint(s) over the 5000 steps this run trains"])
        self.assertEqual(estimates["write_count"], 11)
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
        # The disk estimate counts saves over the trainer's native span instead (PeakCheckpointsTests).
        del allowed
        cases = {
            "_disk_warnings": (plan, lambda: plan._disk_warnings(run, {"save_every_n_steps": 10})),
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
        # 1000-step source resumed +50 with a cadence of 10: 5 step saves and the final file of 1 GiB on both executors.
        run = _resume(save_every=10, additional=50)
        run["safety"] = {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 1, "allow_storage_risk": True}
        expected = {"bytes": 6 * 1024**3, "count": 6, "per_checkpoint_gib": 1}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                    patch.object(plan, "peak_checkpoints", wraps=training_artifacts.peak_checkpoints) as runpod_owner:
                runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 50}, {"bytes": 0})
            with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                    patch.object(plan, "peak_checkpoints", wraps=training_artifacts.peak_checkpoints) as local_owner:
                local = plan._local_launch_disk_preflight(
                    Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                    enforce_model_download_safety=False, download_estimate={"bytes": 0},
                )
        runpod_owner.assert_called_once()
        local_owner.assert_called_once()
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
        def owner(checkpoint: dict[str, Any], steps: int | None, build=sd_scripts_run) -> int | None:
            return training_artifacts.expected_checkpoints(build(), checkpoint, steps)

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

    def test_the_count_reads_the_cadence_the_trainer_is_given(self) -> None:
        # With no cadence set, Musubi Tuner is given the recipe's steps (one save), sd-scripts
        # writes no step saves, and AI-Toolkit saves at a default Kura does not know.
        self.assertEqual(training_artifacts.expected_checkpoints(musubi_run(), {}, 1000), 1)
        self.assertIsNone(training_artifacts.expected_checkpoints(sd_scripts_run(), {}, 1000))
        self.assertIsNone(training_artifacts.expected_checkpoints(ai_toolkit_run(), {"last_step_save": "final_only"}, 1000))
        run = musubi_run()
        with patch.object(training_artifacts, "checkpoint_save_cadence", wraps=training_artifacts.checkpoint_save_cadence) as cadence:
            training_artifacts.expected_checkpoints(run, {"save_every_n_steps": 50}, 1000)
        cadence.assert_called_once_with(run, 50)
        # The plan's preflight line now names the one save a Musubi run with no cadence makes.
        records = [record["fact"] for record in plan._checkpoint_preflight_report(musubi_run())]
        self.assertEqual(records, ["checkpoint cadence implies about 1 checkpoint(s)"])

    def test_every_count_reads_the_one_owner(self) -> None:
        run = _resume(save_every=10, additional=50)
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        cases = {
            "_disk_warnings": (plan, lambda: plan._disk_warnings(run, {"save_every_n_steps": 10})),
            "_checkpoint_count_safety": (plan, lambda: plan._checkpoint_count_safety(run, trained_steps(run))),
            "_checkpoint_preflight_report": (plan, lambda: plan._checkpoint_preflight_report(allowed)),
            "_checkpoint_expected": (monitor, lambda: _monitor_expected(run, 10)),
        }
        # The disk estimate counts the peak instead (PeakCheckpointsTests).
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
        self.assertEqual(checkpoint, {"save_every_n_steps": 50, "keep_last": 0, "unset_keep_last": "trainer_default", "last_step_save": "final_only"})
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        self.assertIn("about 20 checkpoints", plan._disk_warnings(run, checkpoint)[0])
        refusals = [record["fact"] for record in plan._checkpoint_preflight_report(run) if record["severity"] == "error"]
        self.assertIn("about 20 checkpoints without pruning", refusals[0])
        self.assertEqual(plan._estimate_checkpoint_write_bytes(allowed)["count"], 20)
        self.assertEqual(_monitor_expected(run, checkpoint), 20)
        # A positive keep-last leaves fewer: no count, no warning, no refusal, in both; the disk
        # estimate counts the peak (the kept saves plus the one written before the cleanup).
        run["backend"]["config"]["native_config"]["save"]["max_step_saves_to_keep"] = 3
        checkpoint = plan._adapter_display(run)["checkpoint"]
        self.assertEqual(plan._disk_warnings(run, checkpoint), [])
        self.assertEqual([record for record in plan._checkpoint_preflight_report(run) if record["severity"] == "error"], [])
        self.assertEqual(plan._estimate_checkpoint_write_bytes(allowed)["count"], 4)
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

    def _assert_pruned(self, run: dict[str, Any], peak: int) -> None:
        """No kept count, so no warning, refusal, or monitor expectation; the disk estimate counts the peak."""
        self.assertEqual(_counts(run), {"disk_warnings": [], "refusals": [], "preflight": [], "write_count": peak, "monitor_expected": None})

    def _assert_counted(self, run: dict[str, Any], expected: int, peak: int) -> None:
        """The kept count drives every count but the disk estimate, which adds the final file."""
        counts = _counts(run)
        self.assertIn(f"about {expected} checkpoints", counts["disk_warnings"][0])
        self.assertIn(f"about {expected} checkpoints without pruning", counts["refusals"][0])
        self.assertEqual(counts["preflight"], [f"checkpoint cadence implies about {expected} checkpoint(s)"])
        self.assertEqual(counts["write_count"], peak)
        self.assertEqual(counts["monitor_expected"], expected)

    def test_ai_toolkit_with_no_keep_last_is_pruned_by_the_trainer_default(self) -> None:
        # The pinned trainer keeps max_step_saves_to_keep (default 5) when the run sets none.
        for config in ({"save_every_n_steps": 50}, {"native_config": {"save": {"save_every": 50}}}):
            run = ai_toolkit_run()
            run["backend"]["config"].update(config)
            with self.subTest(config=config):
                # Kura does not copy the trainer's default count: the peak is every save, which
                # AI-Toolkit writes on steps 50..950 (it skips the last step) plus its final file.
                self._assert_pruned(run, 20)

    def test_ai_toolkit_keep_last_zero_keeps_every_save(self) -> None:
        for config in (
            {"save_every_n_steps": 50, "save_last_n_steps": 0},
            {"native_config": {"save": {"save_every": 50, "max_step_saves_to_keep": 0}}},
        ):
            run = ai_toolkit_run()
            run["backend"]["config"].update(config)
            with self.subTest(config=config):
                self._assert_counted(run, 20, 20)

    def test_musubi_epoch_retention_does_not_prune_step_checkpoints(self) -> None:
        # Musubi prunes step checkpoints only by save_last_n_steps (which Kura owns), not by epochs.
        run = musubi_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "extra_args": ["--save_last_n_epochs", "2"]})
        self.assertNotIn("keep_last", plan._adapter_display(run)["checkpoint"])
        self._assert_counted(run, 20, 21)

    def test_a_fresh_run_with_retention_states_no_count(self) -> None:
        sd = sd_scripts_run()
        sd["backend"]["config"].update({"save_every_n_steps": 50, "save_last_n_steps": 100})
        musubi = musubi_run()
        musubi["backend"]["config"].update({"save_every_n_steps": 50, "prune_checkpoints_before_step": 1000})
        # sd-scripts prunes during training: a 100-step window over saves every 50 keeps 3
        # (100 // 50 + 1) and holds a fourth while it writes the next. Musubi prunes after
        # training ends, so every save and the final file are on disk at the peak.
        for run, peak in ((sd, 4), (musubi, 21)):
            with self.subTest(backend=run["backend"]["name"]):
                self._assert_pruned(run, peak)

    def test_the_guard_points_to_each_backends_own_retention(self) -> None:
        run = sd_scripts_run()
        run["backend"]["config"]["save_every_n_steps"] = 50
        counts = _counts(run)
        for text in (counts["refusals"][0], counts["disk_warnings"][0]):
            self.assertNotIn("prune_checkpoints_before_step", text)
            self.assertIn("kura run capabilities sd-scripts", text)


class PeakCheckpointsTests(unittest.TestCase):
    """How many checkpoints are on disk at once is decided with the retention rules, and only the disk estimate reads it."""

    def test_the_peak_follows_when_each_trainer_prunes(self) -> None:
        def peak(checkpoint: dict[str, Any], steps: int = 1000, build=musubi_run) -> int:
            run = build()
            run["recipe"]["steps"] = steps
            return training_artifacts.peak_checkpoints(run, checkpoint)["count"]

        # Every save: one per cadence over the steps, and the final file every trainer writes
        # besides them (sd-scripts and Musubi Tuner save on the last step too).
        self.assertEqual(peak({"save_every_n_steps": 50}), 21)
        self.assertEqual(peak({"save_every_n_steps": 50}, 1020), 21)
        self.assertEqual(peak({"save_every_n_steps": 5000}), 1)
        no_steps = musubi_run()
        del no_steps["recipe"]["steps"]
        self.assertIsNone(training_artifacts.peak_checkpoints(no_steps, {"save_every_n_steps": 50}))
        # AI-Toolkit decides saves on its 0-based iteration index 50..950 (the image names them
        # by the updates they hold, 51..951) and leaves the last step to its final file.
        final_only = {"save_every_n_steps": 50, "last_step_save": "final_only"}
        self.assertEqual(peak(final_only), 20)
        self.assertEqual(peak(final_only, 1020), 21)
        self.assertEqual(peak(final_only, 50), 1)
        # Musubi Tuner prunes after training: every save is on disk first.
        self.assertEqual(peak({"save_every_n_steps": 50, "prune_before_step": 1000}), 21)
        # AI-Toolkit removes older saves after writing the new one: keep_last plus one (at the
        # end, keep_last step saves and the final file), never more than every save.
        self.assertEqual(peak({**final_only, "keep_last": 3}), 4)
        self.assertEqual(peak({**final_only, "keep_last": "3"}), 4)
        self.assertEqual(peak({**final_only, "keep_last": 30}), 20)
        self.assertEqual(peak({**final_only, "keep_last": 0}), 20)
        # Its unset default is not copied: the conservative bound is every save.
        self.assertEqual(peak({**final_only, "unset_keep_last": "trainer_default"}), 20)
        # sd-scripts keeps the saves within the window (window // every + 1) and removes the
        # oldest after writing the next one; at the end the final file joins the kept saves.
        self.assertEqual(peak({"save_every_n_steps": 50, "retention_window_steps": 100}, build=sd_scripts_run), 4)
        self.assertEqual(peak({"save_every_n_steps": 10, "retention_window_steps": 30}, build=sd_scripts_run), 5)
        self.assertEqual(peak({"save_every_n_steps": 50, "retention_window_steps": 10}, build=sd_scripts_run), 2)
        self.assertEqual(peak({"save_every_n_steps": 50, "retention_window_steps": 5000}, build=sd_scripts_run), 21)

    def test_each_backend_display_declares_its_last_step_save(self) -> None:
        # The display says what the core needs; no backend name is read by the peak.
        for run, every_save in ((ai_toolkit_run(), 20), (musubi_run(), 21), (sd_scripts_run(), 21)):
            run["backend"]["config"]["save_every_n_steps"] = 50
            run["safety"] = {}
            with self.subTest(backend=run["backend"]["name"]):
                self.assertEqual(plan._estimate_checkpoint_write_bytes(run)["count"], every_save)
        self.assertEqual(plan._adapter_display(ai_toolkit_run())["checkpoint"]["last_step_save"], "final_only")

    def test_an_unset_cadence_counts_what_the_trainer_is_given(self) -> None:
        # sd-scripts gets no step cadence (final file only); Musubi Tuner is given the recipe's
        # steps (one step save and the final file); AI-Toolkit saves at a default Kura does not
        # know, so only its final file is counted and the estimate says so.
        expected = {
            "sd-scripts": {"bytes": 1024**3, "count": 1, "per_checkpoint_gib": 1},
            "musubi-tuner": {"bytes": 2 * 1024**3, "count": 2, "per_checkpoint_gib": 1},
            "ai-toolkit": {"bytes": 1024**3, "count": 1, "per_checkpoint_gib": 1, "trainer_default_saves_not_counted": True},
        }
        for run in (ai_toolkit_run(), musubi_run(), sd_scripts_run()):
            run["backend"]["config"].pop("save_every_n_steps", None)
            name = run["backend"]["name"]
            with self.subTest(backend=name):
                estimate = plan._estimate_checkpoint_write_bytes(run)
                self.assertEqual(estimate, expected[name])
                text = plan._checkpoint_estimate_text(estimate)
                if name == "ai-toolkit":
                    self.assertEqual(text, "checkpoints: 1 × 1 GiB (safety.checkpoint_estimate_gb; trainer-default saves not counted)")
                else:
                    self.assertNotIn("trainer-default", text)

    def test_a_logical_progress_resume_counts_saves_on_logical_multiples(self) -> None:
        # Resume +170 from step 1030 to 1200 with a cadence of 100. AI-Toolkit's progress is
        # logical: it saves on index 1100 (named 1101, the updates it holds) and leaves 1200 to
        # its final file: 2 (170 // 100 + 1 counted
        # from 1030 would land on 1130). sd-scripts' progress is logical too, and it saves on its
        # last step: 1100, 1200, and the final file: 3. Musubi Tuner counts progress from zero in
        # its process, 0..170: a save on 100 and the final file: 2.
        for build, expected in ((ai_toolkit_run, 2), (sd_scripts_run, 3), (musubi_run, 2)):
            source = build()
            source["backend"]["config"]["save_every_n_steps"] = 100
            run = as_resume(source, source_step=1030, additional=170)
            checkpoint = plan._adapter_display(run)["checkpoint"]
            with self.subTest(backend=source["backend"]["name"]):
                self.assertEqual(training_artifacts.peak_checkpoints(run, checkpoint)["count"], expected)

    def test_a_process_local_resume_caps_the_cadence_as_the_trainer_is_given(self) -> None:
        # Resume +50 with no cadence set: Kura gives Musubi Tuner a cadence of 50 (the steps
        # added), so it saves once and writes its final file. sd-scripts counts logical steps
        # on Resume, is given no cadence, and writes only its final file.
        for build, cadence, count in ((musubi_run, 50, 2), (sd_scripts_run, None, 1)):
            run = as_resume(build(), source_step=1000, additional=50)
            checkpoint = plan._adapter_display(run)["checkpoint"]
            with self.subTest(backend=run["backend"]["name"]):
                self.assertEqual(training_artifacts.checkpoint_save_cadence(run, checkpoint.get("save_every_n_steps")), cadence)
                self.assertEqual(training_artifacts.peak_checkpoints(run, checkpoint)["count"], count)

    def test_the_resume_cap_applies_only_when_kura_manages_state(self) -> None:
        for build in (musubi_run, sd_scripts_run):
            for configured, managed, expected in ((None, False, {"musubi-tuner": 1000, "sd-scripts": None}), (30, False, 30), (500, True, {"musubi-tuner": 50, "sd-scripts": 500}), (500, False, 500)):
                source = build()
                source["recovery"] = {"training_state": {"enabled": managed}}
                run = as_resume(source, source_step=1000, additional=50)
                name = run["backend"]["name"]
                want = expected[name] if isinstance(expected, dict) else expected
                with self.subTest(backend=name, configured=configured, managed=managed):
                    self.assertEqual(training_artifacts.checkpoint_save_cadence(run, configured), want)

    def test_the_disk_estimate_reads_the_peak_and_the_counts_do_not(self) -> None:
        run = sd_scripts_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "save_last_n_steps": 100})
        allowed = {**run, "safety": {"allow_many_checkpoints": True}}
        checkpoint = plan._adapter_display(run)["checkpoint"]
        with patch.object(plan, "peak_checkpoints", wraps=training_artifacts.peak_checkpoints) as owner:
            self.assertEqual(plan._estimate_checkpoint_write_bytes(allowed)["count"], 4)
            owner.assert_called_once_with(allowed, checkpoint)
        with patch.object(plan, "peak_checkpoints", wraps=training_artifacts.peak_checkpoints) as owner:
            self.assertEqual(plan._disk_warnings(run, checkpoint), [])
            plan._checkpoint_count_safety(run, 1000)
            plan._checkpoint_preflight_report(run)
            self.assertIsNone(_monitor_expected(run, checkpoint))
            owner.assert_not_called()

    def test_the_disk_estimate_does_not_wait_for_allow_many_checkpoints(self) -> None:
        # A plain run with few checkpoints and a Musubi run whose prune passes the guard both
        # write their peak; allowing many checkpoints changes the guard, not the estimate.
        plain = musubi_run()
        plain["backend"]["config"]["save_every_n_steps"] = 500
        pruned = musubi_run()
        pruned["backend"]["config"].update({"save_every_n_steps": 50, "prune_checkpoints_before_step": 1000})
        for name, run, peak in (("plain", plain, 3), ("musubi prune", pruned, 21)):
            with self.subTest(run=name):
                plan._checkpoint_count_safety(run, trained_steps(run))
                for safety in ({}, {"allow_many_checkpoints": True}):
                    estimate = plan._estimate_checkpoint_write_bytes({**run, "safety": safety})
                    self.assertEqual(estimate, {"bytes": peak * 1024**3, "count": peak, "per_checkpoint_gib": 1})
                sized = plan._estimate_checkpoint_write_bytes({**run, "safety": {"checkpoint_estimate_gb": 3}})
                self.assertEqual(sized, {"bytes": 3 * peak * 1024**3, "count": peak, "per_checkpoint_gib": 3})

    def test_runpod_refuses_a_container_disk_below_the_peak(self) -> None:
        run = musubi_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "prune_checkpoints_before_step": 1000})
        run["safety"] = {"checkpoint_estimate_gb": 2}
        with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                patch.object(plan, "_disk_cache_estimate", return_value={}):
            with self.assertRaises(ValueError) as refused:
                plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 30}, {"bytes": 0})
            report = plan._runpod_disk_preflight_report(run, {"container_disk_gb": 30}, {"bytes": 0})
            run["safety"]["allow_runpod_disk_risk"] = True
            result = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 30}, {"bytes": 0})
            allowed = plan._runpod_disk_preflight_report(run, {"container_disk_gb": 30}, {"bytes": 0})
        self.assertIn("container_disk_gb=30 is below estimated remote writes of about 42 GiB", str(refused.exception))
        part = "checkpoints: 21 × 2 GiB (safety.checkpoint_estimate_gb)"
        self.assertIn(part, str(refused.exception))
        self.assertEqual(report[0]["severity"], "error")
        self.assertIn(part, report[0]["fact"])
        self.assertEqual(allowed[0]["severity"], "info")
        self.assertIn(part, allowed[0]["fact"])
        self.assertEqual(result["estimated_write_bytes"], 42 * 1024**3)

    def test_local_disk_refusal_and_pass_show_the_checkpoint_part(self) -> None:
        run = musubi_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "prune_checkpoints_before_step": 1000})
        run["safety"] = {"checkpoint_estimate_gb": 2, "allow_storage_risk": True}
        part = "checkpoints: 21 × 2 GiB (safety.checkpoint_estimate_gb)"
        real_probe = plan.probe_storages

        def roomy(paths, config=None):
            # The CI machine's real free space must not decide the test: give every path 1 PiB.
            return {name: replace(status, linux_free_bytes=1024**5, host_free_bytes=1024**5, effective_free_bytes=1024**5)
                    for name, status in real_probe(paths, config).items()}

        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                patch.object(plan, "_disk_cache_estimate", return_value={}), \
                patch.object(plan, "probe_storages", side_effect=roomy):
            workspace = Path(directory).resolve()
            refused = plan._local_disk_preflight_report(run, workspace, {"docker": {"min_free_gb": 10**9}}, {"bytes": 0})
            passed = plan._local_disk_preflight_report(run, workspace, {"docker": {"min_free_gb": 1}}, {"bytes": 0})
        self.assertEqual(refused[0]["severity"], "error")
        self.assertIn(part, refused[0]["fact"])
        self.assertEqual(passed[0]["severity"], "info")
        self.assertIn(part, passed[0]["fact"])

    def test_local_disk_refusal_and_pass_name_the_setting_behind_the_minimum(self) -> None:
        real_probe = plan.probe_storages

        def roomy(paths, config=None):
            return {name: replace(status, linux_free_bytes=1024**5, host_free_bytes=1024**5, effective_free_bytes=1024**5)
                    for name, status in real_probe(paths, config).items()}

        def report(run, docker):
            with tempfile.TemporaryDirectory() as directory, \
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                    patch.object(plan, "_disk_cache_estimate", return_value={}), \
                    patch.object(plan, "probe_storages", side_effect=roomy):
                return plan._local_disk_preflight_report(run, Path(directory).resolve(), {"docker": docker}, {"bytes": 0})[0]

        run = musubi_run()
        run["safety"] = {"allow_storage_risk": True}
        workspace_part = "GiB minimum free, set by docker.min_free_gb in workspace.yaml"
        refused = report(run, {"min_free_gb": 10**9})
        self.assertEqual(refused["severity"], "error")
        self.assertIn(f"{10**9} {workspace_part}, plus estimated writes", refused["fact"])
        passed = report(run, {})
        self.assertEqual(passed["severity"], "info")
        self.assertIn(f"100 {workspace_part} (default), plus estimated writes", passed["fact"])
        run["safety"]["max_run_disk_gb"] = 10**9
        raised = report(run, {})
        self.assertIn(f"{10**9} GiB minimum free, set by safety.max_run_disk_gb in run.yaml", raised["fact"])

    def test_local_and_runpod_disk_preflights_count_the_same_peak(self) -> None:
        plain = musubi_run()
        plain["backend"]["config"]["save_every_n_steps"] = 500
        for name, run, retention, peak in (
            ("sd-scripts", sd_scripts_run(), {"save_every_n_steps": 50, "save_last_n_steps": 100}, 4),
            ("musubi-tuner", musubi_run(), {"save_every_n_steps": 50, "prune_checkpoints_before_step": 1000}, 21),
            ("no retention", plain, {}, 3),
        ):
            run["backend"]["config"].update(retention)
            run["safety"] = {"checkpoint_estimate_gb": 1, "allow_storage_risk": True}
            expected = {"bytes": peak * 1024**3, "count": peak, "per_checkpoint_gib": 1}
            with self.subTest(backend=name), tempfile.TemporaryDirectory() as directory:
                with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                        patch.object(plan, "_disk_cache_estimate", return_value={}):
                    runpod = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 500}, {"bytes": 0})
                with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                        patch.object(plan, "_disk_cache_estimate", return_value={}):
                    local = plan._local_launch_disk_preflight(
                        Path(directory).resolve(), run, {"docker": {"min_free_gb": 1}},
                        enforce_model_download_safety=False, download_estimate={"bytes": 0},
                    )
                self.assertEqual(runpod["estimates"]["checkpoints"], expected)
                self.assertEqual(local["estimates"]["checkpoints"], expected)
                self.assertEqual(runpod["estimated_write_bytes"], expected["bytes"])
                self.assertEqual(local["paths"]["workspace"]["estimated_write_bytes"], expected["bytes"])


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



def _compiled_cadence(run: dict[str, Any], scratch: Path) -> int | str | None:
    """The step-save cadence each trainer is given, read from its compiled command or config."""
    name = run["backend"]["name"]
    if name == "musubi-tuner":
        found = re.findall(r"--save_every_n_steps (\d+)", command_musubi_tuner(run)["argv"][2])
        return int(found[-1]) if found else None
    if name == "sd-scripts":
        # A managed run's command carries a JSON argv; an unmanaged one a shell line.
        found = re.findall(r'"?--save_every_n_steps"?[ ,]"?(\d+)', command_sd_scripts(run)["argv"][2])
        return int(found[-1]) if found else None
    destination = scratch / "ai-toolkit"
    destination.parent.mkdir(parents=True, exist_ok=True)
    freeze_fixture(run, destination.parent)
    compile_ai_toolkit(run, destination)
    config = yaml.safe_load(destination.with_suffix(".yaml").read_text(encoding="utf-8"))
    return config["config"]["process"][0]["save"].get("save_every") or "trainer_default"


class CheckpointCadenceParityTests(unittest.TestCase):
    @posix_only(DATASET_IO)
    def test_the_peak_reads_the_cadence_each_trainer_is_given(self) -> None:
        for build in (musubi_run, sd_scripts_run, ai_toolkit_run):
            for configured in (None, 30):
                source = build()
                if configured is not None:
                    source["backend"]["config"]["save_every_n_steps"] = configured
                name = source["backend"]["name"]
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = publish_source(root, source)
                    resumed = as_resume(source, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"], additional=20)
                    for label, run in (("fresh", source), ("resume", resumed)):
                        with self.subTest(backend=name, configured=configured, run=label):
                            checkpoint = plan._adapter_display(run)["checkpoint"]
                            self.assertEqual(
                                training_artifacts.checkpoint_save_cadence(run, checkpoint.get("save_every_n_steps")),
                                _compiled_cadence(run, root / label),
                            )
                    if name == "ai-toolkit":
                        continue  # AI-Toolkit refuses a Resume without managed state.
                    unmanaged = deepcopy(source)
                    unmanaged["recovery"] = {"training_state": {"enabled": False}}
                    unmanaged_resume = as_resume(unmanaged, artifact_id=manifest["id"], manifest_sha256=manifest["manifest_sha256"], additional=20)
                    for label, run in (("unmanaged-fresh", unmanaged), ("unmanaged-resume", unmanaged_resume)):
                        with self.subTest(backend=name, configured=configured, run=label):
                            self.assertEqual(
                                training_artifacts.checkpoint_save_cadence(run, configured),
                                _compiled_cadence(run, root / label),
                            )

    def test_every_command_builder_and_reader_asks_the_one_owner(self) -> None:
        import kura.backends.musubi_command as musubi_command

        owner = training_artifacts.checkpoint_save_cadence
        for build, module, call in (
            (musubi_run, musubi_command, lambda run: musubi_command.command_musubi_tuner(run)),
            (sd_scripts_run, training_artifacts, lambda run: command_sd_scripts(run)),
            (musubi_run, plan, lambda run: plan._training_state_cadence(run)),
            (musubi_run, training_artifacts, lambda run: plan._checkpoint_preflight_report(run)),
            (musubi_run, training_artifacts, lambda run: plan._estimate_checkpoint_write_bytes(run)),
        ):
            run = build()
            with self.subTest(module=module.__name__, backend=run["backend"]["name"]), \
                    patch.object(module, "checkpoint_save_cadence", wraps=owner) as cadence:
                call(run)
                cadence.assert_called()

    def test_the_plan_formats_the_owners_state_cadence(self) -> None:
        cases = ((musubi_run, None, 1000), (sd_scripts_run, None, 1000), (ai_toolkit_run, None, "trainer default"), (musubi_run, 30, 30))
        for build, configured, shown in cases:
            run = build()
            if configured is not None:
                run["backend"]["config"]["save_every_n_steps"] = configured
            with self.subTest(backend=run["backend"]["name"], configured=configured):
                self.assertEqual(plan._training_state_cadence(run), shown)


class RunPodContainerDiskDefaultTests(unittest.TestCase):
    def test_the_plan_and_the_executor_read_one_container_disk_default(self) -> None:
        import kura.executors.runpod as runpod

        self.assertEqual(runpod.DEFAULT_CONTAINER_DISK_GB, 150)
        run = musubi_run()
        with patch.object(plan, "_runpod_input_transfer_estimate", return_value=None):
            payload = plan._runpod_launch_disk_preflight(run, {}, {"bytes": 0})
        settings = runpod._runpod_settings({"gpu_type_ids": ["NVIDIA RTX A5000"]})
        self.assertEqual(payload["container_disk_gib"], runpod.DEFAULT_CONTAINER_DISK_GB)
        self.assertEqual(settings["container_disk_gb"], runpod.DEFAULT_CONTAINER_DISK_GB)
        create = runpod._runpod_graphql_create_input({"gpuTypeIds": ["NVIDIA RTX A5000"]})
        self.assertEqual(create["containerDiskInGb"], runpod.DEFAULT_CONTAINER_DISK_GB)


if __name__ == "__main__":
    unittest.main()
