"""One bounded write estimate (docs/adr/local-disk-protection.md, decision 3): `run_write_estimate`
sizes the downloads Kura makes and bounds the run's own weights-sized files by its rank, for the
local launch, the RunPod container-disk check, and the plan."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import kura.run_commands.plan as plan
from kura.run_envelope import validate_train_run_fields, without_retired_train_run_keys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_resume_steps import ai_toolkit_run, musubi_run, sd_scripts_run  # noqa: E402

MIB = 1024**2
GIB = 1024**3


def _roomy(real_probe, free_bytes: int = 1024**5):
    def probe(paths, config=None):
        return {name: replace(status, linux_free_bytes=free_bytes, host_free_bytes=free_bytes, effective_free_bytes=free_bytes,
                              confidence="exact") for name, status in real_probe(paths, config).items()}
    return probe


class BoundTests(unittest.TestCase):
    def test_the_bound_holds_every_file_the_acceptance_runs_wrote(self) -> None:
        # Measured over the 13 acceptance runs of 2026-10-17 (SDXL, Anima, Z-Image, FLUX.2 Klein 4B,
        # MOSS-Image, Iris-3B on sd-scripts, Musubi Tuner, and AI-Toolkit): checkpoints 44-163 MiB at
        # rank 16 and 267 MiB at rank 32 (Z-Image on Musubi Tuner), at most 10.2 MiB per rank;
        # training-state folders 133-403 MiB, held here to the smallest rank, 16.
        for kind, rank, measured_mib in (("checkpoint", 16, 44), ("checkpoint", 16, 163), ("checkpoint", 32, 267),
                                         ("state", 16, 133), ("state", 16, 403)):
            files = 1 if kind == "checkpoint" else plan.STATE_WEIGHT_FILES
            with self.subTest(kind=kind, measured_mib=measured_mib):
                self.assertGreaterEqual(files * rank * plan.WEIGHTS_BYTES_PER_RANK, measured_mib * MIB)

    def test_each_backend_counts_its_checkpoints_trainer_states_and_kura_copies(self) -> None:
        def bound(build, *, executor="docker", **config):
            run = build()
            run["backend"]["config"].pop("save_every_n_steps", None)
            run["backend"]["config"].update(config)
            return plan.own_write_bound(run, executor=executor)

        # Musubi Tuner, cadence 50 over 1000 steps, two kept generations: 21 checkpoints at once
        # (no pruning while it trains); the trainer keeps the states of the last cadence plus the
        # one being written and its final state (3); Kura keeps two copies and stages one (3).
        musubi = bound(musubi_run, save_every_n_steps=50)
        self.assertEqual((musubi["checkpoints"], musubi["trainer_states"], musubi["kura_state_copies"]), (21, 3, 3))
        self.assertEqual(musubi["weights_files"], 21 + 3 * (3 + 3))
        self.assertEqual(musubi["bytes"], musubi["weights_files"] * 4 * 64 * MIB)
        # sd-scripts with no cadence writes its final file and its final state only.
        sd = bound(sd_scripts_run)
        self.assertEqual((sd["checkpoints"], sd["trainer_states"], sd["kura_state_copies"], sd["rank"]), (1, 1, 3, 8))
        # AI-Toolkit: Kura's hook keeps two state folders and stages a third, and the trainer
        # writes its own optimizer.pt beside its checkpoints: 4 states.
        aitk = bound(ai_toolkit_run)
        self.assertEqual((aitk["checkpoints"], aitk["trainer_states"], aitk["kura_state_copies"]), (1, 4, 3))
        self.assertTrue(aitk["trainer_default_saves_not_counted"])
        # On RunPod Kura's copies are made after the download, on the local machine, not on the Pod.
        self.assertEqual(bound(musubi_run, executor="runpod", save_every_n_steps=50)["kura_state_copies"], 0)

    def test_one_kept_generation_and_no_training_state_count_less(self) -> None:
        one = musubi_run()
        one["backend"]["config"]["save_every_n_steps"] = 50
        one["recovery"] = {"training_state": {"keep_generations": 1}}
        counted = plan.own_write_bound(one, executor="docker")
        self.assertEqual((counted["trainer_states"], counted["kura_state_copies"]), (2, 2))
        off = musubi_run()
        off["recovery"] = {"training_state": {"enabled": False}}
        counted = plan.own_write_bound(off, executor="docker")
        self.assertEqual((counted["trainer_states"], counted["kura_state_copies"]), (0, 0))

    def test_the_state_window_is_the_one_the_trainer_is_given(self) -> None:
        from kura.training_artifacts import managed_state_save_args

        run = musubi_run()
        run["backend"]["config"]["save_every_n_steps"] = 50
        import kura.training_artifacts as training_artifacts

        with patch.object(plan, "state_retention_window", wraps=training_artifacts.state_retention_window) as estimate, \
                patch.object(training_artifacts, "state_retention_window", wraps=training_artifacts.state_retention_window) as command:
            plan.own_write_bound(run, executor="docker")
            managed_state_save_args(run, 50)
        self.assertEqual(estimate.call_args, command.call_args)
        self.assertEqual(estimate.call_args.args[1], 50)

    def test_a_run_without_a_rank_is_not_bounded_and_says_so(self) -> None:
        run = sd_scripts_run()
        run["backend"]["config"].pop("network_dim")
        counted = plan.own_write_bound(run, executor="docker")
        self.assertEqual((counted["rank"], counted["bytes"]), (None, 0))
        local = plan.own_write_text(counted, executor="docker")
        self.assertIn("not bounded", local)
        self.assertIn("network_dim", local)
        self.assertIn("disk watch", local)
        # A Pod has no disk watch: the RunPod line says to leave room instead.
        remote = plan.own_write_text(plan.own_write_bound(run, executor="runpod"), executor="runpod")
        self.assertIn("not bounded", remote)
        self.assertIn("leave room in runpod.container_disk_gb", remote)
        self.assertNotIn("disk watch", remote)


class OneOwnerTests(unittest.TestCase):
    def test_the_local_launch_the_runpod_check_and_the_plan_read_the_one_estimate(self) -> None:
        run = musubi_run()
        run["safety"] = {"allow_storage_risk": True}
        seen = []
        real = plan.run_write_estimate

        def spy(*args, **kwargs):
            seen.append(kwargs["executor"])
            return real(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(plan, "run_write_estimate", side_effect=spy), \
                patch.object(plan, "_runpod_input_transfer_estimate", return_value=None), \
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
                patch.object(plan, "probe_storages", side_effect=_roomy(plan.probe_storages)):
            workspace = Path(directory).resolve()
            local = plan._local_launch_disk_preflight(workspace, run, {}, enforce_model_download_safety=False, download_estimate={"bytes": 5 * GIB})
            remote = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 500}, {"bytes": 5 * GIB})
        self.assertEqual(seen, ["docker", "runpod"])
        own = plan.own_write_bound(run, executor="docker")["bytes"]
        self.assertEqual(local["paths"]["workspace"]["estimated_write_bytes"], own)
        self.assertEqual(local["paths"]["hf_cache"]["estimated_write_bytes"], 5 * GIB)
        self.assertEqual(remote["estimated_write_bytes"], 5 * GIB + plan.own_write_bound(run, executor="runpod")["bytes"])


class LaunchCheckTests(unittest.TestCase):
    def test_the_local_launch_refuses_when_free_space_minus_the_estimate_is_below_the_reserve(self) -> None:
        run = musubi_run()
        run["backend"]["config"]["save_every_n_steps"] = 50
        run["safety"] = {"allow_storage_risk": True}
        own = plan.own_write_bound(run, executor="docker")["bytes"]
        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
            workspace = Path(directory).resolve()
            with patch.object(plan, "probe_storages", side_effect=_roomy(plan.probe_storages, 20 * GIB + own - 1)):
                with self.assertRaisesRegex(ValueError, "requires at least"):
                    plan._local_launch_disk_preflight(workspace, run, {}, enforce_model_download_safety=False, download_estimate={"bytes": 0})
            with patch.object(plan, "probe_storages", side_effect=_roomy(plan.probe_storages, 20 * GIB + own)):
                plan._local_launch_disk_preflight(workspace, run, {}, enforce_model_download_safety=False, download_estimate={"bytes": 0})

    def test_runpod_refuses_only_when_measured_writes_alone_exceed_the_container_disk(self) -> None:
        run = musubi_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "network_dim": 64})
        with patch.object(plan, "_runpod_input_transfer_estimate", return_value={"remote_peak_bytes": 2 * GIB}):
            # The bound alone is far above 30 GiB: shown, not refused.
            payload = plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 30}, {"bytes": 10 * GIB})
            report = plan._runpod_disk_preflight_report(run, {"container_disk_gb": 30}, {"bytes": 10 * GIB})
            with self.assertRaisesRegex(ValueError, "container_disk_gb=30 .* measured .* 31 GiB"):
                plan._runpod_launch_disk_preflight(run, {"container_disk_gb": 30}, {"bytes": 29 * GIB})
        self.assertGreater(payload["estimated_write_bytes"], 30 * GIB)
        self.assertEqual(report[0]["severity"], "warning")
        self.assertIn("weights-sized files", report[0]["fact"])


class RetiredKeyTests(unittest.TestCase):
    def test_the_declared_estimates_are_refused_naming_what_replaces_them(self) -> None:
        for key in ("checkpoint_estimate_gb", "disk_cache_estimate_gb", "model_download_estimate_gb"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"safety.{key} is retired.*disk watch"):
                validate_train_run_fields({"safety": {key: 2}})
        self.assertEqual(without_retired_train_run_keys({"safety": {"checkpoint_estimate_gb": 2, "disk_cache_estimate_gb": 1,
                                                                    "model_download_estimate_gb": 9, "allow_storage_risk": True}}),
                         {"safety": {"allow_storage_risk": True}})
        # The older sd-scripts spelling is dropped on copy too, no longer moved under safety.
        self.assertEqual(without_retired_train_run_keys({"backend": {"name": "sd-scripts", "config": {"disk_cache_estimate_gb": 1}}}),
                         {"backend": {"name": "sd-scripts", "config": {}}})

    def test_the_trainer_downloads_and_disk_cache_are_shown_not_counted(self) -> None:
        report = plan._model_download_preflight_report(ai_toolkit_run(), {"bytes": 0}, executor="docker", local_cache=Path("/c"))
        fact = report[0]["fact"]
        self.assertIn("not counted", fact)
        self.assertIn("disk watch", fact)
        self.assertNotIn("estimate_gb", fact)
        cache = plan._disk_cache_preflight_report(musubi_run())
        self.assertEqual(cache[0]["severity"], "info")
        self.assertIn("not counted", cache[0]["fact"])
        self.assertNotIn("estimate_gb", cache[0]["fact"])


if __name__ == "__main__":
    unittest.main()
