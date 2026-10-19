"""Review leftovers on the training side: argv values, old records, and Resume guidance."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from kura.backends import validate_backend_config
from kura.backends.musubi_command import command_musubi_tuner
from kura.backends.musubi_models import _musubi_flux2_model_version
from kura.backends.sd_scripts import training_state_contract_sd_scripts
from kura.run_commands.plan import _parse_duration_seconds
from kura.run_envelope import without_retired_train_run_keys
from kura.training_artifacts import expected_checkpoints

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend_fixtures import with_required  # noqa: E402


def _musubi(config: dict) -> dict:
    return {"id": "r", "type": "train", "recipe": {"steps": 1, "seed": 1}, "datasets": [{"id": "tiny"}],
            "backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", config)}}


class TrainingLeftoverTests(unittest.TestCase):
    def test_true_or_false_for_a_field_that_takes_a_value_is_refused_on_argv_backends(self) -> None:
        for backend, config in (
            ("sd-scripts", {"architecture": "sd15", "learning_rate": False}),
            ("musubi-tuner", {"architecture": "wan", "network_dim": True}),
        ):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "take a value, not true or false"):
                validate_backend_config({"backend": {"name": backend, "config": config}})

    def test_the_shared_musubi_flow_flags_are_fields_on_every_architecture(self) -> None:
        # Upstream's Krea 2 guide trains with shift 2.5 (docs/krea2.md); the trainer reads the shared flag.
        validate_backend_config(_musubi({"architecture": "krea2", "timestep_sampling": "shift", "discrete_flow_shift": 2.5, "weighting_scheme": "none"}))
        with self.assertRaisesRegex(ValueError, "requires discrete_flow_shift for architecture='zimage'"):
            run = _musubi({"architecture": "zimage"})
            del run["backend"]["config"]["discrete_flow_shift"]
            validate_backend_config(run)

    def test_a_compiled_manifest_is_planned_as_it_is(self) -> None:
        # A run compiled when the shared flags went through extra_args.
        frozen = {"backend": {"name": "musubi-tuner", "config": {"architecture": "qwen_image", "extra_args": ["--discrete_flow_shift", "2.2"], "learning_rate": True}}}
        validate_backend_config(frozen, frozen=True)

    def test_only_settings_and_source_mismatch_refusals_tell_a_resume_how_to_go_on(self) -> None:
        from kura.cli import _compile_refusal
        from kura.run_envelope import SettingsRefused
        from kura.training_artifacts import ResumeSourceMismatch

        resume = {"continuation": {"mode": "resume"}, "parent_run": "source-run"}
        settings = _compile_refusal(resume, SettingsRefused("run.yaml: x is not a training run setting."))
        self.assertIn("`kura run new --from source-run`", settings)
        self.assertIn("runs/source-run/resolved/env.lock", settings)
        self.assertNotIn("..", settings)
        # After a Kura update moves a trainer, the saved state's image or adapter differs.
        mismatch = _compile_refusal(resume, ResumeSourceMismatch("Resume runtime image identity differs from the source training state"))
        self.assertIn("`kura run new --from source-run`", mismatch)
        self.assertIn("runs/source-run/resolved/env.lock", mismatch)
        self.assertNotIn("fix the settings", mismatch)
        self.assertNotIn("kura run new", _compile_refusal(resume, ValueError("dataset 'tiny' digest does not match the current dataset files")))

    def test_krea2_fp8_base_without_fp8_scaled_is_refused_at_compile(self) -> None:
        run = _musubi({"architecture": "krea2", "model_bundle": "krea2-raw", "fp8_base": True})
        with self.assertRaisesRegex(ValueError, "Krea 2 supports only scaled fp8: set fp8_scaled: true"):
            command_musubi_tuner(run)

    def test_musubi_fp8_is_refused_naming_fp8_base(self) -> None:
        with self.assertRaisesRegex(ValueError, "'fp8': use fp8_base"):
            validate_backend_config({"backend": {"name": "musubi-tuner", "config": {"architecture": "framepack", "fp8": True}}})

    def test_a_klein_variant_named_in_model_base_in_any_spelling_must_agree(self) -> None:
        for base in ("flux-2-klein-9b.safetensors", "models/flux2-klein-9b_fp8", "black-forest-labs/FLUX.2-klein-base-9B"):
            run = _musubi({"architecture": "flux2", "model_version": "klein-4b"})
            run["model"] = {"base": base}
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, "names variant klein-(base-)?9b, but model_version is 'klein-4b'"):
                _musubi_flux2_model_version(run)

    def test_a_conditional_field_names_the_selector_it_depends_on(self) -> None:
        with self.assertRaisesRegex(ValueError, "set backend.config.model_arch, which it depends on"):
            validate_backend_config({"backend": {"name": "ai-toolkit", "config": {"bypass_guidance_embedding": True}}, "model": {"base": "x/y"}})

    def test_an_older_runs_records_still_read(self) -> None:
        # Selector spellings a run compiled before one spelling may hold.
        contract = training_state_contract_sd_scripts({"backend": {"name": "sd-scripts", "config": {"architecture": "flux", "mode": "lora"}}})
        self.assertEqual(contract["capability"], "best_effort_resume")
        # A frozen capacity duration with an upper-case unit; an authored one has the one form.
        self.assertEqual(_parse_duration_seconds("2H", recorded=True), 7200)
        with self.assertRaises(ValueError):
            _parse_duration_seconds("2H")
        # The removed A40 overrides are dropped when a Resume or --from copies the run.
        copied = without_retired_train_run_keys({"backend": {"name": "musubi-tuner", "config": {"architecture": "flux2", "allow_a40_large_micro_batch": True}}})
        self.assertEqual(copied["backend"]["config"], {"architecture": "flux2"})

    def test_a_resume_counts_its_saves_on_logical_multiples(self) -> None:
        from test_resume_steps import as_resume, sd_scripts_run

        run = as_resume(sd_scripts_run(), source_step=950, additional=150)
        # Saves land on logical multiples of 100 between 950 and 1100: 1000 and 1100.
        self.assertEqual(expected_checkpoints(run, {"save_every_n_steps": 100}, 150), 2)

    def test_run_new_refuses_gpu_without_runpod(self) -> None:
        from kura.cli import cmd_run_new

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(stderr):
            previous = Path.cwd()
            os.chdir(directory)
            try:
                code = cmd_run_new(argparse.Namespace(experiment="e", slug="s", backend="musubi-tuner", executor="docker", gpu=["NVIDIA A40"], source=None))
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertIn("--gpu: compute.gpu chooses a RunPod GPU; a local run uses the local GPU", stderr.getvalue())


    def test_run_new_writes_each_repeated_gpu_into_the_list_in_order(self) -> None:
        import yaml

        from kura.cli import build_parser, cmd_run_new

        args = build_parser().parse_args(["run", "new", "--experiment", "e", "--slug", "s", "--backend", "musubi-tuner",
                                          "--executor", "runpod", "--gpu", "NVIDIA RTX A5000", "--gpu", "NVIDIA A40"])
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                Path("workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cmd_run_new(args), 0)
                [run_yaml] = Path("runs").glob("*/run.yaml")
                run = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
            finally:
                os.chdir(previous)
        self.assertEqual(run["compute"]["gpu"], ["NVIDIA RTX A5000", "NVIDIA A40"])
        self.assertIn("compute.gpu", build_parser()._subparsers._group_actions[0].choices["run"]._subparsers._group_actions[0].choices["new"].format_help())


if __name__ == "__main__":
    unittest.main()
