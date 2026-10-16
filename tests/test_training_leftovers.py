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
        self.assertIn("--gpu chooses a RunPod GPU; use it with --executor runpod", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
