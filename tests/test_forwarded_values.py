"""Forwarded values reach the trainer as written; values without an upstream baseline are required."""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from kura.backends import BACKENDS, backend_capabilities, validate_backend_config
from kura.backends.shared import argv_value
from kura.cli import cmd_run_capabilities

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.backend_fixtures import with_required  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only  # noqa: E402


def _run(backend: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"forwarded-{backend}", "type": "train",
        "model": {"base": "/models/base" if backend == "sd-scripts" else "example/model"},
        "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
        "backend": {"name": backend, "config": config},
    }


ARGV_RUNS = {
    "sd-scripts": lambda config: _run("sd-scripts", {
        "architecture": "sd15", "model_paths": {"base": "/models/base"},
        "dataset_config": {"datasets": [{"subsets": [{"dataset_id": "tiny", "num_repeats": 1}]}]},
        **config,
    }),
    "musubi-tuner": lambda config: _run("musubi-tuner", {
        "architecture": "flux2", "model_version": "klein-base-4b", "timestep_sampling": "flux2_shift",
        "model_paths": {"dit": "/models/dit", "vae": "/models/vae", "text_encoder": "/models/text"},
        **config,
    }),
}


def _compiled_argv(run: dict[str, Any], destination: Path) -> str:
    freeze_fixture(run, destination)
    return " ".join(BACKENDS[run["backend"]["name"]].compile(run, destination)["argv"])


# The pinned AI-Toolkit loader (toolkit/config.py:54-65 in the image digest pinned in
# kura.images), copied onto a subclass so the test leaves PyYAML's SafeLoader untouched.
class _PinnedAiToolkitLoader(yaml.SafeLoader):
    pass


_PinnedAiToolkitLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile("""^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


class ForwardedValueTests(unittest.TestCase):
    def test_one_function_renders_every_value_as_argv(self) -> None:
        self.assertEqual(argv_value("--flag", True), ["--flag"])
        self.assertEqual(argv_value("--flag", False), [])
        self.assertEqual(argv_value("--lr", 1e-4), ["--lr", "0.0001"])
        self.assertEqual(argv_value("--n", 0), ["--n", "0"])
        self.assertEqual(argv_value("--args", ["a=1", 2]), ["--args", "a=1", "2"])
        with self.assertRaisesRegex(ValueError, "not a mapping"):
            argv_value("--args", {"a": 1})
        # A value never becomes another option, such as an output directory Kura owns.
        for value in ([0.1, "--output_dir", "/elsewhere"], "--resume=/elsewhere"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "may not start with --"):
                argv_value("--learning_rate", value)
        self.assertEqual(argv_value("--shift", -1), ["--shift", "-1"])

    @posix_only(DATASET_IO)
    def test_sdxl_text_encoder_learning_rates_take_the_trainers_one_flag(self) -> None:
        run = ARGV_RUNS["sd-scripts"]({"architecture": "sdxl", "text_encoder_lr": [1e-5, 2e-5]})
        run["backend"]["config"] = with_required("sd-scripts", run["backend"]["config"])
        validate_backend_config(run)
        with tempfile.TemporaryDirectory() as directory:
            argv = _compiled_argv(run, Path(directory) / "sd-scripts")
        self.assertIn("--text_encoder_lr 1e-05 2e-05", argv)

    @posix_only(DATASET_IO)
    def test_zero_false_and_a_mistyped_value_reach_every_argv_trainer_as_written(self) -> None:
        written = {
            "learning_rate": 0, "network_dim": 0, "network_alpha": 1e-4, "optimizer_type": "NotAnOptimizer",
            "gradient_checkpointing": False,
        }
        extra = {"sd-scripts": {"mixed_precision": "bf16"}, "musubi-tuner": {"max_data_loader_n_workers": 0}}
        for backend, build in ARGV_RUNS.items():
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                run = build({**written, **extra[backend]})
                validate_backend_config(run)
                argv = _compiled_argv(run, Path(directory) / backend)
                for flag in ("--learning_rate 0", "--network_dim 0", "--network_alpha 0.0001", "--optimizer_type NotAnOptimizer"):
                    self.assertIn(flag, argv)
                self.assertNotIn("--gradient_checkpointing", argv)
                if backend == "musubi-tuner":
                    self.assertIn("--max_data_loader_n_workers 0", argv)

    def test_an_omitted_required_value_is_refused_naming_the_field_and_the_trainer_default(self) -> None:
        for backend, build in ARGV_RUNS.items():
            with self.subTest(backend=backend):
                run = build({})
                run["backend"]["config"] = with_required(backend, run["backend"]["config"])
                validate_backend_config(run)
                del run["backend"]["config"]["learning_rate"]
                with self.assertRaisesRegex(ValueError, r"requires learning_rate \(the trainer would otherwise use 2e-6 \(.*\)\)\. Kura adds no training default"):
                    validate_backend_config(run)
        lllite = ARGV_RUNS["sd-scripts"]({"architecture": "anima", "mode": "controlnet_lllite"})
        lllite["backend"]["config"]["model_paths"] = {"dit": "/models/dit", "qwen3": "/models/qwen3", "vae": "/models/vae"}
        validate_backend_config({**lllite, "backend": {"name": "sd-scripts", "config": with_required("sd-scripts", lllite["backend"]["config"])}})
        self.assertNotIn("network_dim", with_required("sd-scripts", lllite["backend"]["config"]))

    def test_a_resume_that_does_not_compile_is_sent_to_fork_from_weight(self) -> None:
        from kura.cli import _compile_refusal

        run = ARGV_RUNS["musubi-tuner"]({"learning_rate": 1e-4, "network_dim": 4, "network_alpha": 4})
        with self.assertRaisesRegex(ValueError, "requires optimizer_type") as caught:
            validate_backend_config(run)
        self.assertNotIn("Fork from Weight", _compile_refusal(run, caught.exception))
        run["continuation"] = {"mode": "resume"}
        self.assertIn("create a Fork from Weight and fix the settings there", _compile_refusal(run, caught.exception))
        # Planning the frozen manifest of such a run reads it as it is.
        validate_backend_config(run, frozen=True)

    def test_capabilities_list_each_required_value_with_the_default_it_replaces(self) -> None:
        sd_scripts = backend_capabilities("sd-scripts")["required_fields"]
        self.assertEqual(sd_scripts["network_dim"]["when_any"], [{"mode": ["lora"]}])
        self.assertIn("library/args.py:87", sd_scripts["learning_rate"]["otherwise"])
        musubi = backend_capabilities("musubi-tuner")["required_fields"]
        self.assertIn("qwen_image", musubi["timestep_sampling"]["when_any"][0]["architecture"])
        self.assertNotIn("ideogram4", musubi["timestep_sampling"]["when_any"][0]["architecture"])
        # A required field with no clause of its own is required wherever its condition applies.
        self.assertEqual(musubi["h3_guidance_loss_scale"]["when_any"], [{"architecture": ["minimax_h3"], "h3_loss_method": ["guidance"]}])
        self.assertEqual(backend_capabilities("ai-toolkit")["required_fields"], {})

    def test_the_capabilities_text_lists_the_required_values_the_refusal_points_to(self) -> None:
        for backend, expected in (
            ("sd-scripts", "  guidance_scale when architecture=flux1; the trainer would otherwise use 3.5 (library/flux_train_utils.py:773)"),
            ("musubi-tuner", "  learning_rate; the trainer would otherwise use 2e-6 (training/parser_common.py:323)"),
        ):
            with self.subTest(backend=backend):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cmd_run_capabilities(argparse.Namespace(backend=backend, json=False)), 0)
                self.assertIn("backend.config fields you must write (Kura adds no training default):", stdout.getvalue())
                self.assertIn(expected, stdout.getvalue())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cmd_run_capabilities(argparse.Namespace(backend="ai-toolkit", json=False))
        self.assertNotIn("fields you must write", stdout.getvalue())

    def test_the_ai_toolkit_ui_names_the_trainer_reads_are_model_arch_values(self) -> None:
        # The pinned trainer drops a :tag and reads flex1 as flux (toolkit/config_modules.py:768-775).
        choices = backend_capabilities("ai-toolkit")["config_value_choices"]["model_arch"]
        for value in ("flex1", "zimage:turbo", "flux"):
            self.assertIn(value, choices)
        self.assertNotIn("sd15", choices)  # a UI name the trainer does not read; the arch is sd1

    def test_a_key_written_with_no_value_is_refused_on_every_backend(self) -> None:
        for backend, config in (
            ("sd-scripts", {"architecture": "sd15", "learning_rate": None}),
            ("musubi-tuner", {"architecture": "wan", "network_dim": None}),
            ("ai-toolkit", {"model_arch": "sdxl", "learning_rate": None}),
        ):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, r"key\(s\) written with no value: (learning_rate|network_dim); write a value, or remove the key"):
                    validate_backend_config({"backend": {"name": backend, "config": config}, "model": {"base": "x/y"}})

    @posix_only(DATASET_IO)
    def test_the_pinned_ai_toolkit_loader_reads_the_values_as_written(self) -> None:
        run = _run("ai-toolkit", {
            "model_arch": "sdxl", "learning_rate": 1e-5, "network_dim": 0, "network_alpha": 1e-4,
            "gradient_checkpointing": False, "low_vram": False,
        })
        validate_backend_config(run)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            freeze_fixture(run, destination, target="sample.png")
            BACKENDS["ai-toolkit"].compile(run, destination)
            text = (destination / "ai-toolkit.yaml").read_text(encoding="utf-8")
        process = yaml.load(text, Loader=_PinnedAiToolkitLoader)["config"]["process"][0]
        self.assertEqual(process["train"]["lr"], 1e-5)
        self.assertIs(process["train"]["gradient_checkpointing"], False)
        self.assertEqual(process["network"]["linear"], 0)
        self.assertEqual(process["network"]["linear_alpha"], 1e-4)
        self.assertIs(process["model"]["low_vram"], False)


if __name__ == "__main__":
    unittest.main()
