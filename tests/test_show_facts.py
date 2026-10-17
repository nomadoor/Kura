"""Capabilities and plan show the facts adapters already own; sd-scripts reports independent problems together."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from copy import deepcopy
from typing import Any

import yaml

from kura.backends import BACKENDS, backend_capabilities, validate_backend_config
from kura.backends.ai_toolkit import display_ai_toolkit
from kura.backends.ai_toolkit_baseline import load_baseline
from kura.backends.musubi_models import _musubi_model_expectations
from kura.backends.sd_scripts import ENTRYPOINTS, command_sd_scripts, display_sd_scripts
from kura.backends.sd_scripts_models import OPTIONAL_ROLES, ROLE_CONTRACTS
from kura.backends.shared import MODEL_DOWNLOAD_KEYS
from kura.cli import cmd_run_capabilities
from kura.images import BUILD_SOURCES, PINNED_IMAGES
from kura.run_commands.plan import format_run_plan
from kura.run_envelope import SettingsRefused

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.backend_fixtures import with_required  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only  # noqa: E402


def _capabilities_text(backend: str) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert cmd_run_capabilities(argparse.Namespace(backend=backend, json=False)) == 0
    return output.getvalue()


def _capabilities_json(backend: str) -> dict[str, Any]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert cmd_run_capabilities(argparse.Namespace(backend=backend, json=True)) == 0
    return json.loads(output.getvalue())


SD_ROLE_PATHS = {
    "sd15": {"base": "/models/sd15.safetensors"},
    "sdxl": {"base": "/models/sdxl.safetensors"},
    "flux1": {"dit": "/models/flux.safetensors", "clip_l": "/models/clip.safetensors", "t5xxl": "/models/t5.safetensors", "ae": "/models/ae.safetensors"},
    "anima": {"dit": "/models/anima.safetensors", "qwen3": "/models/qwen3.safetensors", "vae": "/models/vae.safetensors"},
}


def sd_run(architecture: str = "sd15", mode: str = "lora", *, subset: dict[str, Any] | None = None, general: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "architecture": architecture, "mode": mode, "model_paths": dict(SD_ROLE_PATHS[architecture]),
        "dataset_config": {
            "general": {"resolution": [512, 512], "caption_extension": ".txt", **(general or {})},
            "datasets": [{"batch_size": 1, "subsets": [{"dataset_id": "tiny", **(subset if subset is not None else {"num_repeats": 1})}]}],
        },
        "learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "mixed_precision": "bf16",
    }
    if mode == "lora":
        config.update({"network_dim": 8, "network_alpha": 1})
    return {
        "id": "facts", "type": "train", "model": {"base": architecture},
        "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
        "backend": {"name": "sd-scripts", "config": with_required("sd-scripts", config)},
    }


class CapabilitiesShowAdapterFactsTests(unittest.TestCase):
    def test_every_backend_names_its_pinned_trainer(self) -> None:
        for name, adapter in BACKENDS.items():
            with self.subTest(backend=name):
                pinned = backend_capabilities(name)["pinned_trainer"]
                build_arg, value = BUILD_SOURCES[adapter.image_name]
                self.assertEqual(pinned, {"image": PINNED_IMAGES[adapter.image_name], "source": {build_arg: value}})
                self.assertIn(value, _capabilities_text(name))

    def test_ai_toolkit_shows_the_pinned_upstream_ui_baseline_of_every_model_arch(self) -> None:
        payload = _capabilities_json("ai-toolkit")
        facts = payload["model_facts"]
        baseline = load_baseline()
        self.assertEqual(facts["baseline"]["label"], "pinned upstream UI baseline")
        self.assertEqual(facts["baseline"]["commit"], baseline["upstream"]["commit"])
        by_arch = facts["baseline"]["model_arch"]
        self.assertEqual(set(by_arch), set(payload["config_value_choices"]["model_arch"]))
        moss = by_arch["moss_image:base"]
        entry = baseline["entries"]["moss_image:base"]
        self.assertEqual(moss["entry"], "moss_image:base")
        self.assertEqual(moss["name_or_path"], "OpenMOSS-Team/MOSS-Image-Base")
        self.assertEqual(moss["model"]["model_kwargs"], entry["model"]["model_kwargs"])
        self.assertEqual((moss["train"], moss["model"], moss["dataset"]), (entry["train"], entry["model"], entry["dataset"]))
        # The UI entry of another name that the arch selects (sd1 is the UI's sd15).
        self.assertEqual(by_arch["sd1"]["entry"], "sd15")
        # An arch with no entry, and one several entries train, say so instead of inventing values.
        self.assertIsNone(by_arch["sd2"])
        self.assertIn("wan21:14b", by_arch["wan21"]["error"])
        self.assertIn("authored values win", facts["native_config"])

        text = _capabilities_text("ai-toolkit")
        self.assertIn("pinned upstream UI baseline", text)
        self.assertIn(baseline["upstream"]["commit"], text)
        self.assertIn("moss_image:base (entry moss_image:base): name_or_path OpenMOSS-Team/MOSS-Image-Base", text)
        self.assertIn("native_config:", text)

    def test_sd_scripts_shows_required_and_optional_roles_of_each_entrypoint(self) -> None:
        facts = backend_capabilities("sd-scripts")["model_facts"]
        self.assertEqual(set(facts["model_roles"]), {f"architecture={a}, mode={m}" for a, m in ENTRYPOINTS})
        for (architecture, mode), required in ROLE_CONTRACTS.items():
            with self.subTest(architecture=architecture, mode=mode):
                roles = facts["model_roles"][f"architecture={architecture}, mode={mode}"]
                self.assertEqual(roles, {"required": list(required), "optional": list(OPTIONAL_ROLES[(architecture, mode)])})
        self.assertEqual(facts["model_roles"]["architecture=sdxl, mode=lora"]["optional"], ["vae"])
        self.assertEqual(facts["model_sources"]["model_downloads.<role>"]["keys"], list(MODEL_DOWNLOAD_KEYS))
        self.assertEqual(facts["model_sources"]["model_downloads.<role>"]["required"], ["repo", "filename"])
        self.assertIn("container", facts["model_sources"]["model_paths.<role>"])
        text = _capabilities_text("sd-scripts")
        self.assertIn("architecture=anima, mode=lora: required dit, qwen3, vae; optional llm_adapter, t5_tokenizer", text)
        self.assertIn("model_downloads.<role>: keys repo, filename, revision, repo_type (required: repo, filename)", text)

    def test_every_optional_sd_scripts_role_reaches_the_trainer(self) -> None:
        # The capabilities table and the command builder must agree: an optional role is forwarded.
        for (architecture, mode), optional in OPTIONAL_ROLES.items():
            for role in optional:
                with self.subTest(architecture=architecture, mode=mode, role=role):
                    run = sd_run(architecture, mode)
                    run["backend"]["config"]["model_paths"][role] = f"/models/{role}-optional"
                    self.assertIn(f"/models/{role}-optional", json.dumps(command_sd_scripts(run)["argv"]))
        self.assertEqual(set(OPTIONAL_ROLES), set(ROLE_CONTRACTS))

    def test_musubi_shows_the_roles_each_architecture_reads(self) -> None:
        payload = backend_capabilities("musubi-tuner")
        roles = payload["model_facts"]["model_roles"]
        for architecture in payload["config_value_choices"]["architecture"]:
            config: dict[str, Any] = {"architecture": architecture}
            if architecture == "flux2":
                for version in payload["config_value_choices"]["model_version"]:
                    if version in ("original", "layered", "edit", "edit-2509", "edit-2511"):
                        continue
                    run = {"backend": {"name": "musubi-tuner", "config": {**config, "model_version": version}}}
                    self.assertEqual(roles[f"architecture=flux2, model_version={version}"]["formats"], _musubi_model_expectations(run))
                continue
            with self.subTest(architecture=architecture):
                run = {"backend": {"name": "musubi-tuner", "config": config}}
                self.assertEqual(roles[f"architecture={architecture}"]["formats"], _musubi_model_expectations(run))
        self.assertIn("model_downloads.<role>", payload["model_facts"]["model_sources"])
        self.assertIn("architecture=wan: dit (safetensors)", _capabilities_text("musubi-tuner"))

    def test_sd_scripts_num_repeats_is_shown_as_required(self) -> None:
        nested = backend_capabilities("sd-scripts")["nested_config_fields"]
        for path in ("dataset_config.general", "dataset_config.datasets[]", "dataset_config.datasets[].subsets[]"):
            self.assertIn("subset", nested[path]["num_repeats"]["required"])
        self.assertIn("num_repeats (integer, min=1, required:", _capabilities_text("sd-scripts"))


class AiToolkitEffectiveSettingsTests(unittest.TestCase):
    def _run(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "facts-ai", "type": "train", "model": {"base": "OpenMOSS-Team/MOSS-Image-Base"},
            "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            "backend": {"name": "ai-toolkit", "config": config},
        }

    def test_display_marks_each_train_and_model_value_authored_or_baseline(self) -> None:
        run = self._run({
            "model_arch": "moss_image:base", "learning_rate": 5e-5, "low_vram": False,
            "native_config": {"train": {"cache_text_embeddings": True, "unload_text_encoder": True}, "model": {"model_kwargs": {"extra": 1}}},
        })
        settings = display_ai_toolkit(run)["effective_settings"]
        self.assertEqual(settings["entry"], "moss_image:base")
        self.assertEqual(settings["commit"], load_baseline()["upstream"]["commit"])
        train, model = settings["train"], settings["model"]
        self.assertEqual(train["lr"], {"value": 5e-5, "source": "authored"})
        self.assertEqual(train["cache_text_embeddings"], {"value": True, "source": "authored"})
        self.assertEqual(train["optimizer"], {"value": "adamw8bit", "source": "baseline"})
        self.assertEqual(model["low_vram"], {"value": False, "source": "authored"})
        self.assertEqual(model["qtype"], {"value": "convrot8", "source": "baseline"})
        self.assertEqual(model["model_kwargs.extra"], {"value": 1, "source": "authored"})
        self.assertEqual(model["model_kwargs.uncond_skip_layers"], {"value": [22], "source": "baseline"})
        self.assertNotIn("baseline", display_ai_toolkit(run))

    @posix_only(DATASET_IO)
    def test_displayed_values_are_the_compiled_ones(self) -> None:
        run = self._run({
            "model_arch": "moss_image:base", "learning_rate": 5e-5, "model_edit": False,
            "native_config": {"train": {"cache_text_embeddings": True}},
        })
        settings = display_ai_toolkit(run)["effective_settings"]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            freeze_fixture(run, destination)
            BACKENDS["ai-toolkit"].compile(run, destination)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
        for section in ("train", "model"):
            for key, row in settings[section].items():
                with self.subTest(section=section, key=key):
                    value: Any = process[section]
                    for part in key.split("."):
                        value = value[part]
                    self.assertEqual(value, row["value"])

    def test_plan_renders_the_effective_settings_block(self) -> None:
        run = self._run({"model_arch": "moss_image:base", "native_config": {"train": {"cache_text_embeddings": True}}})
        display = display_ai_toolkit(run)
        payload = {"resources": {"training": {key: value for key, value in display.items() if key not in {"checkpoint", "memory"}}}}
        from kura.run_commands.plan import _resources_payload  # noqa: PLC0415

        resources = _resources_payload(run, {}, {}, adapter_display=display)
        self.assertNotIn("effective_settings", resources["training"])
        payload = {"resources": resources}
        text = format_run_plan(payload)
        self.assertIn("trainer settings (authored, or the pinned upstream UI baseline", text)
        self.assertIn("train.cache_text_embeddings  true  authored", text)
        self.assertIn("model.qtype  convrot8  baseline", text)

    def test_an_arch_with_no_baseline_entry_reports_why(self) -> None:
        settings = display_ai_toolkit(self._run({"model_arch": "sd2"}))["effective_settings"]
        self.assertIn("noise scheduler", settings["error"])


class SdScriptsNumRepeatsTests(unittest.TestCase):
    def test_general_or_dataset_num_repeats_satisfies_the_requirement(self) -> None:
        for where in ("general", "dataset"):
            with self.subTest(where=where):
                run = sd_run(subset={}, general={"num_repeats": 3} if where == "general" else None)
                if where == "dataset":
                    run["backend"]["config"]["dataset_config"]["datasets"][0]["num_repeats"] = 3
                validate_backend_config(run)

    def test_no_num_repeats_anywhere_is_refused_naming_every_place(self) -> None:
        with self.assertRaisesRegex(ValueError, r"subsets\[0\] has no num_repeats; write it on the subset, its dataset block, or dataset_config.general"):
            validate_backend_config(sd_run(subset={}))

    @posix_only(DATASET_IO)
    def test_inherited_num_repeats_is_the_projected_repeat(self) -> None:
        run = sd_run(subset={}, general={"num_repeats": 3})
        with tempfile.TemporaryDirectory() as directory:
            resolved = Path(directory) / "runs" / "facts" / "resolved"
            resolved.mkdir(parents=True)
            report = freeze_fixture(run, resolved)
            from kura.backends.sd_scripts import compile_sd_scripts  # noqa: PLC0415

            compile_sd_scripts(run, resolved / "sd-scripts")
            toml = tomllib.loads((resolved / "sd-scripts" / "dataset.toml").read_text(encoding="utf-8"))
        view = report["datasets"][0]["views"][0]
        self.assertEqual((view["repeat"], view["repeat_pointer"]), (3, "/general/num_repeats"))
        self.assertEqual(toml["general"]["num_repeats"], 3)
        self.assertNotIn("num_repeats", toml["datasets"][0]["subsets"][0])

    def test_plan_display_shows_only_the_written_num_repeats(self) -> None:
        inherited = display_sd_scripts(sd_run(subset={}, general={"num_repeats": 4}))
        self.assertEqual(inherited["dataset_config"]["datasets"][0]["subsets"][0]["num_repeats"], 4)
        unset = display_sd_scripts(sd_run(subset={}))
        self.assertNotIn("num_repeats", unset["dataset_config"]["datasets"][0]["subsets"][0])


class SdScriptsProblemsTogetherTests(unittest.TestCase):
    def test_dataset_selector_and_role_problems_are_one_refusal(self) -> None:
        run = sd_run("anima", "lora", subset={})
        config = run["backend"]["config"]
        config["cache_text_encoder_outputs"] = True
        del config["model_paths"]["qwen3"]
        with self.assertRaises(SettingsRefused) as caught:
            validate_backend_config(run)
        message = str(caught.exception)
        self.assertIn("has no num_repeats", message)
        self.assertIn("text-encoder caching requires network_train_unet_only=true", message)
        self.assertIn("missing required role(s): qwen3", message)

    def test_one_problem_is_reported_alone(self) -> None:
        run = sd_run("sdxl")
        del run["backend"]["config"]["model_paths"]["base"]
        with self.assertRaisesRegex(SettingsRefused, r"^sd-scripts model_paths/model_downloads missing required role\(s\): base$"):
            validate_backend_config(run)

    def test_an_invalid_selector_still_stops_first(self) -> None:
        run = sd_run("sd15", subset={})
        run["backend"]["config"]["mode"] = "controlnet_lllite"
        for key in ("network_dim", "network_alpha"):
            run["backend"]["config"].pop(key)
        with self.assertRaises(ValueError) as caught:
            validate_backend_config(run)
        self.assertIn("unsupported Kura built-in sd-scripts adapter", str(caught.exception))
        self.assertNotIn("num_repeats", str(caught.exception))
        missing = deepcopy(sd_run(subset={}))
        del missing["backend"]["config"]["architecture"]
        with self.assertRaises(ValueError) as caught:
            validate_backend_config(missing)
        self.assertNotIn("num_repeats", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
