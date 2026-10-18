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
from unittest.mock import patch

import yaml

from kura.backends import BACKENDS, backend_capabilities, validate_backend_config
from kura.backends.ai_toolkit import display_ai_toolkit
from kura.backends.ai_toolkit_baseline import load_baseline
from kura.backends.musubi_models import _musubi_model_expectations
from kura.backends.sd_scripts import ENTRYPOINTS, command_sd_scripts, display_sd_scripts
from kura.backends.sd_scripts_models import OPTIONAL_ROLES, ROLE_CONTRACTS
from kura.backends.musubi_command import command_musubi_tuner
from kura.backends.shared import MODEL_DOWNLOAD_KEYS, family_card_hint
from kura.backends.ai_toolkit import AI_TOOLKIT_OPTIMIZER_DECIDED
from kura.cli import cmd_run_capabilities, main as cli_main
from kura.training_artifacts import LR_SCHEDULER_DECIDED, RESUME_LR_SCHEDULER
from kura.images import BUILD_SOURCES, PINNED_IMAGES
from kura.run_commands.plan import format_run_plan
from kura.run_envelope import SettingsRefused

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.backend_fixtures import with_required  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only  # noqa: E402


def _capabilities_text(backend: str, arch: str | None = None) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert cmd_run_capabilities(argparse.Namespace(backend=backend, json=False, arch=arch)) == 0
    return output.getvalue()


def _capabilities_json(backend: str, arch: str | None = None) -> dict[str, Any]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert cmd_run_capabilities(argparse.Namespace(backend=backend, json=True, arch=arch)) == 0
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

    def test_ai_toolkit_shows_the_pinned_upstream_ui_baseline_of_one_model_arch_on_request(self) -> None:
        baseline = load_baseline()
        choices = backend_capabilities("ai-toolkit")["config_value_choices"]["model_arch"]
        by_arch = {}
        for value in choices:
            facts = backend_capabilities("ai-toolkit", arch=value)["model_facts"]
            self.assertEqual(set(facts["baseline"]["model_arch"]), {value})
            by_arch[value] = facts["baseline"]["model_arch"][value]
        facts = _capabilities_json("ai-toolkit", "moss_image:base")["model_facts"]
        self.assertEqual(facts["baseline"]["label"], "pinned upstream UI baseline")
        self.assertEqual(facts["baseline"]["commit"], baseline["upstream"]["commit"])
        moss = facts["baseline"]["model_arch"]["moss_image:base"]
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

        text = _capabilities_text("ai-toolkit", "moss_image:base")
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


class CapabilitiesSurfaceAndArchTests(unittest.TestCase):
    """The default output is the surface; --arch adds one architecture's facts (text and JSON alike)."""

    def test_the_default_output_is_the_surface_without_per_arch_baselines(self) -> None:
        for name in BACKENDS:
            with self.subTest(backend=name):
                text = _capabilities_text(name)
                self.assertLess(len(text.encode("utf-8")), 16_000)
                self.assertNotIn("(entry ", text)
                payload = _capabilities_json(name)
                self.assertNotIn("model_arch", payload["model_facts"].get("baseline", {}))
                for field in payload["config_fields"]:
                    self.assertIn(field, text)
        text = _capabilities_text("ai-toolkit")
        self.assertIn("--arch <value>", text)
        self.assertIn("moss_image:base", text)  # still listed among the selector values

    def test_arch_filters_roles_and_conditional_fields_to_that_architecture(self) -> None:
        payload = _capabilities_json("musubi-tuner", "zimage")
        self.assertEqual(set(payload["model_facts"]["model_roles"]), {"architecture=zimage"})
        conditional = payload["conditional_fields"]
        self.assertIn("fp8_llm", conditional)
        self.assertEqual(conditional["fp8_llm"]["when_any"], [{"architecture": ["zimage", "framepack"]}])
        self.assertIn("text_encoder_batch_size", conditional)  # no architecture clause: applies to every one
        self.assertNotIn("convrot_int8", conditional)
        self.assertNotIn("one_frame", conditional)
        self.assertIn("discrete_flow_shift", payload["required_fields"])
        self.assertNotIn("h3_guidance_loss_scale", payload["required_fields"])
        text = _capabilities_text("musubi-tuner", "zimage")
        self.assertIn("architecture=zimage: dit (safetensors)", text)
        self.assertNotIn("architecture=wan:", text)
        self.assertNotIn("convrot_int8:", text)

        sd = _capabilities_json("sd-scripts", "anima")
        self.assertEqual(set(sd["model_facts"]["model_roles"]), {"architecture=anima, mode=lora", "architecture=anima, mode=controlnet_lllite"})
        self.assertNotIn("guidance_scale", sd["required_fields"])
        self.assertIn("discrete_flow_shift", sd["required_fields"])

    def test_text_and_json_carry_the_same_filtered_facts(self) -> None:
        for name, arch in (("ai-toolkit", "zimage:turbo"), ("musubi-tuner", "flux2"), ("sd-scripts", "sdxl")):
            with self.subTest(backend=name):
                payload, text = _capabilities_json(name, arch), _capabilities_text(name, arch)
                for field in payload["conditional_fields"]:
                    self.assertIn(f"  {field}: ", text)
                for key in payload["model_facts"].get("model_roles", {}):
                    self.assertIn(f"  {key}: ", text)
                for value in payload["model_facts"].get("baseline", {}).get("model_arch", {}):
                    self.assertIn(f"  {value}", text)

    def test_an_unknown_arch_is_refused_naming_the_values(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = cmd_run_capabilities(argparse.Namespace(backend="musubi-tuner", json=False, arch="z-image"))
        self.assertEqual(code, 1)
        self.assertIn("z-image", stderr.getvalue())
        self.assertIn("zimage", stderr.getvalue())
        self.assertIn("qwen_image", stderr.getvalue())
        with self.assertRaisesRegex(ValueError, "sdxl"):
            backend_capabilities("sd-scripts", arch="sd1")

    def test_decided_fields_without_a_list_say_why_from_their_owner(self) -> None:
        for name in BACKENDS:
            with self.subTest(backend=name):
                payload = _capabilities_json(name)
                self.assertEqual(payload["decided_notes"]["lr_scheduler"], LR_SCHEDULER_DECIDED)
                self.assertIn(f"  lr_scheduler: {LR_SCHEDULER_DECIDED}", _capabilities_text(name))
        self.assertIn(RESUME_LR_SCHEDULER, LR_SCHEDULER_DECIDED)
        ai = _capabilities_json("ai-toolkit")
        self.assertEqual(ai["decided_notes"]["optimizer_type"], AI_TOOLKIT_OPTIMIZER_DECIDED)
        self.assertNotIn("exact values only", _capabilities_text("ai-toolkit").split("lr_scheduler:")[-1].splitlines()[0])


class MissingRoleRefusalNamesTheFamilyCardTests(unittest.TestCase):
    """Kura keeps no default-weights table; a missing role points at the family card, which may list
    the files a recorded run trained from."""

    def test_sd_scripts_names_the_card_of_its_architecture(self) -> None:
        run = sd_run("anima")
        del run["backend"]["config"]["model_paths"]["qwen3"]
        with self.assertRaises(SettingsRefused) as caught:
            validate_backend_config(run)
        self.assertIn("missing required role(s): qwen3; " + family_card_hint("anima"), str(caught.exception))

    def test_musubi_names_the_card_of_its_architecture(self) -> None:
        def musubi(config: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": "facts-mu", "type": "train", "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
                "backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", config)},
            }

        cases = (
            ({"architecture": "zimage", "model_paths": {"dit": "/models/dit.safetensors"}}, "zimage", "vae, text_encoder"),
            ({"architecture": "zimage"}, "zimage", "model_paths, model_downloads, or model_bundle"),
            (
                {"architecture": "wan", "task": "i2v-14B", "model_paths": {"dit": "/m/d.safetensors", "vae": "/m/v.safetensors", "t5": "/m/t5.pth"}},
                "wan", "model_downloads.clip",
            ),
        )
        for config, architecture, missing in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    command_musubi_tuner(musubi(config))
                message = str(caught.exception)
                self.assertIn(missing, message)
                self.assertIn(family_card_hint(architecture), message)
        self.assertEqual(
            family_card_hint("zimage"),
            "the family card .kura/knowledge/model-families/zimage.md lists verified entries when one exists",
        )

    def test_a_missing_model_source_names_the_bundles_kura_knows_for_the_architecture(self) -> None:
        def musubi(config: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": "facts-mu", "type": "train", "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
                "backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", config)},
            }

        with self.assertRaises(ValueError) as caught:
            command_musubi_tuner(musubi({"architecture": "flux2", "model_version": "klein-base-4b", "timestep_sampling": "flux2_shift"}))
        message = str(caught.exception)
        self.assertIn("model_bundle flux2-klein-base-4b downloads this model_version's files", message)
        self.assertIn("flux2-klein-4b, flux2-klein-base-9b, flux2-klein-9b", message)
        with self.assertRaises(ValueError) as caught:
            command_musubi_tuner(musubi({"architecture": "zimage"}))
        self.assertNotIn("model_bundle flux", str(caught.exception))

    def test_capabilities_show_model_bundle_as_a_model_source(self) -> None:
        text = _capabilities_text("musubi-tuner", "flux2")
        self.assertIn("model roles, each named in backend.config.model_paths or model_downloads, or downloaded by a model_bundle:", text)
        self.assertIn(
            "  model_bundle: one name that downloads every role's files, instead of model_downloads; "
            "architecture=flux2: flux2-klein-base-4b, flux2-klein-4b, flux2-klein-base-9b, flux2-klein-9b",
            text,
        )
        self.assertNotIn("krea2-raw", text.split("model entry shapes:")[1])
        self.assertNotIn("model_bundle", _capabilities_text("sd-scripts").split("model entry shapes:")[1])
        self.assertNotIn("model_bundle", _capabilities_text("musubi-tuner", "zimage").split("model entry shapes:")[1])

    def test_capabilities_say_where_verified_entries_are(self) -> None:
        for name in ("sd-scripts", "musubi-tuner"):
            with self.subTest(backend=name):
                self.assertIn(family_card_hint("<architecture>"), _capabilities_text(name))


class ClosedPipeTests(unittest.TestCase):
    def test_a_closed_pipe_ends_any_command_quietly(self) -> None:
        class ClosedPipe(io.StringIO):
            def write(self, text: str) -> int:
                raise BrokenPipeError

            def fileno(self) -> int:
                return 1

        with patch.object(sys, "argv", ["kura", "run", "capabilities", "ai-toolkit"]), \
                patch("kura.cli._load_secrets"), patch("kura.cli._refresh_managed_files"), \
                patch("kura.cli.os.dup2") as dup2, patch.object(sys, "stdout", ClosedPipe()):
            with self.assertRaises(SystemExit) as stopped:
                cli_main()
        self.assertEqual(stopped.exception.code, 0)
        dup2.assert_called_once()


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
        from kura.run_commands.plan import _plan_requirements, _resources_payload  # noqa: PLC0415

        resources = _resources_payload(run, {}, _plan_requirements(run, {}), adapter_display=display)
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
        with self.assertRaisesRegex(
            SettingsRefused,
            r"^sd-scripts model_paths/model_downloads missing required role\(s\): base; the family card "
            r"\.kura/knowledge/model-families/sdxl\.md lists verified entries when one exists$",
        ):
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


class PlanTrainingRowsTests(unittest.TestCase):
    """`kura run plan` training rows: a value that does not apply is a placeholder or no row."""

    def _training(self, backend: str, config: dict[str, Any]) -> dict[str, Any]:
        from kura.run_commands.plan import _plan_requirements, _resources_payload  # noqa: PLC0415

        run = {
            "id": "plan-rows", "type": "train", "model": {"base": "custom"},
            "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            "backend": {"name": backend, "config": config},
        }
        with patch("kura.run_commands.plan._local_gpu_payload", return_value={"name": "unknown", "vram_total_mb": "unknown"}):
            return _resources_payload(run, {}, _plan_requirements(run, {}))["training"]

    def test_empty_container_is_not_set(self) -> None:
        from kura.run_commands.plan import NOT_SET, _plan_value  # noqa: PLC0415

        for value in ({}, [], None, ""):
            with self.subTest(value=value):
                self.assertEqual(_plan_value(value), NOT_SET)
        for value in (0, False, {"a": 1}, ["x"]):
            with self.subTest(value=value):
                self.assertEqual(_plan_value(value), value)

    def test_ai_toolkit_plan_has_no_dataset_row(self) -> None:
        training = self._training("ai-toolkit", {"model_arch": "moss_image:base"})
        self.assertNotIn("dataset", training)

    def test_musubi_plan_shows_selector_only_where_a_task_applies(self) -> None:
        training = self._training("musubi-tuner", {"architecture": "flux_kontext"})
        self.assertNotIn("selector", training)
        self.assertNotIn("precision", training)
        self.assertEqual(training["save_precision"], "(not set)")
        training = self._training("musubi-tuner", {"architecture": "flux_kontext", "save_precision": "fp16"})
        self.assertEqual(training["save_precision"], "fp16")
        training = self._training("musubi-tuner", {"architecture": "wan", "task": "t2v-14B"})
        self.assertEqual(training["selector"], "t2v-14B")
        self.assertEqual(self._training("musubi-tuner", {"architecture": "wan"})["selector"], "(not set)")
