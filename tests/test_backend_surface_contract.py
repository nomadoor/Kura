"""Registry-wide proof for authored backend configuration surfaces."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from kura.backends import BACKENDS, BackendAdapter, backend_capabilities, validate_backend_config
from kura.backends.ai_toolkit import AI_TOOLKIT_VIDEO_SUFFIXES
from kura.backends.musubi_command import _script_command as musubi_script_command
from kura.cli import cmd_run_capabilities, cmd_run_compile, cmd_run_plan
from kura.init_templates import cmd_init
from kura.media_types import frozen_suffixes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only
from tests.backend_fixtures import with_required


class BackendSurfaceContractTests(unittest.TestCase):
    def _write_ai_toolkit_projection(
        self,
        run: dict[str, object],
        resolved: Path,
    ) -> None:
        self._write_ai_toolkit_manifest_projection(
            run, resolved, target="sample.png",
        )

    def _write_ai_toolkit_manifest_projection(
        self,
        run: dict[str, object],
        resolved: Path,
        *,
        target: str,
        controls: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return freeze_fixture(run, resolved, target=target, controls=controls)

    def _write_musubi_projection(self, run: dict[str, object], resolved: Path) -> None:
        freeze_fixture(run, resolved)

    def _write_sd_scripts_projection(self, run: dict[str, object], resolved: Path) -> None:
        freeze_fixture(run, resolved)

    def test_boolean_fields_take_only_true_or_false_on_every_backend(self) -> None:
        # A quoted "false" used to switch Musubi's precache on while the registry read it as off.
        cases = (
            ("musubi-tuner", {"architecture": "hidream_o1", "precache": "false"}, "precache='false'"),
            ("musubi-tuner", {"architecture": "framepack", "one_frame": "true"}, "one_frame='true'"),
            ("musubi-tuner", {"architecture": "krea2", "gradient_checkpointing": "true"}, "gradient_checkpointing='true'"),
            ("sd-scripts", {"architecture": "sdxl", "gradient_checkpointing": 1}, "gradient_checkpointing=1"),
            # Written into AI-Toolkit's YAML as the string "false", which its Python reads as on.
            ("ai-toolkit", {"model_arch": "sdxl", "low_vram": "false"}, "low_vram='false'"),
        )
        for backend, config, named in cases:
            with self.subTest(backend=backend, config=config):
                with self.assertRaisesRegex(ValueError, f"{backend} backend.config field.* must be true or false: {re.escape(named)}"):
                    validate_backend_config({"backend": {"name": backend, "config": config}})
        validate_backend_config({"backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", {"architecture": "hidream_o1", "precache": False})}})
        for name, adapter in BACKENDS.items():
            with self.subTest(declared=name):
                self.assertLessEqual(adapter.surface.boolean_fields, adapter.surface.fields)
        self.assertIn("precache", BACKENDS["musubi-tuner"].surface.boolean_fields)
        # Agents learn this from the capabilities output before they write the value.
        self.assertIn("precache", backend_capabilities("musubi-tuner")["boolean_fields"])

    def test_every_field_read_as_a_boolean_is_declared_boolean(self) -> None:
        # A field read with _truthy or _append_flag must be refused by the shared validator
        # when it is not true or false, or a quoted value would be read as off.
        import ast
        import kura.backends as backends

        root = Path(backends.__file__).parent
        files = {"musubi-tuner": ("musubi_command.py", "musubi_datasets.py", "musubi_models.py"), "sd-scripts": ("sd_scripts.py",)}
        for backend, names in files.items():
            surface = BACKENDS[backend].surface
            read: set[str] = set()
            for name in names:
                for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    if node.func.id == "_append_flag" and len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                        read.add(node.args[2].value)
                    elif node.func.id == "_truthy" and node.args and isinstance(node.args[0], ast.Call):
                        inner = node.args[0]
                        if isinstance(inner.func, ast.Attribute) and inner.func.attr == "get" and inner.args and isinstance(inner.args[0], ast.Constant):
                            read.add(inner.args[0].value)
            with self.subTest(backend=backend):
                self.assertTrue(read)
                self.assertEqual(sorted((read & surface.fields) - surface.boolean_fields), [])

    def _model_download_runs(self, downloads: dict, model_paths: dict | None = None) -> dict:
        """The same download declaration for each backend that reads model_downloads, under each one's role name."""
        runs = {}
        for backend, role, config in (
            ("musubi-tuner", "dit", {"architecture": "flux2", "model_version": "dev"}),
            ("sd-scripts", "base", {"architecture": "sdxl", "mode": "lora"}),
        ):
            native = {**config, "model_downloads": {role: deepcopy(downloads)}}
            if model_paths is not None:
                native["model_paths"] = {role: model_paths}
            runs[backend] = (role, {"id": "r", "type": "train", "backend": {"name": backend, "config": native}})
        return runs

    def test_model_downloads_are_read_one_way_for_every_backend(self) -> None:
        from kura.backends.musubi_models import _musubi_model_lock
        from kura.backends.sd_scripts_models import sd_scripts_model_lock

        locks = {"musubi-tuner": _musubi_model_lock, "sd-scripts": sd_scripts_model_lock}
        listed = {"repo": "org/model", "filename": ["a.safetensors", "b.safetensors"]}
        for backend, (role, run) in self._model_download_runs(listed).items():
            with self.subTest(backend=backend, case="a list of files"):
                specs, paths = BACKENDS[backend].download_specs(run)
                # Every listed file is fetched; the trainer is pointed at the first.
                self.assertEqual([item["filename"] for item in specs if item["key"] == role], ["a.safetensors", "b.safetensors"])
                self.assertTrue(paths[role].endswith("/a.safetensors"))
                recorded = next(item for item in locks[backend](run)["models"] if item["role"] == role)
                self.assertEqual((recorded["repo_id"], recorded["filename"], recorded["path"]), ("org/model", "a.safetensors", paths[role]))
        for key in ("filname", "local_dir"):
            for backend, (role, run) in self._model_download_runs({"repo": "org/model", "filename": "a.safetensors", key: "x"}).items():
                with self.subTest(backend=backend, case=key):
                    with self.assertRaisesRegex(ValueError, f"model_downloads.{role} contains unsupported key.*{key}"):
                        BACKENDS[backend].download_specs(run)
        for key, accepted in (("repo_id", "repo"), ("file", "filename"), ("filenames", "filename \\(a list\\)")):
            declaration = {"repo": "org/model", "filename": "a.safetensors", key: "x"}
            for backend, (role, run) in self._model_download_runs(declaration).items():
                with self.subTest(backend=backend, case=key):
                    with self.assertRaisesRegex(ValueError, f"model_downloads.{role} uses {key} \\(write {accepted}\\)"):
                        BACKENDS[backend].download_specs(run)

    def test_the_plan_counts_every_model_the_adapter_fetches(self) -> None:
        from kura.run_commands.plan import _estimate_backend_download_bytes

        # A Musubi bundle is named in model_bundle, with no model.base.
        run = {"id": "r", "type": "train", "backend": {"name": "musubi-tuner", "config": {"architecture": "krea2", "model_bundle": "krea2-raw"}}}
        specs, _ = BACKENDS["musubi-tuner"].download_specs(run)
        with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 7}):
            estimate = _estimate_backend_download_bytes(run)
        self.assertTrue(specs)
        self.assertEqual([(item["key"], item["filename"]) for item in estimate["items"]], [(item["key"], item["filename"]) for item in specs])

    def test_both_backends_read_model_downloads_through_the_owner(self) -> None:
        from kura.backends import shared

        for backend, (role, run) in self._model_download_runs({"repo": "org/model", "filename": "a.safetensors"}).items():
            module = "kura.backends.musubi_models" if backend == "musubi-tuner" else "kura.backends.sd_scripts_models"
            with self.subTest(backend=backend), patch(f"{module}.model_downloads", wraps=shared.model_downloads) as owner:
                BACKENDS[backend].download_specs(run)
            owner.assert_called()

    def test_a_malformed_download_value_is_refused_not_dropped(self) -> None:
        for declaration in (
            {"repo": "org/model", "filename": ["a.safetensors", ""]},
            {"repo": "org/model", "filename": []},
            {"repo": "org/model", "filename": ""},
            {"repo": "org/model", "filename": "a.safetensors", "revision": 3},
            {"filename": "a.safetensors"},
        ):
            for backend, (role, run) in self._model_download_runs(declaration).items():
                with self.subTest(backend=backend, declaration=declaration), self.assertRaisesRegex(ValueError, f"model_downloads.{role} requires repo and filename"):
                    BACKENDS[backend].download_specs(run)

    def test_a_download_mapping_is_checked_even_when_model_paths_names_the_role(self) -> None:
        for backend, (role, run) in self._model_download_runs({"repo": "org/model", "filname": "a.safetensors"}, model_paths="/workspace/models/a.safetensors").items():
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "unsupported key.*filname"):
                BACKENDS[backend].download_specs(run)
        with self.assertRaisesRegex(ValueError, "model_paths must map model roles to file paths"):
            BACKENDS["musubi-tuner"].download_specs({"backend": {"name": "musubi-tuner", "config": {"architecture": "krea2", "model_paths": ["x"]}}})

    def test_a_role_named_in_both_model_downloads_and_model_paths_is_refused(self) -> None:
        # One of the two would be dropped without a word; the user keeps the one they mean.
        for backend, (role, run) in self._model_download_runs({"repo": "org/model", "filename": "a.safetensors"}, model_paths="/workspace/models/a.safetensors").items():
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError, f"model_downloads.{role} and model_paths.{role} both name the {role} file; keep model_paths.{role}.*or model_downloads.{role}",
            ):
                BACKENDS[backend].download_specs(run)

    def test_the_launch_preflight_names_the_doubled_role_of_a_compiled_run(self) -> None:
        from kura.run_commands.plan import collect_run_preflight, enforce_preflight_errors

        for backend, (role, run) in self._model_download_runs({"repo": "org/model", "filename": "a.safetensors"}, model_paths="/workspace/models/a.safetensors").items():
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                run["safety"] = {"allow_large_model_downloads": True}
                with self.assertRaisesRegex(ValueError, f"model-downloads: .*model_downloads.{role} and model_paths.{role} both name"):
                    enforce_preflight_errors(collect_run_preflight(run, Path(directory), config={}, executor="docker"))

    def test_a_bundle_role_may_still_be_pointed_at_your_own_file(self) -> None:
        run = {"id": "r", "type": "train", "backend": {"name": "musubi-tuner", "config": {
            "architecture": "krea2", "model_bundle": "krea2-raw", "model_paths": {"vae": "/workspace/models/vae.safetensors"},
        }}}
        specs, paths = BACKENDS["musubi-tuner"].download_specs(run)
        self.assertNotIn("vae", {item["key"] for item in specs})
        self.assertTrue(specs)

    def test_an_empty_model_path_is_refused_not_skipped(self) -> None:
        for backend, (role, run) in self._model_download_runs({"repo": "org/model", "filename": "a.safetensors"}, model_paths="").items():
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "model_paths must map model roles to file paths"):
                BACKENDS[backend].download_specs(run)

    def test_every_registered_backend_rejects_unknown_top_level_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, adapter in BACKENDS.items():
                with self.subTest(backend=name):
                    run = {"backend": {"name": name, "config": {"kura_unknown_sentinel": True}}}
                    with self.assertRaisesRegex(ValueError, "kura_unknown_sentinel"):
                        adapter.compile(run, Path(directory) / name)

    def test_musubi_and_sd_scripts_reject_legacy_minimal_projection_locks(self) -> None:
        runs = {
            "musubi-tuner": {
                "id": "legacy-musubi",
                "backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", {
                    "architecture": "flux2", "model_version": "klein-base-4b",
                    "model_paths": {
                        "dit": "/models/dit", "vae": "/models/vae",
                        "text_encoder": "/models/text",
                    },
                })},
                "model": {"base": "example/model"},
                "datasets": [{"id": "tiny"}],
                "recipe": {"steps": 1, "seed": 1},
            },
            "sd-scripts": {
                "id": "legacy-sd",
                "backend": {"name": "sd-scripts", "config": with_required("sd-scripts", {
                    "architecture": "sd15", "model_paths": {"base": "/models/base"},
                    "dataset_config": {"datasets": [{"subsets": [{
                        "dataset_id": "tiny", "num_repeats": 1,
                    }]}]},
                })},
                "model": {"base": "/models/base"},
                "datasets": [{"id": "tiny"}],
                "recipe": {"steps": 1, "seed": 1},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend, run in runs.items():
                with self.subTest(backend=backend):
                    resolved = root / backend
                    resolved.mkdir()
                    (resolved / "dataset-projection.lock.json").write_text(json.dumps({
                        "schema_version": 1,
                        "backend": backend,
                        "datasets": [{"id": "tiny", "native": {}, "views": [{}]}],
                    }), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "not written by the verified dataset handoff"):
                        BACKENDS[backend].compile(run, resolved)

    def test_plausible_general_ml_substitutions_name_the_fix(self) -> None:
        for name in BACKENDS:
            substitutions = [
                ("optimizer", "optimizer_type"), ("scheduler", "lr_scheduler"),
                ("lr", "learning_rate"), ("rank", "network_dim"),
            ]
            if "batch_size" in BACKENDS[name].surface.fields:
                substitutions.append(("batch", "batch_size"))
            for wrong, expected in substitutions:
                with self.subTest(backend=name, wrong=wrong):
                    run = {"backend": {"name": name, "config": {wrong: "example"}}}
                    with self.assertRaisesRegex(ValueError, rf"{wrong}.*{expected}"):
                        validate_backend_config(run)

    def test_semantic_messages_do_not_confuse_training_and_save_precision(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {"mixed_precision": "fp16"}}}
        with self.assertRaisesRegex(ValueError, "training precision is fixed to bf16") as raised:
            validate_backend_config(run)
        self.assertNotIn("use 'save_precision'", str(raised.exception))

    def test_sd_scripts_known_unsupported_fields_explain_the_supported_path(self) -> None:
        cases = (
            ("deepspeed", True, "reviewed backend.config.command"),
            ("fused_backward_pass", True, "reviewed backend.config.command"),
            ("batch", 2, "dataset_config.datasets[].batch_size"),
        )
        capabilities = backend_capabilities("sd-scripts")
        for field, value, expected in cases:
            with self.subTest(field=field):
                self.assertIn(expected, capabilities["unsupported_fields"][field])
                run = {"backend": {"name": "sd-scripts", "config": {field: value}}}
                with self.assertRaisesRegex(ValueError, re.escape(expected)):
                    validate_backend_config(run)

    def test_epoch_vocabulary_points_to_the_common_recipe(self) -> None:
        for name in BACKENDS:
            with self.subTest(backend=name):
                run = {"backend": {"name": name, "config": {"epochs": 10}}}
                with self.assertRaisesRegex(ValueError, "recipe.steps"):
                    validate_backend_config(run)

    def test_real_cli_compile_rejects_plausible_wrong_key_for_every_backend(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            os.chdir(root)
            try:
                for name in BACKENDS:
                    with self.subTest(backend=name):
                        run_id = "wrong-" + name
                        run_dir = root / "runs" / run_id
                        run_dir.mkdir(parents=True)
                        run = {
                            "schema_version": 2, "id": run_id, "type": "train",
                            "backend": {"name": name, "config": {"optimizer": "adamw8bit"}},
                            "model": {"base": "example/model"}, "recipe": {"steps": 1, "seed": 1},
                        }
                        (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
                        (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
                        error = io.StringIO()
                        with contextlib.redirect_stderr(error):
                            self.assertEqual(cmd_run_compile(argparse.Namespace(run_id=run_id)), 1)
                        self.assertIn("optimizer_type", error.getvalue())
                        self.assertIn(f"kura run capabilities {name}", error.getvalue())
            finally:
                os.chdir(previous)

    def test_run_plan_rejects_surface_errors_before_user_approval(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "invalid-plan"
            run_dir.mkdir(parents=True)
            run = {
                "schema_version": 2, "id": "invalid-plan", "type": "train",
                "backend": {"name": "sd-scripts", "config": {
                    "architecture": "sdxl", "mode": "lora", "deepspeed": True, "zzz_unknown": 1,
                }},
                "model": {"base": "example/model"}, "recipe": {"steps": 1, "seed": 1},
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            os.chdir(root)
            try:
                output, error = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="invalid-plan", json=False)), 1)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("cannot show run plan", error.getvalue())
                self.assertIn("deepspeed", error.getvalue())
                self.assertIn("zzz_unknown", error.getvalue())
            finally:
                os.chdir(previous)

    def test_run_plan_reads_a_compiled_run_that_holds_a_retired_backend_key(self) -> None:
        # An older Kura compiled Musubi FLUX.2 runs with allow_a40_*; plan reads that manifest as it
        # is, while a draft that still holds the key is refused.
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run = {
                "schema_version": 2, "id": "old-run", "type": "train",
                "backend": {"name": "musubi-tuner", "config": {"architecture": "wan", "allow_a40_uncheckpointed_9b": True}},
                "model": {"base": "example/model"}, "recipe": {"steps": 1, "seed": 1},
            }
            run_dir = root / "runs" / "old-run"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            os.chdir(root)
            try:
                results, codes = {}, {}
                for compiled in (False, True):
                    if compiled:
                        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
                    error = io.StringIO()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                        codes[compiled] = cmd_run_plan(argparse.Namespace(run_id="old-run", json=True))
                    results[compiled] = error.getvalue()
            finally:
                os.chdir(previous)
        self.assertEqual(codes, {False: 1, True: 0})
        self.assertIn("allow_a40_uncheckpointed_9b", results[False])
        self.assertEqual(results[True], "")

    def test_contract_is_required_by_the_registry_type(self) -> None:
        adapter = BACKENDS["ai-toolkit"]
        with self.assertRaises(TypeError):
            BackendAdapter(
                name="undeclared",
                image_name=adapter.image_name,
                compile=adapter.compile,
                command=adapter.command,
                display=adapter.display,
                requirements=adapter.requirements,
            )

    def test_capabilities_are_derived_from_the_registered_contract(self) -> None:
        for name, adapter in BACKENDS.items():
            with self.subTest(backend=name):
                payload = backend_capabilities(name)
                self.assertEqual(payload["backend"], name)
                conditional = set(payload["conditional_fields"])
                self.assertEqual(set(payload["config_fields"]) | conditional, set(adapter.surface.fields))
                self.assertFalse(set(payload["config_fields"]) & conditional)
                self.assertEqual(set(payload["escape_hatches"]), set(adapter.surface.escape_hatches))

    def test_capabilities_expose_selector_applicability(self) -> None:
        musubi = backend_capabilities("musubi-tuner")
        self.assertEqual(
            musubi["conditional_fields"]["timestep_boundary"]["when_any"],
            [{"architecture": ["wan"]}],
        )
        self.assertNotIn("selector_aliases", musubi)
        self.assertIn("flux2", musubi["config_value_choices"]["architecture"])
        sd_scripts = backend_capabilities("sd-scripts")
        self.assertEqual(
            sd_scripts["conditional_fields"]["cond_emb_dim"]["when_any"],
            [{"architecture": ["anima"], "mode": ["controlnet_lllite"]}],
        )
        self.assertIn("fixed to bf16", musubi["unsupported_fields"]["mixed_precision"])
        self.assertIn("recipe.steps", musubi["unsupported_fields"]["epochs"])
        self.assertEqual(
            backend_capabilities("ai-toolkit")["nested_config_fields"]["dataset_config"],
            {
                "do_i2v": {"type": "boolean"},
                "do_audio": {"type": "boolean"},
                "fps": {"type": "integer", "minimum": 1},
                "generated_controls": {
                    "type": "string-list",
                    "choices": ("depth", "pose", "line", "inpaint", "mask"),
                },
                "num_frames": {"type": "integer", "minimum": 1},
            },
        )
        ai_toolkit = backend_capabilities("ai-toolkit")
        self.assertIn("flatten_groups", ai_toolkit["config_fields"])
        self.assertIn("dataset_options", ai_toolkit["config_fields"])
        self.assertEqual(
            ai_toolkit["nested_config_fields"]["dataset_options.<dataset-id>"],
            {"blocks": {"type": "list of group and num_repeats mappings"}},
        )
        model_arch_choices = backend_capabilities("ai-toolkit")["config_value_choices"]["model_arch"]
        self.assertIn("sd1", model_arch_choices)
        self.assertIn("qwen_image_2", model_arch_choices)
        self.assertNotIn("sd15", model_arch_choices)

    def test_selector_values_have_one_spelling_and_are_refused_first(self) -> None:
        cases = (
            ("sd-scripts", {"architecture": "flux", "guidance_scale": 1.0}, "architecture 'flux' is not accepted; use one of: anima, flux1, sd15, sdxl"),
            ("sd-scripts", {"architecture": "Anima", "attn_mode": "torch"}, "architecture 'Anima' is not accepted"),
            ("sd-scripts", {"architecture": "anima", "mode": "lllite", "lllite_cond_dim": 32}, "mode 'lllite' is not accepted; use one of: controlnet_lllite, lora"),
            ("musubi-tuner", {"architecture": "flux_2", "fp8_base": True}, "architecture 'flux_2' is not accepted"),
            ("musubi-tuner", {"architecture": "Flux2", "f1": True}, "architecture 'Flux2' is not accepted"),
            ("musubi-tuner", {"architecture": 5, "fp8_base": True}, "architecture 5 is not accepted"),
            ("musubi-tuner", {"architecture": "unknownarch", "fp8_base": True}, "architecture 'unknownarch' is not accepted"),
            ("ai-toolkit", {"model_arch": "FLUX"}, "model_arch 'FLUX' is not accepted"),
            ("ai-toolkit", {"model_arch": "sd1:portrait"}, "model_arch 'sd1:portrait' is not accepted"),
            ("ai-toolkit", {"model_arch": "Flex2", "bypass_guidance_embedding": True}, "model_arch 'Flex2' is not accepted"),
        )
        for backend, config, message in cases:
            with self.subTest(backend=backend, config=config):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    validate_backend_config({"backend": {"name": backend, "config": config}})

    def test_decided_integers_have_one_form_on_every_backend(self) -> None:
        for backend, base in (("sd-scripts", {"architecture": "sd15"}), ("musubi-tuner", {"architecture": "wan"}), ("ai-toolkit", {"model_arch": "sdxl"})):
            for value in (2.5, "250", 0, True):
                with self.subTest(backend=backend, value=value):
                    with self.assertRaisesRegex(ValueError, "save_every_n_steps must be an integer of at least 1"):
                        validate_backend_config({"backend": {"name": backend, "config": {**base, "save_every_n_steps": value}}})
        validate_backend_config({"backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl", "save_last_n_steps": 0}}, "model": {"base": "x/y"}})
        with self.assertRaisesRegex(ValueError, "save_last_n_steps must be an integer of at least 1"):
            validate_backend_config({"backend": {"name": "sd-scripts", "config": {"architecture": "sd15", "save_last_n_steps": 0}}})

    def test_capabilities_list_the_fields_kura_decides_from(self) -> None:
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                decided = backend_capabilities(backend)["decided_fields"]
                self.assertTrue({"save_every_n_steps", "gradient_accumulation_steps", "lr_scheduler"} <= set(decided))

    @posix_only(DATASET_IO)
    def test_ai_toolkit_native_config_cannot_set_what_kura_decides_from(self) -> None:
        for section, key in (("save", "save_every"), ("save", "max_step_saves_to_keep"), ("train", "gradient_accumulation_steps"), ("train", "gradient_accumulation"), ("train", "lr_scheduler"), ("train", "optimizer")):
            with self.subTest(key=f"{section}.{key}"), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                run = {
                    "id": "native-owned", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl", "native_config": {section: {key: 1}}}},
                    "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
                }
                self._write_ai_toolkit_projection(run, destination)
                with self.assertRaisesRegex(ValueError, f"overrides Kura-owned field.*{section}.{key}"):
                    BACKENDS["ai-toolkit"].compile(run, destination)

    def test_escape_hatches_cannot_change_the_schedule_or_the_steps(self) -> None:
        from kura.backends import musubi_command, sd_scripts

        for flags in (musubi_command.MUSUBI_OWNED_FLAGS, sd_scripts.OWNED_FLAGS):
            self.assertLessEqual({"--lr_scheduler_type", "--max_train_epochs"}, flags)

    def test_musubi_pruning_takes_a_positive_integer_only(self) -> None:
        for value in ("1500", 2.5, True, 0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prune_checkpoints_before_step must be an integer of at least 1"):
                validate_backend_config({"backend": {"name": "musubi-tuner", "config": {"architecture": "wan", "prune_checkpoints_before_step": value}}})

    def test_an_unknown_selector_is_reported_before_other_value_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "architecture 'flux' is not accepted"):
            validate_backend_config({"backend": {"name": "sd-scripts", "config": {"architecture": "flux", "save_every_n_steps": "x", "cache_latents": "yes"}}})

    def test_model_arch_is_not_a_musubi_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported key\\(s\\): 'model_arch'"):
            validate_backend_config({"backend": {"name": "musubi-tuner", "config": {"model_arch": "flux2"}}})

    def test_listed_ai_toolkit_values_are_accepted_as_written(self) -> None:
        for arch in ("f-lite", "ltx2.5", "flex2", "zimage:turbo"):
            with self.subTest(arch=arch):
                validate_backend_config({"backend": {"name": "ai-toolkit", "config": {"model_arch": arch}}, "model": {"base": "example/model"}})
        validate_backend_config({"backend": {"name": "ai-toolkit", "config": {"model_arch": "krea2:turbo", "model_edit": True}}, "model": {"base": "example/model"}})

    def test_a_conditional_field_is_refused_when_its_selector_is_omitted(self) -> None:
        cases = (
            {"architecture": "wan", "convrot_int8_bwd": "bf16"},
            {"architecture": "wan", "gradient_checkpointing_cpu_offload": True},
            {"architecture": "krea2", "convrot_int8_bwd": "bf16"},
            {"architecture": "krea2", "gradient_checkpointing_cpu_offload": True},
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "not applicable"):
                    validate_backend_config({"backend": {"name": "musubi-tuner", "config": config}})
        validate_backend_config({"backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", {
            "architecture": "krea2", "convrot_int8": True, "convrot_int8_bwd": "bf16",
        })}})

    def test_ai_toolkit_legacy_dataset_selectors_name_manifest_migrations(self) -> None:
        base = {
            "id": "legacy-ai-toolkit-dataset",
            "backend": {"name": "ai-toolkit", "config": {"model_arch": "qwen_image_2"}},
            "model": {"base": "Qwen/Qwen-Image-2.1"},
            "datasets": [{"id": "paired"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        folder = deepcopy(base)
        folder["backend"]["config"]["dataset_folder"] = "/workspace/datasets/paired/images"
        with self.assertRaisesRegex(
            ValueError,
            r"dataset_folder.*items\.jsonl.*role.*target",
        ):
            validate_backend_config(folder)

        control = deepcopy(base)
        control["backend"]["config"]["dataset_config"] = {"control_subdir": "control"}
        with self.assertRaisesRegex(
            ValueError,
            r"dataset_config\.control_subdir.*items\.jsonl.*role.*control",
        ):
            validate_backend_config(control)

    def test_musubi_rejects_declared_fields_on_the_wrong_architecture(self) -> None:
        for field, value in (("timestep_boundary", 900), ("noise_scale_start", 0.5)):
            with self.subTest(field=field):
                run = {"backend": {"name": "musubi-tuner", "config": {"architecture": "flux2", field: value}}}
                with self.assertRaisesRegex(ValueError, rf"{field}.*not applicable.*architecture='flux2'"):
                    validate_backend_config(run)

    def test_musubi_minimax_h3_fields_are_declared_for_that_architecture(self) -> None:
        run = {"backend": {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "t2va",
            "model_bundle": "minimax-h3-ref2va-pruned-int8",
            "h3_loss_method": "teacher_matching",
            "h3_teacher_conditions": "subject_ref",
            "h3_teacher_condition_sigma_min": 0.15,
            "h3_teacher_condition_sigma_max": 1.0,
            "h3_teacher_loss_dc_weight": 0.3,
            "h3_teacher_loss_mag_weight": 0.5,
            "h3_teacher_preservation_weight": 1.0,
            "h3_timestep_focus_min": 0.0,
            "h3_timestep_focus_max": 1.0,
            "h3_timestep_focus_prob": 0.0,
            "one_frame": True,
            "h3_guidance_loss_scale": 4.0,
            "text_encoder_blocks_to_swap": 50,
            "video_only": True,
        }}}

        with self.assertRaisesRegex(ValueError, "h3_guidance_loss_scale.*not applicable"):
            validate_backend_config(run)

        del run["backend"]["config"]["h3_guidance_loss_scale"]
        run["backend"]["config"] = with_required("musubi-tuner", run["backend"]["config"])
        validate_backend_config(run)

    def test_musubi_minimax_h3_loss_fields_reject_the_wrong_loss_method(self) -> None:
        cases = (
            ({"h3_loss_method": "training_adapter", "h3_guidance_loss_scale": 4.0}, "h3_guidance_loss_scale"),
            ({"h3_loss_method": "guidance", "h3_teacher_conditions": "ref"}, "h3_teacher_conditions"),
        )
        for fields, expected in cases:
            with self.subTest(fields=fields):
                run = {"backend": {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3", **fields,
                }}}
                with self.assertRaisesRegex(ValueError, rf"{expected}.*not applicable"):
                    validate_backend_config(run)

    def test_musubi_minimax_h3_architecture_requirements_guard_batch_and_resolution(self) -> None:
        base = {
            "backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", {
                "architecture": "minimax_h3",
            })},
            "datasets": [{"id": "tiny"}],
        }
        invalid = (
            ({"batch_size": 2}, r"minimax_h3.*batch_size=1"),
            ({"batch_size": True}, r"take a value, not true or false: batch_size=True"),
            ({"resolution": [960, 550]}, r"minimax_h3.*resolution.*multiples of 32"),
            ({"resolution": [960]}, r"minimax_h3.*resolution.*multiples of 32"),
            (
                {"dataset_options": {"tiny": {"blocks": [{"resolution": [960, 550]}]}}},
                r"minimax_h3.*dataset_options\.tiny\.blocks\[0\]\.resolution.*multiples of 32",
            ),
        )
        for fields, expected in invalid:
            with self.subTest(fields=fields):
                run = deepcopy(base)
                run["backend"]["config"].update(fields)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_backend_config(run)

        validate_backend_config(base)
        explicit = deepcopy(base)
        explicit["backend"]["config"].update({
            "batch_size": 1,
            "resolution": [1024, 768],
        })
        validate_backend_config(explicit)

        unrelated = deepcopy(base)
        unrelated["backend"]["config"] = with_required("musubi-tuner", {
            "architecture": "wan",
            "batch_size": 2,
            "resolution": [960, 550],
        })
        validate_backend_config(unrelated)

    def test_musubi_capabilities_expose_manifest_projection_options_only(self) -> None:
        capabilities = backend_capabilities("musubi-tuner")

        self.assertNotIn("dataset_config", capabilities["config_fields"])
        self.assertNotIn("dataset_config", capabilities["escape_hatches"])
        self.assertNotIn("h3_dataset_config", capabilities["config_fields"])
        self.assertNotIn("h3_dataset_config", capabilities["conditional_fields"])
        self.assertEqual(
            capabilities["nested_config_fields"]["dataset_options.<dataset-id>"]["fp_1f_target_index"]["minimum"],
            0,
        )

    def test_musubi_video_dataset_options_are_discoverable(self) -> None:
        capabilities = backend_capabilities("musubi-tuner")

        self.assertIn("dataset_options", capabilities["config_fields"])
        options = capabilities["nested_config_fields"]["dataset_options.<dataset-id>"]
        self.assertEqual(
            options["target_frames"]["grid"],
            "profile-specific: Wan 1+4n; MiniMax-H3 5+17n",
        )
        self.assertEqual(
            options["frame_extraction"],
            {"type": "enum:head|full", "default": "profile-specific"},
        )
        self.assertEqual(options["max_frames"], {"type": "integer", "minimum": 1})
        self.assertEqual(options["control_resolution"], {"type": "integer-pair", "minimum": 1})
        self.assertEqual(options["fp_1f_clean_indices"], {"type": "integer-list", "minimum": 0})
        self.assertEqual(options["fp_1f_target_index"], {"type": "integer", "minimum": 0})
        self.assertEqual(options["fp_1f_no_post"], {"type": "boolean", "default": False})
        self.assertEqual(options["no_resize_control"], {"type": "boolean"})

    def test_musubi_capabilities_expose_krea2_memory_accommodations(self) -> None:
        capabilities = backend_capabilities("musubi-tuner")

        self.assertEqual(
            capabilities["conditional_fields"]["convrot_int8"]["when_any"],
            [{"architecture": ["krea2"]}],
        )
        self.assertEqual(
            capabilities["conditional_fields"]["convrot_int8_bwd"]["when_any"],
            [{"architecture": ["krea2"], "convrot_int8": [True]}],
        )
        self.assertEqual(
            capabilities["conditional_fields"]["gradient_checkpointing_cpu_offload"]["when_any"],
            [{"architecture": ["krea2"], "gradient_checkpointing": [True]}],
        )

    def test_compile_rejects_known_but_inapplicable_fields_before_artifact_generation(self) -> None:
        cases = (
            ("musubi-tuner", {"architecture": "flux2", "timestep_boundary": 900}),
            ("musubi-tuner", {"architecture": "flux2", "noise_scale_start": 0.5}),
            ("sd-scripts", {"architecture": "sd15", "mode": "lora", "cond_emb_dim": 32}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, config in cases:
                with self.subTest(backend=name, field=next(reversed(config))):
                    run = {"backend": {"name": name, "config": config}}
                    with self.assertRaisesRegex(ValueError, "not applicable"):
                        BACKENDS[name].compile(run, root / name)

    def test_sd_scripts_rejects_declared_fields_on_the_wrong_mode(self) -> None:
        cases = (
            ({"architecture": "anima", "mode": "lora", "cond_emb_dim": 32}, "cond_emb_dim.*mode='lora'"),
            ({"architecture": "anima", "mode": "controlnet_lllite", "network_alpha": 4}, "network_alpha.*controlnet_lllite"),
            ({"architecture": "anima", "mode": "controlnet_lllite", "blocks_to_swap": 4}, "blocks_to_swap.*controlnet_lllite"),
        )
        for config, message in cases:
            with self.subTest(field=next(reversed(config))):
                run = {"backend": {"name": "sd-scripts", "config": config}}
                with self.assertRaisesRegex(ValueError, message):
                    validate_backend_config(run)

    def test_every_conditional_field_is_enforced_by_the_shared_validator(self) -> None:
        for name, adapter in BACKENDS.items():
            conditional_names = {item.field for item in adapter.surface.conditions}
            for condition in adapter.surface.conditions:
                with self.subTest(backend=name, field=condition.field):
                    first_clause = condition.when_any[0]
                    booleans = adapter.surface.boolean_fields
                    # Values of the right type, so only applicability can refuse them.
                    field_choices = dict(adapter.surface.config_value_choices).get(condition.field)
                    config = {condition.field: True if condition.field in booleans else field_choices[0] if field_choices else "sentinel"}
                    for selector, allowed in first_clause:
                        config[selector] = allowed[0]
                    candidates = [selector for selector, _ in first_clause if selector not in conditional_names]
                    self.assertTrue(candidates, f"{name}.{condition.field} needs an independently selectable condition")
                    off = candidates[-1]
                    choices = dict(adapter.surface.config_value_choices).get(off)
                    # The off value is a valid selector value, so only applicability can refuse the field.
                    config[off] = (
                        (not config[off]) if off in booleans
                        else next(choice for choice in choices if all(choice not in allowed for selector, allowed in first_clause if selector == off))
                        if choices else "not-a-supported-selector"
                    )
                    run = {"backend": {"name": name, "config": config}}
                    with self.assertRaisesRegex(ValueError, rf"{condition.field}.*not applicable"):
                        validate_backend_config(run)

    def test_capabilities_cli_has_machine_readable_output(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cmd_run_capabilities(type("Args", (), {"backend": "musubi-tuner", "json": True})()), 0)
        self.assertIn('"optimizer_type"', output.getvalue())
        self.assertIn('"validation": "unverified"', output.getvalue())

    def test_capabilities_cli_prints_musubi_selector_values(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cmd_run_capabilities(type("Args", (), {"backend": "musubi-tuner", "json": False})()), 0)
        rendered = output.getvalue()
        self.assertNotIn("aliases", rendered)
        self.assertIn("selector values (spelled exactly as listed", rendered)
        self.assertIn("architecture: flux2, flux_kontext", rendered)

    def test_capabilities_cli_prints_sd_scripts_nested_dataset_fields(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cmd_run_capabilities(type("Args", (), {"backend": "sd-scripts", "json": False})()), 0)
        rendered = output.getvalue()
        self.assertIn("dataset_config.datasets[].subsets[]", rendered)
        self.assertIn("caption_dropout_rate (number", rendered)

    def test_replacing_behavior_does_not_remove_the_contract(self) -> None:
        adapter = BACKENDS["musubi-tuner"]
        changed = replace(adapter, command=lambda run: {"cwd": "/", "argv": ["true"], "env": {}})
        self.assertEqual(changed.surface, adapter.surface)

    def test_musubi_training_command_injection_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one training command"):
            musubi_script_command([["python", "cache.py"]], {"lr_scheduler": "constant"})
        with self.assertRaisesRegex(ValueError, "found 2"):
            musubi_script_command(
                [["python", "a.py", "--max_train_steps", "1"], ["python", "b.py", "--max_train_steps", "1"]],
                {"lr_scheduler": "constant"},
            )

    def test_musubi_resume_saves_state_at_the_derived_endpoint(self) -> None:
        # The patched trainer counts logical steps toward the logical target and keeps the
        # recipe's cadence; the endpoint is its final state, which the runner marks with the
        # step it holds, so nothing caps the cadence at the steps the run adds.
        commands = [["python", "train.py", "--max_train_steps", "2000", "--save_every_n_steps", "2000"]]
        run = {
            "id": "derived",
            "parent_run": "source",
            "recipe": {"steps": 2000, "seed": 1},
            "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            "continuation": {
                "mode": "resume",
                "source": {
                    "artifact_id": "state-2000",
                    "manifest_sha256": "a" * 64,
                    "observed_step": 2000,
                    "recipe_sha256": "b" * 64,
                },
                "additional_steps": 1000,
                "target_step": 3000,
                "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
            },
        }

        result = musubi_script_command(commands, {"lr_scheduler": "constant"}, run)
        script = result[2]

        self.assertIn("--max_train_steps 3000", script)
        self.assertIn("--save_every_n_steps 2000", script)
        self.assertNotIn("--save_state_on_train_end", script)
        self.assertIn("--save_last_n_steps_state 2000", script)
        self.assertIn("/workspace/runs/derived/resolved/musubi/state-runner.py musubi-tuner train.py ", script)

    def test_explicit_command_cannot_silently_discard_other_config(self) -> None:
        for name in BACKENDS:
            with self.subTest(backend=name):
                run = {
                    "backend": {"name": name, "config": {
                        "command": {"cwd": "/workspace", "argv": ["true"], "env": {}},
                        "learning_rate": 0.0001,
                    }},
                    "recipe": {},
                }
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    BACKENDS[name].command(run)

    @posix_only(DATASET_IO)
    def test_ai_toolkit_raw_override_cannot_shadow_owned_model_architecture(self) -> None:
        run = {
            "id": "ambiguous-ai", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "sdxl", "native_config": {"model": {"arch": "flux"}},
            }},
            "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            self._write_ai_toolkit_projection(run, Path(directory))
            with self.assertRaisesRegex(ValueError, "duplicates backend.config.model_arch"):
                BACKENDS["ai-toolkit"].compile(run, Path(directory))

    def test_ai_toolkit_rejects_sd15_before_compiling_native_config(self) -> None:
        run = {
            "id": "invalid-sd15", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd15"}},
            "model": {"base": "hf-internal-testing/tiny-stable-diffusion-pipe"},
            "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(ValueError, "sd15.*sd1"):
                BACKENDS["ai-toolkit"].compile(run, destination)
            self.assertFalse((destination / "ai-toolkit.yaml").exists())

    def test_ai_toolkit_selector_catalog_is_tied_to_the_declared_image_pin(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        base_pin = "ostris/aitoolkit:0.13.32@sha256:7575870019739440bf5a215918d5c128abf80719570637b33e6b06b7830494e0"
        runtime_pin = "nomadoor/kura-ai-toolkit@sha256:2e1c004e01bfc61a23b0814aa62f0fa664ce134e23c7ee57ba643b021ea9f553"
        dockerfile = (repository / "docker" / "ai-toolkit" / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(dockerfile.splitlines()[:2], [f"ARG AI_TOOLKIT_IMAGE={base_pin}", "FROM ${AI_TOOLKIT_IMAGE}"])

        from kura.images import BUILD_SOURCES, PINNED_IMAGES

        self.assertEqual(PINNED_IMAGES["ai-toolkit"], runtime_pin)
        self.assertEqual(BUILD_SOURCES["ai-toolkit"], ("AI_TOOLKIT_IMAGE", base_pin))

    @posix_only(DATASET_IO)
    def test_ai_toolkit_accepts_registered_sd1_without_rewriting_it(self) -> None:
        run = {
            "id": "registered-sd1", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd1"}},
            "model": {"base": "hf-internal-testing/tiny-stable-diffusion-pipe"},
            "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self._write_ai_toolkit_projection(run, destination)
            BACKENDS["ai-toolkit"].compile(run, destination)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
            self.assertEqual(process["model"]["arch"], "sd1")

    @posix_only(DATASET_IO)
    def test_ai_toolkit_cli_reports_sd15_correction_before_dataset_resolution(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "invalid-sd15"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text(yaml.safe_dump({
                "schema_version": 2, "id": "invalid-sd15", "type": "train",
                "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd15"}},
                "model": {"base": "hf-internal-testing/tiny-stable-diffusion-pipe"},
                "recipe": {"steps": 1, "seed": 1}, "datasets": [{"id": "tiny"}],
            }), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            os.chdir(root)
            try:
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    self.assertEqual(cmd_run_compile(argparse.Namespace(run_id="invalid-sd15")), 1)
            finally:
                os.chdir(previous)
            self.assertRegex(error.getvalue(), "sd15.*sd1")
            self.assertFalse((run_dir / "resolved" / "backend-command.lock.json").exists())

    @posix_only(DATASET_IO)
    def test_ai_toolkit_native_model_arch_cannot_claim_first_class_manifest_projection(self) -> None:
        run = {
            "id": "custom-arch", "backend": {"name": "ai-toolkit", "config": {
                "native_config": {"model": {"arch": "custom_extension_arch"}},
            }},
            "model": {"base": "custom/model"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(ValueError, "no verified AI-Toolkit projection profile"):
                self._write_ai_toolkit_projection(run, destination)

    def test_ai_toolkit_minimax_h3_rejects_gradient_checkpointing_for_pinned_runtime(self) -> None:
        run = {
            "id": "minimax-checkpointing", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "minimax_h3", "gradient_checkpointing": True,
            }},
            "model": {"base": "Comfy-Org/MiniMax-H3"},
            "datasets": [{"id": "video"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "gradient_checkpointing=false"):
                BACKENDS["ai-toolkit"].compile(run, Path(directory))

    @posix_only(DATASET_IO)
    def test_ai_toolkit_flex2_bypass_guidance_is_typed_and_compiled(self) -> None:
        run = {
            "id": "flex2-typed", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "flex2", "bypass_guidance_embedding": True,
            }},
            "model": {"base": "ostris/Flex.2-preview"},
            "datasets": [{"id": "images"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self._write_ai_toolkit_projection(run, destination)
            BACKENDS["ai-toolkit"].compile(run, destination)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))[
                "config"
            ]["process"][0]
        self.assertIs(process["train"]["bypass_guidance_embedding"], True)
        self.assertIn(
            "bypass_guidance_embedding",
            backend_capabilities("ai-toolkit")["conditional_fields"],
        )

        invalid = deepcopy(run)
        invalid["backend"]["config"]["bypass_guidance_embedding"] = "true"
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            validate_backend_config(invalid)

    @posix_only(DATASET_IO)
    def test_ai_toolkit_zimage_l2p_companion_source_is_typed_and_compiled(self) -> None:
        run = {
            "id": "zimage-l2p-typed", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "zimage_l2p",
                "extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo",
            }},
            "model": {"base": "zhen-nan/L2P/model-1k-merge.safetensors"},
            "datasets": [{"id": "images"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self._write_ai_toolkit_projection(run, destination)
            BACKENDS["ai-toolkit"].compile(run, destination)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))[
                "config"
            ]["process"][0]
        self.assertEqual(
            process["model"]["extras_name_or_path"],
            "Tongyi-MAI/Z-Image-Turbo",
        )
        self.assertIn(
            "extras_name_or_path",
            backend_capabilities("ai-toolkit")["conditional_fields"],
        )
        self.assertEqual(
            BACKENDS["ai-toolkit"].requirements(run)[1]["role"],
            "model_extras",
        )

        invalid = deepcopy(run)
        invalid["backend"]["config"]["extras_name_or_path"] = "   "
        with self.assertRaisesRegex(ValueError, "must be a nonempty model reference"):
            validate_backend_config(invalid)

    @posix_only(DATASET_IO)
    def test_ai_toolkit_krea2_edit_mode_is_typed_and_compiled(self) -> None:
        run = {
            "id": "krea2-edit-typed", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "krea2", "model_edit": True,
            }},
            "model": {"base": "black-forest-labs/FLUX.2-krea-dev"},
            "datasets": [{"id": "images"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self._write_ai_toolkit_manifest_projection(
                run, destination, target="sample.png", controls=("control.png",),
            )
            BACKENDS["ai-toolkit"].compile(run, destination)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))[
                "config"
            ]["process"][0]
        self.assertIs(process["model"]["model_kwargs"]["edit"], True)
        self.assertIn(
            "model_edit",
            backend_capabilities("ai-toolkit")["conditional_fields"],
        )

        invalid = deepcopy(run)
        invalid["backend"]["config"]["model_edit"] = "true"
        with self.assertRaisesRegex(ValueError, "must be true or false: model_edit='true'"):
            validate_backend_config(invalid)

    def test_ai_toolkit_generated_controls_are_closed_and_flex2_only(self) -> None:
        run = {
            "id": "generated-controls", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "flex2",
                "dataset_config": {"generated_controls": ["depth", "pose"]},
            }},
            "model": {"base": "ostris/Flex.2-preview"},
            "datasets": [{"id": "images"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        validate_backend_config(run)

        invalid = deepcopy(run)
        invalid["backend"]["config"]["dataset_config"]["generated_controls"] = ["depth", "depth"]
        with self.assertRaisesRegex(ValueError, "unique list"):
            validate_backend_config(invalid)

        invalid = deepcopy(run)
        invalid["backend"]["config"]["model_arch"] = "sdxl"
        with self.assertRaisesRegex(ValueError, "verified only for flex2"):
            validate_backend_config(invalid)

    @posix_only(DATASET_IO)
    def test_ai_toolkit_sdxl_evidence_settings_survive_and_the_baseline_fills_the_rest(self) -> None:
        """Authored SDXL evidence settings win; the pinned UI baseline fills only unset keys."""
        ordinary = {
            "model_arch": "sdxl", "network_dim": 1, "network_alpha": 1,
            "save_every_n_steps": 1, "save_last_n_steps": 1,
            "resolution": [256, 256], "learning_rate": 1.0e-6, "batch_size": 1,
            "gradient_accumulation_steps": 1, "gradient_checkpointing": True,
            "mixed_precision": "bf16", "optimizer_type": "adamw8bit", "low_vram": True,
        }
        for executor, expected_cwd in (("docker", "/app/ai-toolkit"), ("runpod", "/app/ai-toolkit")):
            with self.subTest(executor=executor), tempfile.TemporaryDirectory() as directory:
                run_id = f"evidence-sdxl-{executor}"
                run = {
                    "id": run_id, "backend": {"name": "ai-toolkit", "config": ordinary},
                    "model": {"base": "stabilityai/stable-diffusion-xl-base-1.0"},
                    "datasets": [{"id": "flux2-klein-tiny"}],
                    "recipe": {"steps": 1, "seed": 1}, "compute": {"executor": executor},
                }
                destination = Path(directory) / "ai-toolkit"
                self._write_ai_toolkit_projection(run, destination)
                command = BACKENDS["ai-toolkit"].compile(run, destination)
                process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                self.assertEqual(process, {
                    "type": "sd_trainer", "training_folder": f"/workspace/runs/{run_id}/outputs", "device": "cuda:0",
                    "network": {"type": "lora", "linear": 1, "linear_alpha": 1},
                    "save": {"save_every": 1, "max_step_saves_to_keep": 1},
                    "datasets": [{
                        "folder_path": f"/workspace/runs/{run_id}/cache/dataset-view/ai-toolkit/flux2-klein-tiny", "caption_ext": ".txt",
                        "cache_latents_to_disk": False, "resolution": [256, 256],
                    }],
                    "train": {
                        "steps": 1, "train_unet": True, "train_text_encoder": False, "disable_sampling": True,
                        "seed": 1, "lr": 1.0e-6, "optimizer": "adamw8bit", "dtype": "bf16",
                        "batch_size": 1, "gradient_accumulation": 1, "gradient_checkpointing": True,
                        "noise_scheduler": "ddpm", "timestep_type": "sigmoid", "optimizer_params": {"weight_decay": 0.0001},
                        "content_or_style": "balanced", "loss_type": "mse",
                        "unload_text_encoder": False, "cache_text_embeddings": False,
                    },
                    "model": {
                        "name_or_path": "stabilityai/stable-diffusion-xl-base-1.0", "arch": "sdxl",
                        "low_vram": True, "quantize": False, "qtype": "qfloat8", "quantize_te": False,
                        "qtype_te": "qfloat8",
                    },
                })
                self.assertEqual(command["cwd"], expected_cwd)
                self.assertEqual(command["argv"][:2], ["python", "-c"])
                self.assertIn(f"/workspace/runs/{run_id}/resolved/ai-toolkit.yaml", command["argv"][3])
                self.assertEqual(command["env"], {
                    "SEED": "1",
                    "MODELS_PATH": "/workspace/cache/huggingface/ai-toolkit-models",
                    "KURA_AI_TOOLKIT_VIDEO_SUFFIXES": frozen_suffixes(AI_TOOLKIT_VIDEO_SUFFIXES),
                })

    @posix_only(DATASET_IO)
    def test_ai_toolkit_keeps_every_step_save_unless_the_run_sets_a_retention(self) -> None:
        """The pinned trainer prunes to its own max_step_saves_to_keep when none is given; Kura gives 0 (keep every save)."""
        for executor in ("docker", "runpod"):
            for authored, expected in ((None, 0), (0, 0), (3, 3)):
                config = {"model_arch": "sdxl", "save_every_n_steps": 1}
                if authored is not None:
                    config["save_last_n_steps"] = authored
                with self.subTest(executor=executor, save_last_n_steps=authored), tempfile.TemporaryDirectory() as directory:
                    run = {
                        "id": f"retention-{executor}", "backend": {"name": "ai-toolkit", "config": config},
                        "model": {"base": "stabilityai/stable-diffusion-xl-base-1.0"},
                        "datasets": [{"id": "flux2-klein-tiny"}],
                        "recipe": {"steps": 1, "seed": 1}, "compute": {"executor": executor},
                    }
                    destination = Path(directory) / "ai-toolkit"
                    self._write_ai_toolkit_projection(run, destination)
                    BACKENDS["ai-toolkit"].compile(run, destination)
                    process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                    self.assertEqual(process["save"], {"save_every": 1, "max_step_saves_to_keep": expected})
                    self.assertEqual(BACKENDS["ai-toolkit"].display(run)["checkpoint"]["keep_last"], expected)

    def test_ai_toolkit_baseline_fills_only_unset_values(self) -> None:
        from kura.backends.ai_toolkit import ai_toolkit_baseline

        run = {
            "id": "klein", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "flux2_klein_4b", "mixed_precision": "fp16",
                "native_config": {"train": {"timestep_type": "sigmoid"}, "model": {"model_kwargs": {"extra": 1}}},
            }},
            "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        baseline = ai_toolkit_baseline(run)
        self.assertEqual(baseline["entry"], "flux2_klein_4b")
        self.assertEqual(baseline["train"]["noise_scheduler"], "flowmatch")
        self.assertNotIn("dtype", baseline["train"])  # authored mixed_precision wins
        self.assertNotIn("timestep_type", baseline["train"])  # authored native value wins
        self.assertEqual(baseline["model"]["model_kwargs"], {"match_target_res": False})  # only missing subkeys
        self.assertEqual(baseline["dataset"], {"cache_latents_to_disk": False})

        tagged = deepcopy(run)
        tagged["backend"]["config"]["model_arch"] = "zimage:turbo"
        tagged["model"]["base"] = "not/a-listed-path"
        self.assertEqual(ai_toolkit_baseline(tagged)["entry"], "zimage:turbo")
        # model.base never chooses the entry; a base that is another entry's model conflicts.
        tagged["model"]["base"] = "Tongyi-MAI/Z-Image"
        with self.assertRaisesRegex(ValueError, r"model.base 'Tongyi-MAI/Z-Image' is the model of UI entry zimage, but model_arch 'zimage:turbo' selects entry 'zimage:turbo'"):
            ai_toolkit_baseline(tagged)

        # krea2:turbo and krea2:o_edit_turbo share one model path; the selector
        # decides, and edit mode never comes from the baseline.
        for selector, expected in (("krea2:turbo", "krea2:turbo"), ("krea2", None)):
            with self.subTest(selector=selector):
                turbo = deepcopy(run)
                turbo["backend"]["config"] = {"model_arch": selector}
                turbo["model"]["base"] = "krea/Krea-2-Turbo"
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "is the model of UI entry krea2:o_edit_turbo, krea2:turbo, but model_arch 'krea2' selects entry 'krea2'"):
                        ai_toolkit_baseline(turbo)
                    continue
                selected = ai_toolkit_baseline(turbo)
                self.assertEqual(selected["entry"], expected)
                self.assertNotIn("edit", selected["model"].get("model_kwargs", {}))
        edit = deepcopy(run)
        edit["backend"]["config"] = {"model_arch": "krea2:o_edit", "model_edit": True}
        edit["model"]["base"] = "krea/Krea-2-Raw"
        selected = ai_toolkit_baseline(edit)
        self.assertEqual(selected["entry"], "krea2:o_edit")
        self.assertNotIn("edit", selected["model"].get("model_kwargs", {}))

    def test_ai_toolkit_architecture_without_a_baseline_requires_authored_scheduler_and_precision(self) -> None:
        from kura.backends.ai_toolkit import ai_toolkit_baseline

        run = {
            "id": "sd3", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd3"}},
            "model": {"base": "stabilityai/stable-diffusion-3-medium"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with self.assertRaisesRegex(ValueError, "noise_scheduler.*mixed_precision"):
            ai_toolkit_baseline(run)
        run["backend"]["config"].update({"mixed_precision": "bf16", "native_config": {"train": {"noise_scheduler": "flowmatch"}}})
        self.assertEqual(ai_toolkit_baseline(run)["entry"], None)

    def test_ai_toolkit_baseline_is_extracted_from_the_pinned_image(self) -> None:
        from kura.backends.ai_toolkit import AI_TOOLKIT_PINNED_COMMIT, AI_TOOLKIT_PINNED_IMAGE, AI_TOOLKIT_PINNED_MODEL_ARCHS
        from kura.backends.ai_toolkit_baseline import load_baseline

        upstream = load_baseline()["upstream"]
        self.assertEqual(upstream["commit"], AI_TOOLKIT_PINNED_COMMIT)
        self.assertEqual(upstream["image"], AI_TOOLKIT_PINNED_IMAGE)
        from kura.images import PINNED_IMAGES

        self.assertEqual(PINNED_IMAGES["ai-toolkit"], AI_TOOLKIT_PINNED_IMAGE)
        arches = {entry["arch"] for entry in load_baseline()["entries"].values()}
        self.assertLessEqual(arches, AI_TOOLKIT_PINNED_MODEL_ARCHS)

    @posix_only(DATASET_IO)
    def test_ai_toolkit_projects_typed_video_dataset_settings(self) -> None:
        run = {
            "id": "minimax-video", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "minimax_h3",
                "dataset_config": {"num_frames": 5, "fps": 24, "do_audio": False},
            }},
            "model": {"base": "Comfy-Org/MiniMax-H3"},
            "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_ai_toolkit_manifest_projection(
                run, root, target="sample.mp4",
            )
            BACKENDS["ai-toolkit"].compile(run, root)
            process = yaml.safe_load(
                (root / "ai-toolkit.yaml").read_text(encoding="utf-8")
            )["config"]["process"][0]
        self.assertEqual(process["datasets"][0]["num_frames"], 5)
        self.assertEqual(process["datasets"][0]["fps"], 24)
        self.assertIs(process["datasets"][0]["do_audio"], False)
        self.assertIn("/cache/dataset-view/", process["datasets"][0]["folder_path"])

    @posix_only(DATASET_IO)
    def test_ai_toolkit_projects_manifest_controls_inside_each_owned_view(self) -> None:
        run = {
            "id": "qwen-edit", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "qwen_image_2",
            }},
            "model": {"base": "Qwen/Qwen-Image-2.1"},
            "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_ai_toolkit_manifest_projection(
                run, root, target="sample.png", controls=("control.png",),
            )
            BACKENDS["ai-toolkit"].compile(run, root)
            process = yaml.safe_load(
                (root / "ai-toolkit.yaml").read_text(encoding="utf-8")
            )["config"]["process"][0]
        self.assertIn("/cache/dataset-view/", process["datasets"][0]["folder_path"])
        self.assertEqual(len(process["datasets"][0]["control_path"]), 1)
        self.assertIn("/cache/dataset-view/", process["datasets"][0]["control_path"][0])

    @posix_only(DATASET_IO)
    def test_ai_toolkit_passes_new_0_13_30_native_options_through_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = {
                "id": "native-options",
                "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd1", "dataset_config": {}, "native_config": {
                    "train": {"loss_type": "band", "pool_datasets": True},
                    "model": {"model_kwargs": {"rgba": True}},
                }}},
                "model": {"base": "stable-diffusion-v1-5/stable-diffusion-v1-5"}, "datasets": [{"id": "tiny"}],
                "recipe": {"steps": 1, "seed": 1},
            }
            self._write_ai_toolkit_manifest_projection(run, root, target="sample.png", controls=())
            BACKENDS["ai-toolkit"].compile(run, root)
            process = yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
        self.assertEqual((process["train"]["loss_type"], process["train"]["pool_datasets"]), ("band", True))
        self.assertTrue(process["model"]["model_kwargs"]["rgba"])

    @posix_only(DATASET_IO)
    def test_ai_toolkit_projects_every_new_0_13_30_diffusion_architecture(self) -> None:
        cases = (
            ("flux3", "black-forest-labs/flux-3-image-klein-base", (), {}),
            ("flux3", "black-forest-labs/flux-3-image-klein-base", ("control.png",), {}),
            ("iris3b", "speridlabs/iris-3b", (), {}),
            ("ming_image", "inclusionAI/Ming-Image-0.1-Design", (), {}),
            ("moss_image", "OpenMOSS-Team/MOSS-Image-SFT", (), {}),
            ("moss_image:base_edit", "OpenMOSS-Team/MOSS-Image-Base", ("a.png", "b.png"), {"model_edit": True}),
            # Turbo trains the distilled checkpoint through a training adapter the author supplies.
            ("qwen_image_2:turbo", "Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", (), {
                "native_config": {"model": {"assistant_lora_path": "ostris/qwen_image_2_turbo_training_adapter/qwen_image_2_turbo_training_adapter_v1.safetensors"}},
            }),
        )
        for arch, model, controls, authored in cases:
            with self.subTest(arch=arch, controls=len(controls)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = {
                    "id": f"new-{arch.replace(':', '-').replace('_', '-')}",
                    "backend": {"name": "ai-toolkit", "config": {"model_arch": arch, "dataset_config": {}, **authored}},
                    "model": {"base": model}, "datasets": [{"id": "tiny"}],
                    "recipe": {"steps": 1, "seed": 1},
                }
                self._write_ai_toolkit_manifest_projection(run, root, target="sample.png", controls=controls)
                BACKENDS["ai-toolkit"].compile(run, root)
                process = yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                self.assertEqual(process["model"]["arch"], arch)
                self.assertEqual(len(process["datasets"][0].get("control_path") or []), len(controls))
                if arch == "qwen_image_2:turbo":
                    self.assertTrue(process["model"]["model_kwargs"]["turbo"])
                    self.assertEqual(process["model"]["assistant_lora_path"], authored["native_config"]["model"]["assistant_lora_path"])
                if arch.startswith("moss_image"):
                    self.assertEqual(process["model"].get("model_kwargs", {}).get("edit", False), bool(controls))
        # Kura projects one flux3 reference (no run has verified several), so a second one is refused.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = {"id": "new-flux3-two", "backend": {"name": "ai-toolkit", "config": {"model_arch": "flux3", "dataset_config": {}}},
                   "model": {"base": "black-forest-labs/flux-3-image-klein-base"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1}}
            with self.assertRaisesRegex(ValueError, "no verified AI-Toolkit projection profile matches architecture='flux3'"):
                self._write_ai_toolkit_manifest_projection(run, root, target="sample.png", controls=("a.png", "b.png"))
                BACKENDS["ai-toolkit"].compile(run, root)

    def test_ai_toolkit_model_edit_applies_where_the_requirements_name_it(self) -> None:
        # One owner: the surface condition is derived from the requirement tables' keys.
        from kura.backends.ai_toolkit import AI_TOOLKIT_PROFILE_REQUIREMENTS
        from kura.backends.ai_toolkit_baseline import trained_arch
        from kura.backends.registry import AI_TOOLKIT_SURFACE

        [condition] = [item for item in AI_TOOLKIT_SURFACE.conditions if item.field == "model_edit"]
        [[(selector, values)]] = condition.when_any
        self.assertEqual(selector, "model_arch")
        required = {arch for (arch, _profile), fields in AI_TOOLKIT_PROFILE_REQUIREMENTS.items() if "model_edit" in fields}
        self.assertEqual({trained_arch(value) for value in values}, required)
        self.assertEqual(required, {"krea2", "moss_image"})
        for arch in ("flux3", "ming_image", "iris3b"):
            with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "model_edit"):
                validate_backend_config({"backend": {"name": "ai-toolkit", "config": {"model_arch": arch, "model_edit": True}}, "model": {"base": "example/model"}})

    @posix_only(DATASET_IO)
    def test_ai_toolkit_refuses_the_0_13_30_architectures_outside_their_profiles(self) -> None:
        def compiled(arch: str, base: str, controls: tuple[str, ...], **authored):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = {
                    "id": f"refused-{arch.replace(':', '-').replace('_', '-')}",
                    "backend": {"name": "ai-toolkit", "config": {"model_arch": arch, "dataset_config": {}, **authored}},
                    "model": {"base": base}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
                }
                self._write_ai_toolkit_manifest_projection(run, root, target="sample.png", controls=controls)
                BACKENDS["ai-toolkit"].compile(run, root)
                return yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]

        # MOSS-Image reads references only in edit mode, so references without model_edit are refused.
        with self.assertRaisesRegex(ValueError, "model_edit"):
            compiled("moss_image", "OpenMOSS-Team/MOSS-Image-SFT", ("a.png",))
        # The base_edit entry does not turn edit mode on by itself.
        process = compiled("moss_image:base_edit", "OpenMOSS-Team/MOSS-Image-Base", ())
        self.assertNotIn("edit", process["model"].get("model_kwargs", {}))
        # Ming-Image trains from a target alone.
        with self.assertRaisesRegex(ValueError, "no verified AI-Toolkit projection profile matches architecture='ming_image'"):
            compiled("ming_image", "inclusionAI/Ming-Image-0.1-Design", ("a.png",))
        # model.base never chooses the entry: the Base model under moss_image names its own entry.
        with self.assertRaisesRegex(ValueError, "model.base 'OpenMOSS-Team/MOSS-Image-Base' is the model of UI entry moss_image:base, moss_image:base_edit, but model_arch 'moss_image' selects entry 'moss_image'"):
            compiled("moss_image", "OpenMOSS-Team/MOSS-Image-Base", ())

    @posix_only(DATASET_IO)
    def test_ai_toolkit_projects_the_0_13_32_moss_image_edit_entry(self) -> None:
        def compiled(base: str, controls: tuple[str, ...], **authored):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = {
                    "id": "moss-image-edit",
                    "backend": {"name": "ai-toolkit", "config": {"model_arch": "moss_image:edit", "dataset_config": {}, **authored}},
                    "model": {"base": base}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
                }
                self._write_ai_toolkit_manifest_projection(run, root, target="sample.png", controls=controls)
                BACKENDS["ai-toolkit"].compile(run, root)
                return yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]

        # The SFT edit entry fills its UI's kv_cache and match_target_res; edit comes from model_edit.
        process = compiled("OpenMOSS-Team/MOSS-Image-SFT", ("a.png", "b.png"), model_edit=True)
        self.assertEqual(process["model"]["arch"], "moss_image:edit")
        self.assertEqual(
            process["model"]["model_kwargs"],
            {"edit": True, "kv_cache": True, "match_target_res": True, "uncond_skip_layers": [22]},
        )
        self.assertEqual((process["model"]["qtype"], process["model"]["quantize"], process["model"]["low_vram"]), ("convrot8", True, True))
        self.assertEqual(len(process["datasets"][0]["control_path"]), 2)
        # The entry does not turn edit mode on by itself.
        process = compiled("OpenMOSS-Team/MOSS-Image-SFT", ())
        self.assertEqual(process["model"]["model_kwargs"], {"kv_cache": True, "match_target_res": True, "uncond_skip_layers": [22]})
        # model.base never chooses the entry: the Base model names the Base entries.
        with self.assertRaisesRegex(ValueError, "model.base 'OpenMOSS-Team/MOSS-Image-Base' is the model of UI entry moss_image:base, moss_image:base_edit, but model_arch 'moss_image:edit' selects entry 'moss_image:edit'"):
            compiled("OpenMOSS-Team/MOSS-Image-Base", ())

    @posix_only(DATASET_IO)
    def test_ai_toolkit_projects_every_new_0_13_18_diffusion_architecture(self) -> None:
        cases = (
            ("qwen_image_2", "Qwen/Qwen-Image-2.1", "sample.png", (), {}),
            ("anima", "circlestone-labs/Anima-Base-v1.0-Diffusers", "sample.png", (), {}),
            ("mageflow", "microsoft/Mage-Flow-Base", "sample.png", (), {}),
            ("mageflow_edit", "microsoft/Mage-Flow-Edit-Base", "sample.png", ("control.png",), {}),
            ("ltx2.5", "Lightricks/LTX-2.5", "sample.mp4", (), {"num_frames": 9, "fps": 24, "do_audio": False}),
            ("minimax_h3", "MiniMaxAI/MiniMax-H3", "sample.mp4", (), {"num_frames": 5, "fps": 24, "do_audio": False}),
            ("minimax_h3_ref2va", "MiniMaxAI/MiniMax-H3", "sample.mp4", ("portrait.png",), {"num_frames": 5, "fps": 24, "do_audio": False}),
        )
        # minimax_h3_vsa gained a pinned UI entry in 0.13.30 (FastH3 8-step V2), so its baseline fills the rest.
        cases = tuple((*case, {}) for case in cases) + (
            ("minimax_h3_vsa", "FastVideo/FastVideo-FastH3-Comfy", "sample.png", (), {}, {}),
        )
        for arch, model, target, controls, dataset_config, authored in cases:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = {
                    "id": f"new-{arch.replace('.', '-')}",
                    "backend": {"name": "ai-toolkit", "config": {
                        "model_arch": arch, "dataset_config": dataset_config, **authored,
                    }},
                    "model": {"base": model}, "datasets": [{"id": "tiny"}],
                    "recipe": {"steps": 1, "seed": 1},
                }
                self._write_ai_toolkit_manifest_projection(
                    run, root, target=target, controls=controls,
                )
                BACKENDS["ai-toolkit"].compile(run, root)
                process = yaml.safe_load(
                    (root / "ai-toolkit.yaml").read_text(encoding="utf-8")
                )["config"]["process"][0]
                self.assertEqual(process["model"]["arch"], arch)
                self.assertEqual(process["model"]["name_or_path"], model)

    @posix_only(DATASET_IO)
    def test_ai_toolkit_new_video_architectures_compile_each_declared_dataset_mode(self) -> None:
        cases = (
            ("ltx2.5", "image-only", "sample.png", {}),
            ("ltx2.5", "video-audio", "sample.mp4", {"num_frames": 9, "fps": 24, "do_audio": True}),
            ("ltx2.5", "first-frame", "sample.mp4", {"num_frames": 9, "fps": 24, "do_i2v": True}),
            ("minimax_h3", "image-only", "sample.png", {}),
            ("minimax_h3", "video-audio", "sample.mp4", {"num_frames": 5, "fps": 24, "do_audio": True}),
            ("minimax_h3", "first-frame", "sample.mp4", {"num_frames": 5, "fps": 24, "do_i2v": True}),
        )
        for arch, mode, target, dataset_config in cases:
            with self.subTest(arch=arch, mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = {
                    "id": f"compile-{arch}-{mode}",
                    "backend": {"name": "ai-toolkit", "config": {
                        "model_arch": arch,
                        "dataset_config": dataset_config,
                        "gradient_checkpointing": False,
                    }},
                    "model": {"base": "example/model"},
                    "datasets": [{"id": "tiny"}],
                    "recipe": {"steps": 1, "seed": 1},
                }
                self._write_ai_toolkit_manifest_projection(run, root, target=target)
                BACKENDS["ai-toolkit"].compile(run, root)
                process = yaml.safe_load(
                    (root / "ai-toolkit.yaml").read_text(encoding="utf-8")
                )["config"]["process"][0]
                projected = process["datasets"][0]
                self.assertEqual(process["model"]["arch"], arch)
                for field, value in dataset_config.items():
                    self.assertEqual(projected[field], value)

    def test_ai_toolkit_rejects_invalid_video_dataset_settings(self) -> None:
        base = {
            "id": "invalid-video", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "minimax_h3", "dataset_config": {},
            }},
            "model": {"base": "Comfy-Org/MiniMax-H3"},
            "datasets": [{"id": "video"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        for field, value, expected in (
            ("num_frames", True, "must be an integer >= 1"),
            ("unknown", 1, "unsupported key"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = deepcopy(base)
                run["backend"]["config"]["dataset_config"] = {field: value}
                with self.assertRaisesRegex(ValueError, expected):
                    BACKENDS["ai-toolkit"].compile(run, Path(directory))

    def test_musubi_escape_hatches_cannot_shadow_declared_fields(self) -> None:
        base = {
            "id": "ambiguous-musubi", "backend": {"name": "musubi-tuner", "config": {
                "architecture": "flux2", "model_version": "klein-base-4b",
                "model_paths": {"dit": "/models/dit", "vae": "/models/vae", "text_encoder": "/models/text"},
            }},
            "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = json.loads(json.dumps(base))
            # Written or not, a field the architecture declares is never set through extra_args.
            run["backend"]["config"].update({"extra_args": ["--weighting_scheme", "none"]})
            with self.assertRaisesRegex(ValueError, r"--weighting_scheme \(backend.config.weighting_scheme\); write the declared field instead"):
                BACKENDS["musubi-tuner"].compile(run, root / "unset")

            run = json.loads(json.dumps(base))
            run["backend"]["config"].update({"blocks_to_swap": 1, "extra_args": ["--blocks_to_swap", "2"]})
            with self.assertRaisesRegex(ValueError, "write the declared field instead"):
                BACKENDS["musubi-tuner"].compile(run, root / "args")

            run = json.loads(json.dumps(base))
            run["backend"]["config"].update({
                "blocks_to_swap": 1,
                "block_swap_h2d_only": True,
                "gradient_checkpointing": True,
                "extra_args": ["--block_swap_h2d_only"],
            })
            with self.assertRaisesRegex(ValueError, "write the declared field instead"):
                BACKENDS["musubi-tuner"].compile(run, root / "h2d")

    @posix_only(DATASET_IO)
    def test_declared_ordinary_values_reach_each_adapter_artifact(self) -> None:
        runs = {
            "ai-toolkit": {
                "id": "surface-ai", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl", "optimizer_type": "adamw8bit", "lr_scheduler": "constant", "gradient_accumulation_steps": 2}},
                "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            },
            "musubi-tuner": {
                "id": "surface-musubi", "backend": {"name": "musubi-tuner", "config": with_required("musubi-tuner", {
                    "architecture": "flux2", "model_version": "klein-base-4b", "optimizer_type": "adamw8bit", "lr_scheduler": "constant", "gradient_accumulation_steps": 2, "batch_size": 1, "resolution": [64, 64], "blocks_to_swap": 1,
                    "model_paths": {"dit": "/models/dit", "vae": "/models/vae", "text_encoder": "/models/text"},
                })}, "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            },
            "sd-scripts": {
                "id": "surface-sd", "backend": {"name": "sd-scripts", "config": with_required("sd-scripts", {
                    "architecture": "sd15", "optimizer_type": "AdamW8bit", "lr_scheduler": "constant", "gradient_accumulation_steps": 2, "model_paths": {"base": "/models/base"},
                    "dataset_config": {"datasets": [{"subsets": [{"dataset_id": "tiny", "num_repeats": 1}]}]},
                })}, "model": {"base": "/models/base"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "datasets" / "tiny" / "images"
            images.mkdir(parents=True)
            (images / "1.png").write_bytes(b"image")
            (images / "1.txt").write_text("caption\n", encoding="utf-8")
            for name, run in runs.items():
                with self.subTest(backend=name):
                    if name == "ai-toolkit":
                        self._write_ai_toolkit_projection(run, root / name)
                    elif name == "musubi-tuner":
                        self._write_musubi_projection(run, root / name)
                    elif name == "sd-scripts":
                        self._write_sd_scripts_projection(run, root / name)
                    spec = BACKENDS[name].compile(run, root / name)
                    if name == "ai-toolkit":
                        compiled = yaml.safe_load((root / "ai-toolkit" / "ai-toolkit.yaml").read_text(encoding="utf-8"))
                        self.assertEqual(compiled["config"]["process"][0]["train"]["optimizer"], "adamw8bit")
                        self.assertEqual(compiled["config"]["process"][0]["train"]["lr_scheduler"], "constant")
                        # The trainer's real accumulation: one optimizer update per step (config_modules.py:460-465).
                        self.assertEqual(compiled["config"]["process"][0]["train"]["gradient_accumulation"], 2)
                        self.assertNotIn("gradient_accumulation_steps", compiled["config"]["process"][0]["train"])
                    else:
                        argv = " ".join(spec["argv"]).lower()
                        self.assertIn("adamw8bit", argv)
                        self.assertIn("lr_scheduler", argv)
                        self.assertIn("gradient_accumulation_steps", argv)
                        if name == "musubi-tuner":
                            self.assertIn("blocks_to_swap", argv)
                            dataset_toml = (root / name / "musubi" / "dataset.toml").read_text(encoding="utf-8")
                            self.assertIn("batch_size = 1", dataset_toml)
                            self.assertIn("resolution = [64, 64]", dataset_toml)


    @posix_only(DATASET_IO)
    def test_ai_toolkit_accumulation_compiles_the_same_on_docker_and_runpod(self) -> None:
        # One compile owns the native accumulation key; both executors run its frozen output and
        # capture training state with the same runner.
        compiled = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "datasets" / "tiny" / "images"
            images.mkdir(parents=True)
            (images / "1.png").write_bytes(b"image")
            (images / "1.txt").write_text("caption\n", encoding="utf-8")
            for executor in ("docker", "runpod"):
                run = {
                    "id": "accum-ai", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl", "gradient_accumulation_steps": 2}},
                    "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 4, "seed": 1},
                    "compute": {"executor": executor},
                }
                self._write_ai_toolkit_projection(run, root / executor)
                spec = BACKENDS["ai-toolkit"].compile(run, root / executor)
                self.assertIn("ai_toolkit_state", " ".join(spec["argv"]))
                compiled[executor] = yaml.safe_load((root / executor / "ai-toolkit.yaml").read_text(encoding="utf-8"))
        self.assertEqual(compiled["docker"], compiled["runpod"])
        train = compiled["docker"]["config"]["process"][0]["train"]
        self.assertEqual((train["gradient_accumulation"], train["steps"]), (2, 4))
        self.assertNotIn("gradient_accumulation_steps", train)

if __name__ == "__main__":
    unittest.main()
