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
import tempfile

import yaml
import unittest

from kura.backends import BACKENDS, BackendAdapter, backend_capabilities, validate_backend_config
from kura.backends.musubi_command import _script_command as musubi_script_command
from kura.cli import cmd_run_capabilities, cmd_run_compile, cmd_run_plan
from kura.init_templates import cmd_init


class BackendSurfaceContractTests(unittest.TestCase):
    def test_every_registered_backend_rejects_unknown_top_level_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, adapter in BACKENDS.items():
                with self.subTest(backend=name):
                    run = {"backend": {"name": name, "config": {"kura_unknown_sentinel": True}}}
                    with self.assertRaisesRegex(ValueError, "kura_unknown_sentinel"):
                        adapter.compile(run, Path(directory) / name, Path(directory), False)

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
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
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
                "control_subdir": {"type": "relative-path"},
                "do_audio": {"type": "boolean"},
                "fps": {"type": "integer", "minimum": 1},
                "num_frames": {"type": "integer", "minimum": 1},
            },
        )
        model_arch_choices = backend_capabilities("ai-toolkit")["config_value_choices"]["model_arch"]
        self.assertIn("sd1", model_arch_choices)
        self.assertIn("qwen_image_2", model_arch_choices)
        self.assertNotIn("sd15", model_arch_choices)

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
            "model_bundle": "minimax-h3-pruned-int8",
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

    def test_musubi_capabilities_expose_typed_minimax_h3_dataset_contract(self) -> None:
        capabilities = backend_capabilities("musubi-tuner")

        self.assertEqual(
            capabilities["conditional_fields"]["h3_dataset_config"]["when_any"],
            [{"architecture": ["minimax_h3", "minimaxh3"]}],
        )
        self.assertEqual(
            capabilities["nested_config_fields"]["h3_dataset_config.datasets[]"]["source"]["type"],
            "enum:image_directory|image_jsonl|video_directory|video_jsonl",
        )
        self.assertEqual(
            capabilities["nested_config_fields"]["h3_dataset_config.datasets[]"]["fp_1f_target_index"]["minimum"],
            0,
        )
        self.assertEqual(
            capabilities["nested_config_fields"]["h3_dataset_config.general"]["batch_size"],
            {"type": "integer", "minimum": 1, "maximum": 1},
        )

    def test_musubi_capabilities_expose_krea2_memory_accommodations(self) -> None:
        capabilities = backend_capabilities("musubi-tuner")

        self.assertEqual(
            capabilities["conditional_fields"]["convrot_int8"]["when_any"],
            [{"architecture": ["krea2", "krea_2"]}],
        )
        self.assertEqual(
            capabilities["conditional_fields"]["convrot_int8_bwd"]["when_any"],
            [{"architecture": ["krea2", "krea_2"], "convrot_int8": [True]}],
        )
        self.assertEqual(
            capabilities["conditional_fields"]["gradient_checkpointing_cpu_offload"]["when_any"],
            [{"architecture": ["krea2", "krea_2"], "gradient_checkpointing": [True]}],
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
                        BACKENDS[name].compile(run, root / name, root, False)

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
                    config = {condition.field: "sentinel"}
                    for selector, allowed in first_clause:
                        config[selector] = allowed[0]
                    candidates = [selector for selector, _ in first_clause if selector not in conditional_names]
                    self.assertTrue(candidates, f"{name}.{condition.field} needs an independently selectable condition")
                    config[candidates[-1]] = "not-a-supported-selector"
                    run = {"backend": {"name": name, "config": config}}
                    with self.assertRaisesRegex(ValueError, rf"{condition.field}.*not applicable"):
                        validate_backend_config(run)

    def test_capabilities_cli_has_machine_readable_output(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cmd_run_capabilities(type("Args", (), {"backend": "musubi-tuner", "json": True})()), 0)
        self.assertIn('"optimizer_type"', output.getvalue())
        self.assertIn('"validation": "unverified"', output.getvalue())

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

    def test_musubi_short_resume_saves_state_at_the_derived_endpoint(self) -> None:
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

        self.assertIn("--max_train_steps 1000", script)
        self.assertIn("--save_every_n_steps 1000", script)
        self.assertIn("--save_last_n_steps_state 1000", script)

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

    def test_ai_toolkit_raw_override_cannot_shadow_owned_model_architecture(self) -> None:
        run = {
            "id": "ambiguous-ai", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "sdxl", "native_config": {"model": {"arch": "flux"}},
            }},
            "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicates backend.config.model_arch"):
                BACKENDS["ai-toolkit"].compile(run, Path(directory), Path(directory), False)

    def test_ai_toolkit_rejects_sd15_before_compiling_native_config(self) -> None:
        run = {
            "id": "invalid-sd15", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd15"}},
            "model": {"base": "hf-internal-testing/tiny-stable-diffusion-pipe"},
            "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(ValueError, "sd15.*sd1"):
                BACKENDS["ai-toolkit"].compile(run, destination, destination, False)
            self.assertFalse((destination / "ai-toolkit.yaml").exists())

    def test_ai_toolkit_selector_catalog_is_tied_to_the_declared_image_pin(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        base_pin = "ostris/aitoolkit:0.13.18@sha256:9bc99d51efc5b6c38a951b3bf8547bda0f9db58abeb75573548d449f82b34bcc"
        runtime_pin = "nomadoor/kura-ai-toolkit@sha256:9aa6861b0f54f24f0ebad07b6018b431e8c2403d27eed9233595951b466dbc3a"
        dockerfile = (repository / "docker" / "ai-toolkit" / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(dockerfile.splitlines()[:2], [f"ARG AI_TOOLKIT_IMAGE={base_pin}", "FROM ${AI_TOOLKIT_IMAGE}"])

        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cmd_init(argparse.Namespace()), 0)
            finally:
                os.chdir(previous)
            workspace = yaml.safe_load((root / "workspace.yaml").read_text(encoding="utf-8"))
            self.assertEqual(workspace["runpod"]["default_image"]["ai-toolkit"], runtime_pin)
            generated_dockerfile = (root / "docker" / "ai-toolkit" / "Dockerfile").read_text(encoding="utf-8")
            self.assertEqual(generated_dockerfile.splitlines()[:2], [f"ARG AI_TOOLKIT_IMAGE={runtime_pin}", "FROM ${AI_TOOLKIT_IMAGE}"])

    def test_ai_toolkit_accepts_registered_sd1_without_rewriting_it(self) -> None:
        run = {
            "id": "registered-sd1", "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd1"}},
            "model": {"base": "hf-internal-testing/tiny-stable-diffusion-pipe"},
            "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            BACKENDS["ai-toolkit"].compile(run, destination, destination, False)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
            self.assertEqual(process["model"]["arch"], "sd1")

    def test_ai_toolkit_preserves_upstream_arch_tag_and_flex1_alias(self) -> None:
        for arch in ("sd1:portrait", "flex1"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                run = {
                    "id": "upstream-alias", "backend": {"name": "ai-toolkit", "config": {"model_arch": arch}},
                    "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}],
                    "recipe": {"steps": 1, "seed": 1},
                }
                BACKENDS["ai-toolkit"].compile(run, destination, destination, False)
                process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                self.assertEqual(process["model"]["arch"], arch)

    def test_ai_toolkit_cli_reports_sd15_correction_before_dataset_resolution(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
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

    def test_ai_toolkit_native_model_arch_remains_an_unvalidated_escape_hatch(self) -> None:
        run = {
            "id": "custom-arch", "backend": {"name": "ai-toolkit", "config": {
                "native_config": {"model": {"arch": "custom_extension_arch"}},
            }},
            "model": {"base": "custom/model"}, "datasets": [{"id": "tiny"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            BACKENDS["ai-toolkit"].compile(run, destination, destination, False)
            process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
            self.assertEqual(process["model"]["arch"], "custom_extension_arch")

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
                BACKENDS["ai-toolkit"].compile(run, Path(directory), Path(directory), False)

    def test_ai_toolkit_sdxl_evidence_settings_preserve_legacy_process_semantics(self) -> None:
        """The two migrated 2026-07-12 smokes differed only by executor."""
        ordinary = {
            "model_arch": "sdxl", "network_dim": 1, "network_alpha": 1,
            "save_every_n_steps": 1, "save_last_n_steps": 1,
            "dataset_folder": "/workspace/datasets/flux2-klein-tiny/images",
            "resolution": [256, 256], "learning_rate": 1.0e-6, "batch_size": 1,
            "gradient_accumulation_steps": 1, "gradient_checkpointing": True,
            "mixed_precision": "bf16", "optimizer_type": "adamw8bit", "low_vram": True,
        }
        for executor, expected_cwd in (("docker", "/opt/ai-toolkit"), ("runpod", "/app/ai-toolkit")):
            with self.subTest(executor=executor), tempfile.TemporaryDirectory() as directory:
                run_id = f"evidence-sdxl-{executor}"
                run = {
                    "id": run_id, "backend": {"name": "ai-toolkit", "config": ordinary},
                    "model": {"base": "stabilityai/stable-diffusion-xl-base-1.0"},
                    "datasets": [{"id": "flux2-klein-tiny"}],
                    "recipe": {"steps": 1, "seed": 1}, "compute": {"executor": executor},
                }
                destination = Path(directory) / "ai-toolkit"
                command = BACKENDS["ai-toolkit"].compile(run, destination, Path(directory), False)
                process = yaml.safe_load((destination / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                self.assertEqual(process, {
                    "type": "sd_trainer", "training_folder": f"/workspace/runs/{run_id}/outputs", "device": "cuda:0",
                    "network": {"type": "lora", "linear": 1, "linear_alpha": 1},
                    "save": {"save_every": 1, "max_step_saves_to_keep": 1},
                    "datasets": [{
                        "folder_path": "/workspace/datasets/flux2-klein-tiny/images", "caption_ext": ".txt",
                        "cache_latents_to_disk": True, "resolution": [256, 256],
                    }],
                    "train": {
                        "steps": 1, "train_unet": True, "train_text_encoder": False, "disable_sampling": True,
                        "seed": 1, "lr": 1.0e-6, "optimizer": "adamw8bit", "dtype": "bf16",
                        "batch_size": 1, "gradient_accumulation_steps": 1, "gradient_checkpointing": True,
                    },
                    "model": {
                        "name_or_path": "stabilityai/stable-diffusion-xl-base-1.0", "arch": "sdxl",
                        "quantize": False, "quantize_te": False, "low_vram": True,
                    },
                })
                self.assertEqual(command["cwd"], expected_cwd)
                self.assertEqual(command["argv"][:2], ["python", "-c"])
                self.assertIn(f"/workspace/runs/{run_id}/resolved/ai-toolkit.yaml", command["argv"][3])
                self.assertEqual(command["env"], {"SEED": "1", "MODELS_PATH": "/workspace/cache/ai-toolkit/models"})

    def test_ai_toolkit_projects_typed_video_dataset_settings(self) -> None:
        run = {
            "id": "minimax-video", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "minimax_h3",
                "dataset_folder": "/workspace/datasets/video/videos",
                "dataset_config": {"num_frames": 5, "fps": 24, "do_audio": False},
            }},
            "model": {"base": "Comfy-Org/MiniMax-H3"},
            "datasets": [{"id": "video"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            BACKENDS["ai-toolkit"].compile(run, root, root, False)

            process = yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
        self.assertEqual(process["datasets"], [{
            "folder_path": "/workspace/datasets/video/videos",
            "caption_ext": ".txt",
            "cache_latents_to_disk": True,
            "num_frames": 5,
            "fps": 24,
            "do_audio": False,
        }])

    def test_ai_toolkit_projects_control_subdir_inside_each_owned_dataset(self) -> None:
        run = {
            "id": "qwen-edit", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "qwen_image_2",
                "dataset_config": {"control_subdir": "control"},
            }},
            "model": {"base": "Qwen/Qwen-Image-2.1"},
            "datasets": [{"id": "paired"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            BACKENDS["ai-toolkit"].compile(run, root, root, False)

            process = yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
        self.assertEqual(process["datasets"], [{
            "folder_path": "/workspace/datasets/paired/images",
            "caption_ext": ".txt",
            "cache_latents_to_disk": True,
            "control_path": "/workspace/datasets/paired/control",
        }])

    def test_ai_toolkit_rejects_control_subdir_that_escapes_the_dataset(self) -> None:
        run = {
            "id": "unsafe-edit", "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "mageflow_edit",
                "dataset_config": {"control_subdir": "../outside"},
            }},
            "model": {"base": "microsoft/Mage-Flow-Edit-Base"},
            "datasets": [{"id": "paired"}],
            "recipe": {"steps": 1, "seed": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "relative path inside the dataset"):
                BACKENDS["ai-toolkit"].compile(run, Path(directory), Path(directory), False)

    def test_ai_toolkit_projects_every_new_0_13_18_diffusion_architecture(self) -> None:
        cases = (
            ("qwen_image_2", "Qwen/Qwen-Image-2.1", {}),
            ("anima", "circlestone-labs/Anima-Base-v1.0-Diffusers", {}),
            ("mageflow", "microsoft/Mage-Flow-Base", {}),
            ("mageflow_edit", "microsoft/Mage-Flow-Edit-Base", {"control_subdir": "control"}),
            ("ltx2.5", "Lightricks/LTX-2.5", {"num_frames": 9, "fps": 24, "do_audio": False}),
            ("minimax_h3", "MiniMaxAI/MiniMax-H3", {"num_frames": 5, "fps": 24, "do_audio": False}),
            ("minimax_h3_ref2va", "MiniMaxAI/MiniMax-H3", {"num_frames": 5, "fps": 24, "do_audio": False, "control_subdir": "control"}),
            ("minimax_h3_vsa", "MiniMaxAI/MiniMax-H3", {"num_frames": 5, "fps": 24, "do_audio": False}),
        )
        for arch, model, dataset_config in cases:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = {
                    "id": f"new-{arch.replace('.', '-')}",
                    "backend": {"name": "ai-toolkit", "config": {
                        "model_arch": arch, "dataset_config": dataset_config,
                    }},
                    "model": {"base": model}, "datasets": [{"id": "tiny"}],
                    "recipe": {"steps": 1, "seed": 1},
                }

                BACKENDS["ai-toolkit"].compile(run, root, root, False)

                process = yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                self.assertEqual(process["model"]["arch"], arch)
                self.assertEqual(process["model"]["name_or_path"], model)
                if "control_subdir" in dataset_config:
                    self.assertEqual(process["datasets"][0]["control_path"], "/workspace/datasets/tiny/control")

    def test_ai_toolkit_new_video_architectures_compile_each_declared_dataset_mode(self) -> None:
        cases = (
            ("ltx2.5", "image-only", {}),
            ("ltx2.5", "video-audio", {"num_frames": 9, "fps": 24, "do_audio": True}),
            ("ltx2.5", "conditioned-video", {"num_frames": 9, "fps": 24, "control_subdir": "control"}),
            ("minimax_h3", "image-only", {}),
            ("minimax_h3", "video-audio", {"num_frames": 5, "fps": 24, "do_audio": True}),
            ("minimax_h3", "first-frame-control", {"num_frames": 5, "fps": 24, "control_subdir": "control"}),
        )
        for arch, mode, dataset_config in cases:
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

                BACKENDS["ai-toolkit"].compile(run, root, root, False)

                process = yaml.safe_load((root / "ai-toolkit.yaml").read_text(encoding="utf-8"))["config"]["process"][0]
                projected = process["datasets"][0]
                self.assertEqual(process["model"]["arch"], arch)
                self.assertEqual(projected.get("num_frames"), dataset_config.get("num_frames"))
                self.assertEqual(projected.get("fps"), dataset_config.get("fps"))
                self.assertEqual(projected.get("do_audio"), dataset_config.get("do_audio"))
                expected_control = "/workspace/datasets/tiny/control" if "control_subdir" in dataset_config else None
                self.assertEqual(projected.get("control_path"), expected_control)

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
                    BACKENDS["ai-toolkit"].compile(run, Path(directory), Path(directory), False)

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
            run["backend"]["config"].update({"blocks_to_swap": 1, "extra_args": ["--blocks_to_swap", "2"]})
            with self.assertRaisesRegex(ValueError, "duplicates backend.config.extra_args"):
                BACKENDS["musubi-tuner"].compile(run, root / "args", root, False)

            run = json.loads(json.dumps(base))
            run["backend"]["config"].update({
                "blocks_to_swap": 1,
                "block_swap_h2d_only": True,
                "gradient_checkpointing": True,
                "extra_args": ["--block_swap_h2d_only"],
            })
            with self.assertRaisesRegex(ValueError, "duplicates backend.config.extra_args"):
                BACKENDS["musubi-tuner"].compile(run, root / "h2d", root, False)

            run = json.loads(json.dumps(base))
            run["backend"]["config"].update({"batch_size": 1, "dataset_config": {"general": {"batch_size": 2}}})
            with self.assertRaisesRegex(ValueError, "duplicates backend.config.dataset_config.general.batch_size"):
                BACKENDS["musubi-tuner"].compile(run, root / "dataset", root, False)

    def test_declared_ordinary_values_reach_each_adapter_artifact(self) -> None:
        runs = {
            "ai-toolkit": {
                "id": "surface-ai", "backend": {"name": "ai-toolkit", "config": {"optimizer_type": "adamw8bit", "lr_scheduler": "constant", "gradient_accumulation_steps": 2}},
                "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            },
            "musubi-tuner": {
                "id": "surface-musubi", "backend": {"name": "musubi-tuner", "config": {
                    "architecture": "flux2", "model_version": "klein-base-4b", "optimizer_type": "adamw8bit", "lr_scheduler": "constant", "gradient_accumulation_steps": 2, "batch_size": 1, "resolution": [64, 64], "blocks_to_swap": 1,
                    "model_paths": {"dit": "/models/dit", "vae": "/models/vae", "text_encoder": "/models/text"},
                }}, "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
            },
            "sd-scripts": {
                "id": "surface-sd", "backend": {"name": "sd-scripts", "config": {
                    "architecture": "sd15", "optimizer_type": "AdamW8bit", "lr_scheduler": "constant", "gradient_accumulation_steps": 2, "model_paths": {"base": "/models/base"},
                    "dataset_config": {"datasets": [{"subsets": [{"dataset_id": "tiny"}]}]},
                }}, "model": {"base": "/models/base"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 1, "seed": 1},
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
                    spec = BACKENDS[name].compile(run, root / name, root, False)
                    if name == "ai-toolkit":
                        compiled = yaml.safe_load((root / "ai-toolkit" / "ai-toolkit.yaml").read_text(encoding="utf-8"))
                        self.assertEqual(compiled["config"]["process"][0]["train"]["optimizer"], "adamw8bit")
                        self.assertEqual(compiled["config"]["process"][0]["train"]["lr_scheduler"], "constant")
                        self.assertEqual(compiled["config"]["process"][0]["train"]["gradient_accumulation_steps"], 2)
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


if __name__ == "__main__":
    unittest.main()
