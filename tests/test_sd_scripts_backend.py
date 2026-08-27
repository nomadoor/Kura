from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import tomllib
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from kura.backends import backend_capabilities, validate_backend_config
from kura.backends.sd_scripts import OWNED_FLAGS, command_sd_scripts, compile_sd_scripts, display_sd_scripts
from kura.backends.sd_scripts_datasets import (
    DATASET_KEYS,
    GENERAL_KEYS,
    SUBSET_KEYS,
    _validate_field,
    write_sd_scripts_dataset_config,
)
from kura.backends.sd_scripts_models import requirements_sd_scripts, sd_scripts_model_download_specs
from kura.container_scripts import script_source
from kura.run_commands.plan import _sd_scripts_cache_preflight_report, _sd_scripts_disk_cache_estimate


def base_run(architecture: str = "sd15", mode: str = "lora") -> dict:
    roles = {
        "sd15": {"base": "/models/sd15.safetensors"},
        "sdxl": {"base": "/models/sdxl.safetensors"},
        "flux1": {"dit": "/models/flux.safetensors", "clip_l": "/models/clip.safetensors", "t5xxl": "/models/t5.safetensors", "ae": "/models/ae.safetensors"},
        "anima": {"dit": "/models/anima.safetensors", "qwen3": "/models/qwen3.safetensors", "vae": "/models/vae.safetensors"},
    }
    run = {
        "id": "sd-smoke",
        "recipe": {"steps": 1, "seed": 42},
        "datasets": [{"id": "sample", "digest": "sha256:test"}],
        "backend": {
            "name": "sd-scripts",
            "config": {
                "architecture": architecture,
                "mode": mode,
                "model_paths": roles[architecture],
                "dataset_config": {
                    "general": {"resolution": [512, 512], "caption_extension": ".txt"},
                    "datasets": [{"batch_size": 1, "subsets": [{"dataset_id": "sample", "image_subdir": "images", "num_repeats": 1}]}],
                },
                "network_dim": 8,
                "learning_rate": 0.0001,
                "optimizer_type": "AdamW8bit",
                "mixed_precision": "bf16",
            },
        },
    }
    if mode == "controlnet_lllite":
        run["backend"]["config"].pop("network_dim")
    return run


def write_safetensors(path: Path, keys: list[str], metadata: dict[str, str] | None = None) -> None:
    offset = 0
    header = {"__metadata__": metadata or {}}
    body = b""
    for key in keys:
        header[key] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        body += struct.pack("<f", 0.0)
        offset += 4
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


class SdScriptsBackendTests(unittest.TestCase):
    def test_state_capture_rejects_epoch_save_escape_hatches(self) -> None:
        run = base_run("sd15", "lora")
        run["backend"]["config"]["extra_args"] = ["--save_every_n_epochs", "1"]
        with self.assertRaisesRegex(ValueError, "epoch.*training-state"):
            command_sd_scripts(run)

    def test_state_capture_is_on_by_default_and_resume_uses_logical_target(self) -> None:
        run = base_run("sd15", "lora")
        run["id"] = "derived-run"
        run["backend"]["config"]["output_name"] = "source-run"
        run["recipe"]["steps"] = 100
        run["parent_run"] = "source"
        run["continuation"] = {
            "mode": "resume",
            "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 100, "recipe_sha256": "b" * 64},
            "additional_steps": 50,
            "target_step": 150,
            "restoration_contract": {"level": "best_effort_resume", "restored": ["model", "optimizer", "scheduler", "rng", "global_step", "epoch"], "not_restored": ["exact_dataloader_position"]},
        }
        script = command_sd_scripts(run)["argv"][2]
        self.assertIn("--save_state", script)
        self.assertIn("--save_state_on_train_end", script)
        self.assertIn('"--save_last_n_steps_state","100"', script)
        self.assertIn('"--resume","/workspace/artifacts/training-state/state-1/payload"', script)
        self.assertIn("--skip_until_initial_step", script)
        self.assertIn('"--max_train_steps","150"', script)
        self.assertIn("state-runner.py", script)
        self.assertIn("training state verified", script)
        self.assertIn("/workspace/runs/derived-run/resolved/training-state-source.lock.json", script)
        self.assertIn('"--output_name","derived-run"', script)
        self.assertNotIn('"--output_name","source-run"', script)

    def test_sd_scripts_resume_rejects_unsafe_initial_envelopes(self) -> None:
        for architecture, mode, change, message in (
            ("sd15", "lora", {"lr_scheduler": "cosine"}, "constant scheduler"),
            ("sd15", "lora", {"gradient_accumulation_steps": 2}, "gradient_accumulation_steps=1"),
            ("anima", "lora", {}, "not yet supported"),
            ("anima", "controlnet_lllite", {}, "not yet supported"),
        ):
            with self.subTest(architecture=architecture, mode=mode):
                run = base_run(architecture, mode)
                run["backend"]["config"].update(change)
                run["parent_run"] = "source"
                run["continuation"] = {
                    "mode": "resume",
                    "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 10, "recipe_sha256": "b" * 64},
                    "additional_steps": 5,
                    "target_step": 15,
                    "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
                }
                with self.assertRaisesRegex(ValueError, message):
                    command_sd_scripts(run)

    def test_caption_dropout_is_supported_at_every_upstream_inheritance_level(self) -> None:
        for level in ("general", "dataset", "subset"):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                run = base_run("anima", "controlnet_lllite")
                config = run["backend"]["config"]["dataset_config"]
                target = {
                    "general": config["general"],
                    "dataset": config["datasets"][0],
                    "subset": config["datasets"][0]["subsets"][0],
                }[level]
                target.update(
                    {
                        "caption_dropout_rate": 0.15,
                        "caption_dropout_every_n_epochs": 2,
                        "caption_tag_dropout_rate": 0.1,
                    }
                )

                destination = Path(directory) / "dataset.toml"
                write_sd_scripts_dataset_config(run, destination, workspace=None, strict=False)
                parsed = tomllib.loads(destination.read_text(encoding="utf-8"))
                emitted = {
                    "general": parsed["general"],
                    "dataset": parsed["datasets"][0],
                    "subset": parsed["datasets"][0]["subsets"][0],
                }[level]
                self.assertEqual(emitted["caption_dropout_rate"], 0.15)
                self.assertEqual(emitted["caption_dropout_every_n_epochs"], 2)
                self.assertEqual(emitted["caption_tag_dropout_rate"], 0.1)

    def test_reviewed_nested_dataset_scope_is_explicit_at_each_level(self) -> None:
        subset_native = {
            "num_repeats", "caption_extension", "shuffle_caption", "keep_tokens",
            "color_aug", "flip_aug", "random_crop", "caption_dropout_rate",
            "caption_dropout_every_n_epochs", "caption_tag_dropout_rate",
            "caption_prefix", "caption_suffix", "caption_separator",
            "keep_tokens_separator", "secondary_separator", "enable_wildcard",
            "token_warmup_min", "token_warmup_step", "resize_interpolation", "cache_info",
        }
        dataset_native = {
            "batch_size", "resolution", "enable_bucket", "bucket_no_upscale",
            "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps",
            "network_multiplier", "skip_image_resolution",
        }
        staging_only = {"dataset_id", "image_subdir", "caption_subdir", "conditioning_subdir"}

        self.assertEqual(GENERAL_KEYS, subset_native | dataset_native)
        self.assertEqual(DATASET_KEYS, subset_native | dataset_native)
        self.assertEqual(SUBSET_KEYS, subset_native | staging_only)
        for deliberately_unsupported in ("validation_seed", "validation_split", "custom_attributes"):
            self.assertNotIn(deliberately_unsupported, GENERAL_KEYS | DATASET_KEYS | SUBSET_KEYS)

    def test_nested_dataset_surface_rejects_unknown_invalid_types_and_ranges(self) -> None:
        cases = (
            ("kura_unknown", True, "unsupported key"),
            ("caption_dropout_rate", "0.15", "caption_dropout_rate must be a number"),
            ("caption_dropout_rate", 1.1, "caption_dropout_rate must be between 0 and 1"),
            ("enable_bucket", 1, "enable_bucket must be true or false"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                run = base_run("anima", "controlnet_lllite")
                config = run["backend"]["config"]["dataset_config"]
                target = config["general"] if key == "enable_bucket" else config["datasets"][0]["subsets"][0]
                target[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    write_sd_scripts_dataset_config(
                        run, Path(directory) / "dataset.toml", workspace=None, strict=False
                    )

    def test_caption_dropout_every_n_epochs_zero_is_an_accepted_disabled_value(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        native = run["backend"]["config"]
        native["cache_text_encoder_outputs_to_disk"] = True
        native["dataset_config"]["datasets"][0]["subsets"][0]["caption_dropout_every_n_epochs"] = 0
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.toml"
            write_sd_scripts_dataset_config(run, destination, workspace=None, strict=False)
            parsed = tomllib.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(parsed["datasets"][0]["subsets"][0]["caption_dropout_every_n_epochs"], 0)

    def test_field_validator_enforces_arbitrary_declared_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            _validate_field(1, {"type": "integer", "minimum": 2}, field="x")
        with self.assertRaisesRegex(ValueError, "between 0.25 and 0.75"):
            _validate_field(0.8, {"type": "number", "minimum": 0.25, "maximum": 0.75}, field="x")

    def test_preapproval_backend_validation_uses_the_nested_dataset_contract(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["dataset_config"]["datasets"][0]["subsets"][0]["caption_dropout_rate"] = "0.15"
        with self.assertRaisesRegex(ValueError, "caption_dropout_rate must be a number"):
            validate_backend_config(run)

    def test_preapproval_rejects_unknown_dataset_config_root_keys(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["dataset_config"]["typo_root_key"] = True

        with self.assertRaisesRegex(ValueError, "dataset_config contains unsupported key.*typo_root_key"):
            validate_backend_config(run)

    def test_preapproval_rejects_caption_extensions_without_a_leading_dot(self) -> None:
        for level in ("general", "dataset", "subset"):
            with self.subTest(level=level):
                run = base_run("anima", "controlnet_lllite")
                config = run["backend"]["config"]["dataset_config"]
                target = {
                    "general": config["general"],
                    "dataset": config["datasets"][0],
                    "subset": config["datasets"][0]["subsets"][0],
                }[level]
                target["caption_extension"] = "txt"

                with self.assertRaisesRegex(ValueError, "caption_extension must start with a dot"):
                    validate_backend_config(run)

    def test_anima_lllite_caption_rate_can_be_combined_with_text_encoder_disk_cache(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        native = run["backend"]["config"]
        native["cache_text_encoder_outputs_to_disk"] = True
        native["dataset_config"]["datasets"][0]["subsets"][0]["caption_dropout_rate"] = 0.15
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resolved"
            compile_sd_scripts(run, destination, workspace=None, strict=False)
            parsed = tomllib.loads((destination / "dataset.toml").read_text(encoding="utf-8"))
        self.assertEqual(parsed["datasets"][0]["subsets"][0]["caption_dropout_rate"], 0.15)
        self.assertIn("--cache_text_encoder_outputs_to_disk", command_sd_scripts(run)["argv"][2])

    def test_caption_dropout_is_frozen_for_runtime_dataset_logging(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["dataset_config"]["datasets"][0]["subsets"][0]["caption_dropout_rate"] = 0.15
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.toml"
            lock = write_sd_scripts_dataset_config(run, destination, workspace=None, strict=False)
        self.assertEqual(lock["effective_controls"][0]["caption_dropout_rate"], 0.15)

    def test_default_num_repeats_is_frozen_for_runtime_dataset_logging(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["dataset_config"]["datasets"][0]["subsets"][0].pop("num_repeats")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.toml"
            lock = write_sd_scripts_dataset_config(run, destination, workspace=None, strict=False)
            parsed = tomllib.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(parsed["datasets"][0]["subsets"][0]["num_repeats"], 1)
        self.assertEqual(lock["effective_controls"][0]["num_repeats"], 1)

    def test_dynamic_caption_controls_rejected_when_text_encoder_cache_cannot_preserve_them(self) -> None:
        for key, value in (("caption_dropout_every_n_epochs", 2), ("caption_tag_dropout_rate", 0.1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                run = base_run("anima", "controlnet_lllite")
                native = run["backend"]["config"]
                native["cache_text_encoder_outputs_to_disk"] = True
                native["dataset_config"]["datasets"][0]["subsets"][0][key] = value
                with self.assertRaisesRegex(ValueError, rf"{key}.*text-encoder cache"):
                    compile_sd_scripts(run, Path(directory) / "resolved", workspace=None, strict=False)

    def test_sd_scripts_capabilities_expose_nested_dataset_contract(self) -> None:
        nested = backend_capabilities("sd-scripts")["nested_config_fields"]
        for path in (
            "dataset_config.general",
            "dataset_config.datasets[]",
            "dataset_config.datasets[].subsets[]",
        ):
            self.assertIn("caption_dropout_rate", nested[path])
            self.assertEqual(nested[path]["caption_dropout_rate"]["type"], "number")
        self.assertIn("network_multiplier", nested["dataset_config.datasets[]"])
        self.assertIn("caption_prefix", nested["dataset_config.datasets[].subsets[]"])
        self.assertNotIn("validation_split", nested["dataset_config.datasets[]"])

    def test_plan_display_exposes_effective_bucket_and_caption_settings(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        config = run["backend"]["config"]["dataset_config"]
        config["general"].update({"enable_bucket": True, "bucket_no_upscale": True})
        config["datasets"][0]["subsets"][0].update(
            {
                "caption_dropout_rate": 0.15,
                "flip_aug": True,
                "color_aug": False,
                "random_crop": True,
                "resize_interpolation": "lanczos",
                "cache_info": True,
            }
        )

        dataset = display_sd_scripts(run)["dataset_config"]["datasets"][0]

        self.assertEqual(dataset["batch_size"], 1)
        self.assertTrue(dataset["bucket"]["enable_bucket"])
        self.assertTrue(dataset["bucket"]["bucket_no_upscale"])
        self.assertEqual(dataset["subsets"][0]["caption"]["caption_dropout_rate"], 0.15)
        self.assertNotIn("caption_extension", dataset["subsets"][0]["caption"])
        self.assertEqual(
            dataset["subsets"][0]["augmentation"],
            {"color_aug": False, "flip_aug": True, "random_crop": True, "resize_interpolation": "lanczos"},
        )
        self.assertEqual(dataset["subsets"][0]["cache"], {"cache_info": True})

    def test_top_level_dataset_summary_does_not_choose_one_of_conflicting_datasets(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        datasets = run["backend"]["config"]["dataset_config"]["datasets"]
        second = deepcopy(datasets[0])
        second["batch_size"] = 2
        second["resolution"] = [768, 768]
        datasets.append(second)

        display = display_sd_scripts(run)

        self.assertIsNone(display["batch_size"])
        self.assertIsNone(display["resolution"])
        self.assertEqual([item["batch_size"] for item in display["dataset_config"]["datasets"]], [1, 2])

    def test_model_downloads_always_include_selected_primary_filename(self) -> None:
        run = base_run()
        run["backend"]["config"].pop("model_paths")
        run["backend"]["config"]["model_downloads"] = {
            "base": {"repo": "owner/model", "filename": "primary.safetensors", "filenames": ["companion.json"]}
        }

        specs, paths = sd_scripts_model_download_specs(run)

        self.assertEqual([item["filename"] for item in specs], ["primary.safetensors", "companion.json"])
        self.assertTrue(paths["base"].endswith("/primary.safetensors"))

    def test_dataset_config_rejects_missing_declared_dataset_and_invalid_caption_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "dataset.toml"
            run = base_run()
            run["datasets"] = []
            run["backend"]["config"]["dataset_config"]["datasets"][0]["subsets"][0].pop("dataset_id")
            with self.assertRaisesRegex(ValueError, "at least one declared dataset"):
                write_sd_scripts_dataset_config(run, destination, workspace=root, strict=False)

            run = base_run()
            run["backend"]["config"]["dataset_config"]["general"]["caption_extension"] = "txt"
            images = root / "datasets" / "sample" / "images"
            images.mkdir(parents=True)
            (images / "one.png").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "must start with a dot"):
                write_sd_scripts_dataset_config(run, destination, workspace=root, strict=True)

    def test_disk_cache_estimate_rejects_boolean_and_non_finite_values(self) -> None:
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value):
                run = base_run()
                run["backend"]["config"].update({"cache_latents_to_disk": True, "disk_cache_estimate_gb": value})
                self.assertEqual(_sd_scripts_disk_cache_estimate(run)["status"], "unknown")

    def test_tier1_entrypoints_and_model_roles(self) -> None:
        expected = {
            ("sd15", "lora"): ("train_network.py", "networks.lora"),
            ("sdxl", "lora"): ("sdxl_train_network.py", "networks.lora"),
            ("flux1", "lora"): ("flux_train_network.py", "networks.lora_flux"),
            ("anima", "lora"): ("anima_train_network.py", "networks.lora_anima"),
            ("anima", "controlnet_lllite"): ("anima_train_control_net_lllite.py", "--lllite_target_layers self_attn_q"),
        }
        for selector, fragments in expected.items():
            with self.subTest(selector=selector):
                script = command_sd_scripts(base_run(*selector))["argv"][2]
                for fragment in fragments:
                    self.assertIn(fragment, script)
                if selector == ("anima", "lora"):
                    self.assertIn('"--max_train_steps", "1"', script)
                    self.assertIn('"--mixed_precision", "bf16"', script)
                    self.assertIn('"--gradient_accumulation_steps", "1"', script)
                elif selector[0] != "anima":
                    self.assertIn('"--max_train_steps","1"', script)
                    self.assertIn('"--mixed_precision","bf16"', script)
                    self.assertIn('"--gradient_accumulation_steps","1"', script)
                else:
                    self.assertIn("--max_train_steps 1", script)
                    self.assertIn("--mixed_precision bf16", script)
                    self.assertIn("--gradient_accumulation_steps 1", script)

    def test_compile_writes_two_level_toml_and_frozen_stage_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            images = workspace / "datasets" / "sample" / "images"
            images.mkdir(parents=True)
            (images / "one.png").write_bytes(b"png")
            (images / "one.txt").write_text("caption", encoding="utf-8")
            destination = workspace / "runs" / "sd-smoke" / "resolved" / "sd-scripts"

            compile_sd_scripts(base_run(), destination, workspace=workspace, strict=True)

            toml = (destination / "dataset.toml").read_text(encoding="utf-8")
            lock = json.loads((destination / "dataset-stage.lock.json").read_text(encoding="utf-8"))
            self.assertTrue((destination / "state-runner.py").is_file())
        self.assertIn("[[datasets]]", toml)
        self.assertIn("[[datasets.subsets]]", toml)
        self.assertIn('image_dir = "/workspace/runs/sd-smoke/cache/sd-scripts/datasets/000-000/images"', toml)
        self.assertIn("num_repeats = 1", toml)
        self.assertNotIn("/workspace/datasets/", toml)
        self.assertEqual(len(lock["files"]), 2)
        self.assertTrue(all(item["identity"]["sha256"] for item in lock["files"]))

    def test_dataset_level_caption_extension_controls_staged_caption_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            images = workspace / "datasets" / "sample" / "images"
            images.mkdir(parents=True)
            (images / "one.png").write_bytes(b"png")
            (images / "one.caption").write_text("caption", encoding="utf-8")
            run = base_run()
            config = run["backend"]["config"]["dataset_config"]
            config["general"].pop("caption_extension")
            config["datasets"][0]["caption_extension"] = ".caption"
            destination = workspace / "resolved"

            compile_sd_scripts(run, destination, workspace=workspace, strict=True)

            lock = json.loads((destination / "dataset-stage.lock.json").read_text(encoding="utf-8"))
            staged_sources = {item["source"] for item in lock["files"]}
        self.assertIn("datasets/sample/images/one.caption", staged_sources)

    def test_staging_keeps_shared_dataset_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "datasets" / "sample" / "images" / "one.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"png")
            lock = {
                "stage_root": "runs/x/cache/sd-scripts/datasets",
                "files": [{"source": "datasets/sample/images/one.png", "destination": "runs/x/cache/sd-scripts/datasets/000/images/one.png", "identity": {"size_bytes": 3, "sha256": __import__("hashlib").sha256(b"png").hexdigest()}}],
            }
            lock_path = workspace / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            before = sorted(path.relative_to(workspace / "datasets").as_posix() for path in (workspace / "datasets").rglob("*") if path.is_file())
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_dataset_stage.py"), namespace)
            with patch("sys.argv", ["stage", str(lock_path), str(workspace)]):
                namespace["main"]()
            staged = workspace / "runs" / "x" / "cache" / "sd-scripts" / "datasets" / "000" / "images" / "one.png"
            staged.with_suffix(".npz").write_bytes(b"cache")
            staged_is_symlink = staged.is_symlink()
            after = sorted(path.relative_to(workspace / "datasets").as_posix() for path in (workspace / "datasets").rglob("*") if path.is_file())
        self.assertTrue(staged_is_symlink)
        self.assertEqual(before, after)

    def test_staging_logs_frozen_effective_dataset_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            lock_path = workspace / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "stage_root": "runs/x/cache/sd-scripts/datasets",
                        "files": [],
                        "effective_controls": [
                            {"dataset_index": 0, "subset_index": 0, "caption_dropout_rate": 0.15}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_dataset_stage.py"), namespace)
            output = io.StringIO()
            with patch("sys.argv", ["stage", str(lock_path), str(workspace)]), contextlib.redirect_stdout(output):
                namespace["main"]()
        self.assertIn("dataset 0 subset 0 caption_dropout_rate: 0.15", output.getvalue())

    def test_staging_rejects_a_changed_frozen_dataset_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "datasets" / "sample" / "images" / "one.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"changed")
            lock = {"stage_root": "runs/x/cache/sd-scripts/datasets", "files": [{"source": "datasets/sample/images/one.png", "destination": "runs/x/cache/sd-scripts/datasets/000/images/one.png", "identity": {"size_bytes": 3, "sha256": "0" * 64}}]}
            lock_path = workspace / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_dataset_stage.py"), namespace)
            with patch("sys.argv", ["stage", str(lock_path), str(workspace)]), self.assertRaisesRegex(SystemExit, "size changed"):
                namespace["main"]()

    def test_staging_rejects_destructive_stage_root_before_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset = workspace / "datasets" / "keep"
            dataset.mkdir(parents=True)
            marker = dataset / "one.png"
            marker.write_bytes(b"keep")
            lock_path = workspace / "lock.json"
            lock_path.write_text(json.dumps({"stage_root": "datasets", "files": []}), encoding="utf-8")
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_dataset_stage.py"), namespace)
            with patch("sys.argv", ["stage", str(lock_path), str(workspace)]), self.assertRaisesRegex(SystemExit, "run-scoped"):
                namespace["main"]()
            self.assertEqual(marker.read_bytes(), b"keep")

    def test_paired_dataset_requires_matching_stems(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["dataset_config"]["datasets"][0]["subsets"][0]["conditioning_subdir"] = "conditioning"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            images = workspace / "datasets" / "sample" / "images"
            conditions = workspace / "datasets" / "sample" / "conditioning"
            images.mkdir(parents=True)
            conditions.mkdir()
            (images / "one.png").write_bytes(b"png")
            (conditions / "two.png").write_bytes(b"png")
            with self.assertRaisesRegex(ValueError, "stems do not match"):
                compile_sd_scripts(run, workspace / "resolved", workspace=workspace, strict=True)

    def test_anima_publication_converts_all_checkpoints_and_has_non_output_recovery(self) -> None:
        script = command_sd_scripts(base_run("anima", "lora"))["argv"][2]
        self.assertIn("networks/convert_anima_lora_to_comfy.py", script)
        self.assertIn("/recovery/sd-scripts/anima-native", script)
        self.assertIn("/cache/sd-scripts/converted-output", script)
        self.assertIn("sd-scripts Anima publication failed", script)
        self.assertIn("-step[0-9]{8}", script)
        self.assertNotIn("/outputs/native", script)

    def test_anima_publication_script_publishes_final_and_every_step_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            staging = root / "staging"
            outputs = root / "outputs"
            recovery = root / "recovery"
            native.mkdir()
            keys = ["layer.lora_down.weight", "layer.lora_up.weight"]
            for name in ("vivi-step00000100.safetensors", "vivi-step00000200.safetensors", "vivi.safetensors"):
                write_safetensors(native / name, keys)
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_publish_anima.py"), namespace)

            def convert(command, check):
                self.assertTrue(check)
                Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())

            spec = {
                "native_dir": str(native),
                "staging_dir": str(staging),
                "output_dir": str(outputs),
                "recovery_dir": str(recovery),
                "output_name": "vivi",
                "converter": "converter.py",
            }
            with patch("subprocess.run", side_effect=convert), patch("sys.argv", ["publish", json.dumps(spec)]):
                namespace["main"]()

            self.assertEqual(
                sorted(path.name for path in outputs.iterdir()),
                ["vivi-step00000100.safetensors", "vivi-step00000200.safetensors", "vivi.safetensors"],
            )
            self.assertFalse(recovery.exists())
            write_safetensors(outputs / "vivi.safetensors", keys, {"regenerated": "metadata"})
            with patch("subprocess.run", side_effect=convert), patch("sys.argv", ["publish", json.dumps(spec)]):
                namespace["main"]()
            self.assertEqual(
                sorted(path.name for path in outputs.iterdir()),
                ["vivi-step00000100.safetensors", "vivi-step00000200.safetensors", "vivi.safetensors"],
            )

    def test_anima_publication_failure_recovers_every_native_checkpoint_without_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            keys = ["layer.lora_down.weight", "layer.lora_up.weight"]
            names = ["vivi-step00000100.safetensors", "vivi-step00000200.safetensors", "vivi.safetensors"]
            for name in names:
                write_safetensors(native / name, keys)
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_publish_anima.py"), namespace)
            spec = {
                "native_dir": str(native),
                "staging_dir": str(root / "staging"),
                "output_dir": str(root / "outputs"),
                "recovery_dir": str(root / "recovery"),
                "output_name": "vivi",
                "converter": "converter.py",
            }
            with patch("subprocess.run", side_effect=RuntimeError("conversion failed")), patch("sys.argv", ["publish", json.dumps(spec)]):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    namespace["main"]()

            self.assertEqual(sorted(path.name for path in (root / "recovery").iterdir()), names)
            self.assertEqual(list((root / "outputs").iterdir()), [])

    def test_anima_training_wrapper_publishes_stable_checkpoint_before_training_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            keys = ["layer.lora_down.weight", "layer.lora_up.weight"]
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_publish_anima.py"), namespace)

            class Child:
                def __init__(self):
                    self.polls = 0

                def poll(self):
                    self.polls += 1
                    return None if self.polls <= 3 else 0

                def wait(self, timeout=None):
                    return 0

                def send_signal(self, signum):
                    raise AssertionError(signum)

                def terminate(self):
                    raise AssertionError("unexpected terminate")

            sleeps = 0

            def advance(_seconds):
                nonlocal sleeps
                sleeps += 1
                if sleeps == 1:
                    write_safetensors(native / "vivi-step00000100.safetensors", keys)
                    payload = (native / "vivi-step00000100.safetensors").read_bytes()
                    (native / "vivi-step00000100.safetensors").write_bytes(payload[:-2])
                elif sleeps == 2:
                    write_safetensors(native / "vivi-step00000100.safetensors", keys)
                elif sleeps == 3:
                    write_safetensors(native / "vivi.safetensors", keys)

            converted = []

            def convert(command, check):
                self.assertTrue(check)
                converted.append(Path(command[-2]).name)
                Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())

            spec = {
                "native_dir": str(native),
                "staging_dir": str(root / "staging"),
                "output_dir": str(root / "outputs"),
                "recovery_dir": str(root / "recovery"),
                "output_name": "vivi",
                "converter": "converter.py",
                "train_argv": ["train"],
            }
            with patch("subprocess.Popen", return_value=Child()), patch("subprocess.run", side_effect=convert), patch("time.sleep", side_effect=advance):
                self.assertEqual(namespace["train_and_publish"](spec), 0)

            self.assertEqual(converted, ["vivi-step00000100.safetensors", "vivi.safetensors"])
            self.assertEqual(
                sorted(path.name for path in (root / "outputs").iterdir()),
                ["vivi-step00000100.safetensors", "vivi.safetensors"],
            )

    def test_anima_publication_rejects_header_complete_but_tensor_data_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.safetensors"
            write_safetensors(path, ["layer.lora_down.weight", "layer.lora_up.weight"])
            path.write_bytes(path.read_bytes()[:-1])
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_publish_anima.py"), namespace)

            with self.assertRaisesRegex(RuntimeError, "incomplete safetensors data"):
                namespace["lora_header"](path)

    def test_anima_training_wrapper_retries_transient_directory_listing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            final = native / "vivi.safetensors"
            write_safetensors(final, ["layer.lora_down.weight", "layer.lora_up.weight"])
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_publish_anima.py"), namespace)

            child = Mock()
            child.poll.side_effect = [None, 0]
            child.wait.return_value = 0
            listing = Mock(side_effect=[OSError("transient listing failure"), [final]])

            def convert(command, check):
                Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())

            spec = {
                "native_dir": str(native),
                "staging_dir": str(root / "staging"),
                "output_dir": str(root / "outputs"),
                "recovery_dir": str(root / "recovery"),
                "output_name": "vivi",
                "converter": "converter.py",
                "train_argv": ["train"],
            }
            namespace["native_files"] = listing
            with patch("subprocess.Popen", return_value=child), patch("subprocess.run", side_effect=convert), patch("time.sleep"):
                self.assertEqual(namespace["train_and_publish"](spec), 0)

            child.terminate.assert_not_called()
            self.assertTrue((root / "outputs" / "vivi.safetensors").is_file())

    def test_anima_training_wrapper_publishes_and_recovers_checkpoint_after_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            write_safetensors(native / "vivi-step00000100.safetensors", ["layer.lora_down.weight", "layer.lora_up.weight"])
            namespace = {"__name__": "__test__"}
            exec(script_source("sd_scripts_publish_anima.py"), namespace)

            class FailedChild:
                def poll(self):
                    return 7

                def wait(self, timeout=None):
                    return 7

            def convert(command, check):
                Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())

            spec = {
                "native_dir": str(native),
                "staging_dir": str(root / "staging"),
                "output_dir": str(root / "outputs"),
                "recovery_dir": str(root / "recovery"),
                "output_name": "vivi",
                "converter": "converter.py",
                "train_argv": ["train"],
            }
            with patch("subprocess.Popen", return_value=FailedChild()), patch("subprocess.run", side_effect=convert):
                self.assertEqual(namespace["train_and_publish"](spec), 7)

            self.assertTrue((root / "outputs" / "vivi-step00000100.safetensors").is_file())
            self.assertTrue((root / "recovery" / "vivi-step00000100.safetensors").is_file())

    def test_anima_lllite_rejects_unsupported_memory_modes(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["blocks_to_swap"] = 2
        with self.assertRaisesRegex(ValueError, "does not support"):
            command_sd_scripts(run)

    def test_anima_lllite_requires_paired_image_mode(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["lllite_cond_in_channels"] = 4
        with self.assertRaisesRegex(ValueError, "image-only"):
            command_sd_scripts(run)

    def test_anima_text_cache_requires_explicit_dit_only_training(self) -> None:
        run = base_run("anima", "lora")
        run["backend"]["config"]["cache_text_encoder_outputs"] = True
        with self.assertRaisesRegex(ValueError, "network_train_unet_only"):
            command_sd_scripts(run)
        run["backend"]["config"]["network_train_unet_only"] = True
        self.assertIn("--network_train_unet_only", command_sd_scripts(run)["argv"][2])

    def test_flux_local_memory_flags_are_native_and_conflict_checked(self) -> None:
        run = base_run("flux1")
        run["backend"]["config"].update(
            {
                "blocks_to_swap": 18,
                "cache_latents_to_disk": True,
                "cache_text_encoder_outputs_to_disk": True,
                "network_train_unet_only": True,
            }
        )
        command = command_sd_scripts(run)["argv"][2]
        self.assertIn('"--blocks_to_swap","18"', command)
        self.assertIn("--cache_latents_to_disk", command)
        self.assertIn("--cache_text_encoder_outputs_to_disk", command)
        run["backend"]["config"]["cpu_offload_checkpointing"] = True
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            command_sd_scripts(run)

    def test_recipe_and_extra_args_cannot_duplicate_owned_fields(self) -> None:
        run = base_run()
        run["backend"]["config"]["extra_args"] = ["--max_train_steps", "2"]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            command_sd_scripts(run)

    def test_extra_args_cannot_override_memory_or_lllite_fields(self) -> None:
        for flag in ("--blocks_to_swap", "--cache_latents_to_disk", "--lllite_mlp_dim"):
            with self.subTest(flag=flag):
                run = base_run("anima", "controlnet_lllite")
                run["backend"]["config"]["extra_args"] = [flag, "128"]
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    command_sd_scripts(run)

    def test_every_generated_option_is_adapter_owned(self) -> None:
        generated: set[str] = set()
        runs = [base_run("sd15"), base_run("sdxl"), base_run("flux1"), base_run("anima", "lora"), base_run("anima", "controlnet_lllite")]
        runs[0]["backend"]["config"].update({"gradient_checkpointing": True, "fp8_base": True, "cache_latents_to_disk": True})
        runs[1]["backend"]["config"].update({"network_train_unet_only": True, "cache_text_encoder_outputs_to_disk": True, "unet_lr": 1e-4})
        runs[2]["backend"]["config"].update({"blocks_to_swap": 4})
        runs[3]["backend"]["config"].update({"cpu_offload_checkpointing": True, "attn_mode": "sdpa", "qwen_image_vae_2d": True})
        runs[4]["backend"]["config"].update({"lllite_use_aspp": True, "lllite_dropout": 0.1})
        for run in runs:
            script = command_sd_scripts(run)["argv"][2]
            train_lines = [line for line in script.splitlines() if line.startswith("accelerate launch")]
            if train_lines:
                tokens = __import__("shlex").split(train_lines[0])
            else:
                match = __import__("re").search(r'"train_argv": (\[.*?\])}', script)
                self.assertIsNotNone(match)
                tokens = json.loads(match.group(1))
            generated.update(token.split("=", 1)[0] for token in tokens if token.startswith("--"))
        self.assertEqual(generated - OWNED_FLAGS, {"--num_cpu_threads_per_process"})

    def test_boolean_config_values_are_strict(self) -> None:
        for key in ("gradient_checkpointing", "cache_latents_to_disk", "network_train_unet_only", "lllite_use_aspp"):
            with self.subTest(key=key):
                run = base_run("anima", "controlnet_lllite") if key == "lllite_use_aspp" else base_run()
                run["backend"]["config"][key] = 1
                with self.assertRaisesRegex(ValueError, "must be true or false"):
                    command_sd_scripts(run)

    def test_unknown_builtin_config_key_is_rejected(self) -> None:
        run = base_run()
        run["backend"]["config"]["netwrok_dim"] = 8
        with self.assertRaisesRegex(ValueError, "unsupported key.*netwrok_dim"):
            command_sd_scripts(run)

    def test_flow_matching_defaults_are_explicit_and_visible_in_plan_data(self) -> None:
        flux = base_run("flux1")
        flux_command = command_sd_scripts(flux)["argv"][2]
        self.assertIn('"--timestep_sampling","flux_shift"', flux_command)
        self.assertIn('"--guidance_scale","1.0"', flux_command)
        self.assertIn('"--model_prediction_type","raw"', flux_command)
        self.assertEqual(display_sd_scripts(flux)["flow_matching"]["timestep_sampling"], "flux_shift")

        lllite = base_run("anima", "controlnet_lllite")
        lllite_command = command_sd_scripts(lllite)["argv"][2]
        self.assertIn("--timestep_sampling shift", lllite_command)
        self.assertIn("--discrete_flow_shift 3.0", lllite_command)
        self.assertIn("--attn_mode sdpa", lllite_command)
        self.assertEqual(display_sd_scripts(lllite)["flow_matching"]["discrete_flow_shift"], 3.0)
        self.assertEqual(display_sd_scripts(lllite)["memory"]["attn_mode"], "sdpa")

        lllite["backend"]["config"]["attn_mode"] = "flash"
        self.assertEqual(command_sd_scripts(lllite)["argv"][2].count("--attn_mode flash"), 1)

    def test_anima_lllite_rejects_fp8_base(self) -> None:
        run = base_run("anima", "controlnet_lllite")
        run["backend"]["config"]["fp8_base"] = True
        with self.assertRaisesRegex(ValueError, "does not support fp8_base"):
            command_sd_scripts(run)

    def test_invalid_architecture_specific_parameter_is_rejected(self) -> None:
        run = base_run("flux1")
        run["backend"]["config"]["timestep_sampling"] = "typo"
        with self.assertRaisesRegex(ValueError, "timestep_sampling must be one of"):
            command_sd_scripts(run)
        anima = base_run("anima", "controlnet_lllite")
        anima["backend"]["config"]["attn_mode"] = "typo"
        with self.assertRaisesRegex(ValueError, "attn_mode must be one of"):
            command_sd_scripts(anima)
        sd15 = base_run()
        sd15["backend"]["config"]["guidance_scale"] = 1.0
        with self.assertRaisesRegex(ValueError, "only valid for FLUX"):
            command_sd_scripts(sd15)

    def test_explicit_command_is_escape_hatch_and_rejects_secrets(self) -> None:
        run = {"id": "native", "backend": {"name": "sd-scripts", "config": {"command": {"cwd": "/opt/sd-scripts", "argv": ["python", "future.py"], "env": {}}}}}
        self.assertEqual(command_sd_scripts(run)["argv"], ["python", "future.py"])
        secret = deepcopy(run)
        secret["backend"]["config"]["command"]["env"] = {"HF_TOKEN": "bad"}
        with self.assertRaisesRegex(ValueError, "must not contain secrets"):
            command_sd_scripts(secret)

    def test_disk_cache_requires_a_declared_estimate_or_reviewed_exception(self) -> None:
        run = base_run()
        run["backend"]["config"]["cache_latents_to_disk"] = True
        self.assertEqual(_sd_scripts_disk_cache_estimate(run)["status"], "unknown")
        self.assertEqual(_sd_scripts_cache_preflight_report(run)[0]["severity"], "error")
        run["safety"] = {"allow_unknown_disk_cache": True}
        self.assertEqual(_sd_scripts_cache_preflight_report(run)[0]["severity"], "info")
        run["backend"]["config"]["disk_cache_estimate_gb"] = 1.5
        estimate = _sd_scripts_disk_cache_estimate(run)
        self.assertEqual(estimate["status"], "declared-estimate")
        self.assertEqual(estimate["bytes"], int(1.5 * 1024**3))

    def test_checkpoint_retention_is_displayed_as_a_step_window(self) -> None:
        run = base_run()
        run["backend"]["config"].update({"save_every_n_steps": 100, "save_last_n_steps": 1000})
        command = command_sd_scripts(run)["argv"][2]
        checkpoint = display_sd_scripts(run)["checkpoint"]
        self.assertIn('"--save_every_n_steps","100"', command)
        self.assertIn('"--save_last_n_steps","1000"', command)
        self.assertEqual(checkpoint["retention_window_steps"], 1000)
        self.assertNotIn("keep_last", checkpoint)

    def test_declared_requirements_include_download_and_explicit_path(self) -> None:
        run = base_run("flux1")
        run["backend"]["config"]["model_paths"].pop("dit")
        run["backend"]["config"]["model_downloads"] = {"dit": {"repo": "owner/model", "filename": "dit.safetensors", "revision": "a" * 40}}
        requirements = requirements_sd_scripts(run, declared=True)
        by_role = {item["role"]: item for item in requirements}
        self.assertEqual(by_role["dit"]["acquisition"], "kura")
        self.assertEqual(by_role["dit"]["pinning"]["strength"], "immutable-revision")
        self.assertEqual(by_role["ae"]["acquisition"], "local-path")

    def test_empty_explicit_model_path_does_not_suppress_download(self) -> None:
        run = base_run()
        run["backend"]["config"]["model_paths"]["base"] = ""
        run["backend"]["config"]["model_downloads"] = {"base": {"repo": "owner/model", "filename": "base.safetensors", "revision": "a" * 40}}
        script = command_sd_scripts(run)["argv"][2]
        self.assertIn("hf_hub_download", script)
        self.assertIn("/workspace/cache/models/sd-scripts/owner--model/base/base.safetensors", script)

    def test_disk_cache_plan_does_not_treat_integer_as_boolean(self) -> None:
        run = base_run()
        run["backend"]["config"]["cache_latents_to_disk"] = 1
        self.assertFalse(_sd_scripts_disk_cache_estimate(run)["enabled"])

    def test_real_lllite_validator_requires_version_two_metadata(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("sd_scripts_validate.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lllite.safetensors"
            write_safetensors(output, ["lllite_conditioning1.conv1.weight"], {"lllite.version": "2"})
            namespace["validate_output"]({"pattern": str(output), "kind": "anima-lllite"})
            write_safetensors(output, ["lllite_conditioning1.conv1.weight"], {})
            with self.assertRaisesRegex(SystemExit, "lllite.version=2"):
                namespace["validate_output"]({"pattern": str(output), "kind": "anima-lllite"})

    def test_all_tier1_selectors_compile_from_frozen_dataset(self) -> None:
        for architecture, mode in (
            ("sd15", "lora"),
            ("sdxl", "lora"),
            ("flux1", "lora"),
            ("anima", "lora"),
            ("anima", "controlnet_lllite"),
        ):
            with self.subTest(architecture=architecture, mode=mode), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                images = workspace / "datasets" / "sample" / "images"
                images.mkdir(parents=True)
                (images / "one.png").write_bytes(b"png")
                (images / "one.txt").write_text("caption", encoding="utf-8")
                run = base_run(architecture, mode)
                if mode == "controlnet_lllite":
                    conditions = workspace / "datasets" / "sample" / "conditioning"
                    conditions.mkdir()
                    (conditions / "one.png").write_bytes(b"condition")
                    run["backend"]["config"]["dataset_config"]["datasets"][0]["subsets"][0]["conditioning_subdir"] = "conditioning"
                destination = workspace / "runs" / run["id"] / "resolved" / "sd-scripts"
                compile_sd_scripts(run, destination, workspace=workspace, strict=True)
                self.assertTrue((destination / "dataset.toml").is_file())
                self.assertTrue((destination / "dataset-stage.lock.json").is_file())


if __name__ == "__main__":
    unittest.main()
