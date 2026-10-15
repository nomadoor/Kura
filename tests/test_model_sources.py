"""Model sources use one key per meaning, and a malformed entry is refused rather than dropped."""

from __future__ import annotations

import unittest

from kura.backends.musubi_models import _musubi_model_expectations
from kura.comfyui_models import resolve_model_specs
from kura.workspace import migrate_workspace_config, validate_workspace_config

WORKFLOW = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}}}


class ModelSourceTests(unittest.TestCase):
    def test_a_comfyui_model_entry_takes_repo_or_url_and_its_keys_by_one_name(self) -> None:
        specs, unknown = resolve_model_specs(WORKFLOW, {"checkpoints": {"model.safetensors": {"repo": "org/model", "filename": "a/model.safetensors"}}})
        self.assertEqual(unknown, [])
        self.assertEqual((specs[0]["repo"], specs[0]["filename"], specs[0]["target_dir"]), ("org/model", "a/model.safetensors", "checkpoints"))
        for entry, message in (
            ({"repo_id": "org/model", "filename": "x"}, r"repo_id \(write repo\)"),
            ({"direct_url": "https://example.invalid/x", "filename": "x"}, r"direct_url \(write url\)"),
            ({"repo": "org/model", "file": "x"}, r"file \(write filename\)"),
            ({"repo": "org/model", "url": "https://example.invalid/x"}, "exactly one of repo or url"),
            ({"filename": "x"}, "exactly one of repo or url"),
            ({"repo": "org/model", "target_dir": ""}, "every value a non-empty string"),
        ):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, message):
                resolve_model_specs(WORKFLOW, {"checkpoints": {"model.safetensors": entry}})

    def test_the_workspace_model_registry_has_one_shape_and_migrate_moves_the_old_one(self) -> None:
        validate_workspace_config({"schema_version": 2, "comfyui": {"model_registry": {"checkpoints": {"m.safetensors": {"repo": "a/b"}}}}})
        old = {"schema_version": 2, "comfyui": {"model_registry": {"models": {"checkpoints": {"m.safetensors": {"repo_id": "a/b", "file": "m.safetensors"}}}}}}
        with self.assertRaises(ValueError):
            validate_workspace_config(old)
        migrated, notes = migrate_workspace_config(old)
        self.assertEqual(migrated["comfyui"]["model_registry"], {"checkpoints": {"m.safetensors": {"repo": "a/b", "filename": "m.safetensors"}}})
        self.assertIn("moved comfyui.model_registry.models.<folder> to comfyui.model_registry.<folder>", notes)

    def test_a_musubi_model_expectation_names_a_role_and_format_kura_knows(self) -> None:
        def run(expectations: object) -> dict:
            return {"backend": {"name": "musubi-tuner", "config": {"architecture": "wan", "model_expectations": expectations}}}

        self.assertEqual(_musubi_model_expectations(run({"t5": "safetensors"}))["t5"], "safetensors")
        for expectations, message in (
            ({"t5": "safetensor"}, "format.*safetensor; use one of"),
            ({"unet": "safetensors"}, "role.*unet that wan does not use"),
            (["t5"], "must map model roles to a format name"),
        ):
            with self.subTest(expectations=expectations), self.assertRaisesRegex(ValueError, message):
                _musubi_model_expectations(run(expectations))


if __name__ == "__main__":
    unittest.main()
