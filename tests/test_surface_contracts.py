"""Every surface a user authors must refuse values Kura would silently ignore.

This is deliberately cross-surface. Kura lost this property once already: the
rule was implemented for `recipe` and for the sd-scripts adapter, was never
written down as a rule, and was then missed on the third and fourth consumers.
A per-surface test would repeat that. This one fails when a *new* surface is
added without a policy, because the registry below is what it walks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kura.render import authored_cases, promptset, reconcile_promptset  # noqa: E402
from kura.run_envelope import common_recipe, resume_intent, training_state_policy  # noqa: E402
from kura.workspace import (  # noqa: E402
    WORKSPACE_OBSOLETE_KEYS,
    WORKSPACE_SCHEMA,
    validate_workspace_config,
    workspace_schema_description,
)

SENTINEL = "kura_unknown_sentinel"


class SurfaceContractTests(unittest.TestCase):
    """Each entry drives one authoring surface with an undeclared key."""

    def test_workspace_sections_reject_an_undeclared_key(self) -> None:
        for section, schema in WORKSPACE_SCHEMA["fields"].items():
            if schema["kind"] != "mapping":
                continue
            with self.subTest(section=section):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config({section: {SENTINEL: True}})
                self.assertIn(SENTINEL, str(caught.exception))
                self.assertIn("kura doctor workspace", str(caught.exception))

    def test_workspace_rejects_an_undeclared_section(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_workspace_config({SENTINEL: {}})
        self.assertIn("unsupported section", str(caught.exception))

    def test_non_string_mapping_keys_are_rejected_without_a_traceback(self) -> None:
        for config in ({7: {}}, {"docker": {}, 7: {}}, {"runpod": {7: True}}, {"runpod": {"default_image": {7: "image"}}}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config(config)
                self.assertIn("7", str(caught.exception))
                self.assertIn("kura doctor workspace", str(caught.exception))

    def test_workspace_names_the_fix_for_a_plausible_misspelling(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_workspace_config({"comfyui": {"input_stage_mod": "copy"}})
        self.assertIn("input_stage_mode", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            validate_workspace_config({"comfyui_endpoint": "http://127.0.0.1:8188"})
        self.assertIn("comfyui.endpoint", str(caught.exception))

    def test_workspace_reports_an_obsolete_key_as_stale_not_as_a_typo(self) -> None:
        for dotted in WORKSPACE_OBSOLETE_KEYS:
            section, key = dotted.split(".", 1)
            with self.subTest(key=dotted):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config({section: {key: "x"}})
                message = str(caught.exception)
                self.assertIn("obsolete", message)
                self.assertIn("Delete these lines", message)

    def test_dynamic_names_are_allowed_but_their_values_stay_closed(self) -> None:
        validate_workspace_config({
            "docker": {"images": {"my-backend": {"local": "example/image:tag"}}},
            "runpod": {"default_image": {"my-backend": "example/image@sha256:abc"}},
            "comfyui": {"model_registry": {"checkpoints": {"my-model.safetensors": {"repo": "owner/model"}}}},
        })
        malformed = (
            ({"docker": {"images": {"my-backend": {"dockerfil": "Dockerfile"}}}}, "dockerfile"),
            ({"comfyui": {"model_registry": {"checkpoints": {"model.safetensors": {"reop": "owner/model"}}}}}, "repo"),
            ({"runpod": {"backend_ports": {"my-backend": {"port": "22/tcp"}}}}, "must be a list"),
        )
        for config, expected in malformed:
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config(config)
                self.assertIn(expected, str(caught.exception))

    def test_nested_fixed_maps_and_list_items_reject_unknown_keys(self) -> None:
        malformed = (
            ({"runpod": {"object_store": {"buckett": "models"}}}, "bucket"),
            ({"comfyui": {"runpod": {"container_disk_g": 80}}}, "container_disk_gb"),
            ({"docker": {"mounts": [{"source": ".", "target": "/workspace", "writable": True}]}}, "writable"),
        )
        for config, expected in malformed:
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config(config)
                self.assertIn(expected, str(caught.exception))

    def test_interpreted_values_are_type_checked_at_workspace_load(self) -> None:
        malformed = (
            ({"docker": {"gpu": "true"}}, "must be boolean"),
            ({"docker": {"mounts": [{"source": ".", "target": "/workspace", "mode": "write"}]}}, "'ro', 'rw'"),
            ({"comfyui": {"input_stage_mode": "hardlink"}}, "'symlink', 'copy'"),
            ({"comfyui": {"model_registry": {"checkpoints": {"model.safetensors": {"repo": 7}}}}}, "must be string"),
            ({"runpod": {"container_disk_gb": "80"}}, "must be integer"),
            ({"runpod": {"ports": [22]}}, "runpod.ports[0] must be string"),
        )
        for config, expected in malformed:
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config(config)
                self.assertIn(expected, str(caught.exception))

    def test_workspace_does_not_advertise_run_only_or_unused_render_overrides(self) -> None:
        for config, rejected in (
            ({"safety": {"allow_many_checkpoints": True}}, "safety"),
            ({"comfyui": {"runpod": {"template_id": "ignored"}}}, "template_id"),
            ({"comfyui": {"runpod": {"storage_mode": "object_staging"}}}, "storage_mode"),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config(config)
                self.assertIn(rejected, str(caught.exception))

    def test_consumed_docker_build_cache_limit_is_declared(self) -> None:
        validate_workspace_config({"docker": {"build_cache_limit_gb": 20}})

    def test_workspace_schema_description_exposes_nested_contract(self) -> None:
        settings = workspace_schema_description()
        self.assertEqual(settings["docker"]["images"]["<name>"]["dockerfile"], "string")
        self.assertEqual(settings["runpod"]["object_store"]["bucket"], "string")
        self.assertEqual(settings["comfyui"]["runpod"]["container_disk_gb"], "integer")
        self.assertEqual(settings["docker"]["mounts"]["list_of"]["mode"]["choices"], ["ro", "rw"])

    def test_doctor_workspace_reports_nested_contract_error_without_source_reading(self) -> None:
        from kura.doctor import cmd_doctor_workspace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workspace.yaml").write_text("runpod:\n  object_store:\n    buckett: models\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    code = cmd_doctor_workspace(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 1)
            self.assertIn("use 'bucket'", payload["configuration_error"])

    def test_recipe_rejects_an_undeclared_key(self) -> None:
        with self.assertRaises(ValueError) as caught:
            common_recipe({"recipe": {"steps": 1, "seed": 1, SENTINEL: True}})
        self.assertIn(SENTINEL, str(caught.exception))

    def test_recovery_and_continuation_reject_undeclared_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, SENTINEL):
            training_state_policy({"recovery": {"training_state": {SENTINEL: True}}})
        with self.assertRaisesRegex(ValueError, SENTINEL):
            resume_intent({
                "parent_run": "source",
                "continuation": {"mode": "resume", SENTINEL: True},
            })

    def test_promptset_rejects_an_unbound_key(self) -> None:
        patches = {"prompt": {"node": "6", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}}
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset([{"id": "a", "prompt": "x", "seeds": [1], SENTINEL: True}], patches)
        self.assertIn(SENTINEL, str(caught.exception))

    def test_promptset_rejects_an_unsafe_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(json.dumps({"id": "../escape", "prompt": "x"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                promptset(path)

    def test_render_case_rejects_an_undeclared_row_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"id": "a", "values": {"prompt": "x"}, SENTINEL: True}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                authored_cases(path)
        self.assertIn(SENTINEL, str(caught.exception))

    def test_every_workspace_key_kura_writes_is_declared(self) -> None:
        """`kura init` must not ship a setting the contract would reject."""
        from kura.init_templates import cmd_init
        import argparse
        import os

        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                cmd_init(argparse.Namespace())
                written = yaml.safe_load((Path(tmp) / "workspace.yaml").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous)
        validate_workspace_config(written)


if __name__ == "__main__":
    unittest.main()
