"""Every surface a user authors must refuse values Kura would silently ignore.

This is deliberately cross-surface. Kura lost this property once already: the
rule was implemented for `recipe` and for the sd-scripts adapter, was never
written down as a rule, and was then missed on the third and fourth consumers.
A per-surface test would repeat that. This one fails when a *new* surface is
added without a policy, because the registry below is what it walks.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kura.render import promptset, reconcile_promptset  # noqa: E402
from kura.run_envelope import common_recipe  # noqa: E402
from kura.workspace import (  # noqa: E402
    WORKSPACE_OBSOLETE_KEYS,
    WORKSPACE_OPEN_SUBTREES,
    WORKSPACE_SURFACE,
    validate_workspace_config,
)

SENTINEL = "kura_unknown_sentinel"


class SurfaceContractTests(unittest.TestCase):
    """Each entry drives one authoring surface with an undeclared key."""

    def test_workspace_sections_reject_an_undeclared_key(self) -> None:
        for section, accepted in WORKSPACE_SURFACE.items():
            if not accepted:
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

    def test_workspace_names_the_fix_for_a_plausible_misspelling(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_workspace_config({"comfyui": {"input_stage_mod": "copy"}})
        self.assertIn("input_stage_mode", str(caught.exception))

    def test_workspace_reports_an_obsolete_key_as_stale_not_as_a_typo(self) -> None:
        for dotted in WORKSPACE_OBSOLETE_KEYS:
            section, key = dotted.split(".", 1)
            with self.subTest(key=dotted):
                with self.assertRaises(ValueError) as caught:
                    validate_workspace_config({section: {key: "x"}})
                message = str(caught.exception)
                self.assertIn("obsolete", message)
                self.assertIn("Delete these lines", message)

    def test_uninterpreted_subtrees_are_named_and_accept_user_vocabulary(self) -> None:
        """A subtree Kura does not read must be declared, not silently tolerated."""
        for dotted in WORKSPACE_OPEN_SUBTREES:
            section, key = dotted.split(".", 1)
            self.assertIn(section, WORKSPACE_SURFACE, dotted)
            self.assertIn(key, WORKSPACE_SURFACE[section], dotted)
        validate_workspace_config({"comfyui": {"model_registry": {"anything": {"repo": "x"}}}})

    def test_recipe_rejects_an_undeclared_key(self) -> None:
        with self.assertRaises(ValueError) as caught:
            common_recipe({"recipe": {"steps": 1, "seed": 1, SENTINEL: True}})
        self.assertIn(SENTINEL, str(caught.exception))

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
