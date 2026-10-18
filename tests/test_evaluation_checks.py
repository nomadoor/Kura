from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_evaluation_blocks import _is_immutable_run_artifact, candidate_paths, evaluation_errors
from scripts.check_evaluation_knowledge import KNOWLEDGE_DIR, card_errors, card_model_file_blocks, card_model_file_errors, real_smoke_recipes


CARD_BLOCK = """# fam

## Model files

```yaml
backend:
  name: musubi-tuner
  config:
    architecture: zimage
    model_downloads:
      dit:
        repo: Comfy-Org/z_image
        filename: split_files/diffusion_models/z_image_bf16.safetensors
```
source: real smoke musubi-zimage
"""


class CardModelFilesTests(unittest.TestCase):
    """A card's model_downloads block and scripts/real_smoke.py's recipe cannot drift apart."""

    RECIPES = [("musubi-tuner", {"architecture": "zimage", "fp8_base": True, "model_downloads": {
        "dit": {"repo": "Comfy-Org/z_image", "filename": "split_files/diffusion_models/z_image_bf16.safetensors"},
    }})]

    def _errors(self, text: str, recipes: list) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fam.md"
            path.write_text(text, encoding="utf-8")
            return card_model_file_errors(path, recipes)

    def test_a_block_equal_to_a_recipe_passes(self) -> None:
        self.assertEqual(self._errors(CARD_BLOCK, self.RECIPES), [])

    def test_a_block_that_drifted_from_the_recipe_is_refused(self) -> None:
        drifted = [("musubi-tuner", {"architecture": "zimage", "model_downloads": {
            "dit": {"repo": "Comfy-Org/z_image", "filename": "split_files/diffusion_models/z_image_fp8.safetensors"},
        }})]
        errors = self._errors(CARD_BLOCK, drifted)
        self.assertEqual(len(errors), 1)
        self.assertIn("fam.md", errors[0])
        self.assertIn("scripts/real_smoke.py", errors[0])
        self.assertEqual(self._errors(CARD_BLOCK.replace("musubi-tuner", "sd-scripts"), self.RECIPES)[0][:6], errors[0][:6])

    def test_a_block_must_name_the_backend_and_model_downloads(self) -> None:
        errors = self._errors(CARD_BLOCK.replace("  name: musubi-tuner\n", ""), self.RECIPES)
        self.assertTrue(errors)

    def test_the_shipped_cards_match_the_real_smoke_recipes(self) -> None:
        recipes = real_smoke_recipes()
        checked = {}
        for card in sorted(KNOWLEDGE_DIR.glob("*.md")):
            with self.subTest(card=card.name):
                self.assertEqual(card_model_file_errors(card, recipes), [])
                checked[card.stem] = len(card_model_file_blocks(card))
        self.assertGreaterEqual(checked["anima"], 1)
        self.assertGreaterEqual(checked["zimage"], 1)


class EvaluationCheckTests(unittest.TestCase):
    def test_evaluation_absence_requires_no_validation(self) -> None:
        payload = {"type": "render", "inputs": {"train_run": None}}
        self.assertNotIn("evaluation", payload)

    def test_unknown_evaluation_category_warns_but_does_not_fail(self) -> None:
        evaluation = {
            "category": "edit_fidelity",
            "fixed": ["seed"],
            "varied": ["edit"],
            "model_family": "flux_kontext",
            "model_variant": "dev",
            "knowledge": {"card": "none", "basis": ["https://example.invalid/model"], "confidence": "unverified"},
            "prompt_policy": {"prefix_origin": "agent", "transformations": []},
            "limits": "One-axis example.",
        }
        errors, warnings = evaluation_errors(evaluation, label="example")
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)

    def test_missing_card_reference_fails_mechanically(self) -> None:
        evaluation = {
            "category": "outfit_transfer",
            "fixed": ["seed"],
            "varied": ["outfit"],
            "model_family": "anima",
            "model_variant": "base",
            "knowledge": {
                "card": ".claude/skills/lora-evaluation/knowledge/missing.md",
                "card_verified_at": "2026-08-03",
                "source_url": "https://example.invalid/model",
                "source_revision": "main",
                "applies_to_model_revision": "unknown",
                "revision_match": "unverified",
            },
            "prompt_policy": {"prefix_origin": "knowledge_card", "transformations": []},
            "limits": "Example.",
        }
        errors, _ = evaluation_errors(evaluation, label="example")
        self.assertTrue(any("does not exist" in item for item in errors))

    def _card_evaluation(self, card: str) -> dict:
        return {
            "category": "outfit_transfer",
            "fixed": ["seed"],
            "varied": ["outfit"],
            "model_family": "anima",
            "model_variant": "base",
            "knowledge": {
                "card": card,
                "card_verified_at": "2026-08-03",
                "source_url": "https://example.invalid/model",
                "source_revision": "main",
                "applies_to_model_revision": "unknown",
                "revision_match": "unverified",
            },
            "prompt_policy": {"prefix_origin": "knowledge_card", "transformations": []},
            "limits": "Example.",
        }

    def test_frozen_citation_follows_both_card_moves(self) -> None:
        for card in (
            ".claude/skills/lora-evaluation/knowledge/anima.md",
            ".claude/skills/training-parameter-planning/knowledge/anima.md",
        ):
            with self.subTest(card=card):
                self.assertFalse((Path(__file__).resolve().parents[1] / card).is_file())
                evaluation = self._card_evaluation(card)
                errors, _ = evaluation_errors(
                    evaluation,
                    label="immutable manifest",
                    allow_legacy_card_move=True,
                )
                self.assertFalse([item for item in errors if "does not exist" in item])

    def test_frozen_citation_follows_the_move_into_the_package(self) -> None:
        card = "knowledge/model-families/anima.md"
        self.assertFalse((Path(__file__).resolve().parents[1] / card).is_file())
        errors, _ = evaluation_errors(self._card_evaluation(card), label="immutable manifest", allow_legacy_card_move=True)
        self.assertFalse([item for item in errors if "does not exist" in item])
        errors, _ = evaluation_errors(self._card_evaluation(card), label="editable run.yaml")
        self.assertTrue(any("does not exist" in item for item in errors))

    def test_workspace_shipped_knowledge_path_resolves_to_the_shipped_source(self) -> None:
        errors, _ = evaluation_errors(self._card_evaluation(".kura/knowledge/model-families/anima.md"), label="run.yaml")
        self.assertFalse([item for item in errors if "does not exist" in item])
        errors, _ = evaluation_errors(self._card_evaluation(".kura/knowledge/../../../../../../../../etc/passwd"), label="run.yaml")
        self.assertTrue(any("inside the shipped knowledge" in item for item in errors))
        errors, _ = evaluation_errors(self._card_evaluation(".kura/knowledge/../../../../README.md"), label="run.yaml")
        self.assertTrue(any("inside the shipped knowledge" in item for item in errors))

    def test_editable_run_does_not_follow_the_card_move(self) -> None:
        evaluation = self._card_evaluation(".claude/skills/lora-evaluation/knowledge/anima.md")
        errors, _ = evaluation_errors(evaluation, label="editable run.yaml")
        self.assertTrue(any("does not exist" in item for item in errors))

    def test_only_compiled_run_artifacts_are_immutable_move_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft = root / "runs" / "draft" / "run.yaml"
            draft.parent.mkdir(parents=True)
            draft.write_text("type: render\n", encoding="utf-8")
            example_manifest = root / "examples" / "resolved" / "manifest.lock.yaml"
            example_manifest.parent.mkdir(parents=True)
            example_manifest.write_text("type: render\n", encoding="utf-8")
            with patch("scripts.check_evaluation_blocks.ROOT", root):
                self.assertFalse(_is_immutable_run_artifact(draft))
                self.assertFalse(_is_immutable_run_artifact(example_manifest))
                manifest = draft.parent / "resolved" / "manifest.lock.yaml"
                manifest.parent.mkdir()
                manifest.write_text("type: render\n", encoding="utf-8")
                self.assertTrue(_is_immutable_run_artifact(draft))
                self.assertTrue(_is_immutable_run_artifact(manifest))

    def test_repository_candidates_exclude_ignored_workspace_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / "examples" / "run-example.yaml"
            workspace_run = root / "runs" / "render-1" / "run.yaml"
            example.parent.mkdir(parents=True)
            workspace_run.parent.mkdir(parents=True)
            example.write_text("type: render\n", encoding="utf-8")
            workspace_run.write_text("type: render\n", encoding="utf-8")
            with patch("scripts.check_evaluation_blocks.ROOT", root):
                self.assertEqual(candidate_paths(include_workspace_runs=False), [example])
                self.assertEqual(candidate_paths(include_workspace_runs=True), [example, workspace_run])

    def test_card_move_is_not_followed_from_an_arbitrary_directory(self) -> None:
        evaluation = self._card_evaluation("docs/anima.md")
        errors, _ = evaluation_errors(
            evaluation,
            label="example",
            allow_legacy_card_move=True,
        )
        self.assertTrue(any("does not exist" in item for item in errors))

    def test_headerless_family_card_accepts_inline_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            path.write_text("# family\n\n- observed behavior\n  source: run example\n", encoding="utf-8")
            self.assertEqual(card_errors(path), [])

    def test_headerless_family_card_rejects_missing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            path.write_text("# family\n\n- unsupported assertion\n", encoding="utf-8")
            errors = card_errors(path)
        self.assertTrue(any("inline source:" in item for item in errors))

    def test_family_card_requires_confidence_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            path.write_text(
                "source_url: https://example.invalid\nsource_revision: main\nverified_at: 2026-08-03\n"
                "applies_to_model_revision: unverified\nconfidence: certain\n",
                encoding="utf-8",
            )
            errors = card_errors(path)
        self.assertTrue(any("confidence must be" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
