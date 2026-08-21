from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_evaluation_blocks import _is_immutable_run_artifact, candidate_paths, evaluation_errors
from scripts.check_evaluation_knowledge import card_errors


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
