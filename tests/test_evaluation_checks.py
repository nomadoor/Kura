from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_evaluation_blocks import evaluation_errors
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

    def test_frozen_citation_follows_the_card_move(self) -> None:
        evaluation = self._card_evaluation(".claude/skills/lora-evaluation/knowledge/anima.md")
        errors, _ = evaluation_errors(evaluation, label="example")
        self.assertFalse([item for item in errors if "does not exist" in item])

    def test_card_move_is_not_followed_from_an_arbitrary_directory(self) -> None:
        evaluation = self._card_evaluation("docs/anima.md")
        errors, _ = evaluation_errors(evaluation, label="example")
        self.assertTrue(any("does not exist" in item for item in errors))

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
