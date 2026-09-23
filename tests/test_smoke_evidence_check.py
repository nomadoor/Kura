"""Regression tests for support-to-evidence claim parsing."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from kura.provenance import adapter_source_identity

from scripts.check_smoke_evidence import _historical_record_is_retired, _support_evidence_claims


class SmokeEvidenceCheckTests(unittest.TestCase):
    def test_sd1_evidence_migration_reaches_current_ai_toolkit_identity(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        evidence = yaml.safe_load((repository / "docs" / "backend-smoke-evidence.yaml").read_text(encoding="utf-8"))
        migrations = yaml.safe_load((repository / "docs" / "adapter-source-identity-migrations.yaml").read_text(encoding="utf-8"))
        record = next(item for item in evidence["records"] if item["id"] == "ai-toolkit-sd1-publication-docker-2026-09-23")
        migration = next(item for item in migrations["records"] if item["id"] == "ai-toolkit-registered-selector-validation-sd1-2026-09-23")

        self.assertEqual(migration["previous"]["value"], record["adapter_source"]["value"])
        self.assertEqual(migration["replacement"]["value"], adapter_source_identity("ai-toolkit")["value"])

    def test_verified_support_without_evidence_is_returned_for_validation(self) -> None:
        claims = _support_evidence_claims(
            "| Backend | Model family | Adapter | Status | Notes |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Musubi Tuner | Wan | Built-in | ✅ | Local and RunPod verified |\n"
        )

        self.assertEqual(claims, [(3, "Musubi Tuner", "✅", [])])

    def test_evidence_references_are_extracted_from_support_notes(self) -> None:
        claims = _support_evidence_claims(
            "| Backend | Model family | Adapter | Status | Notes |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| AI-Toolkit | SDXL | Generic | ✅ | Evidence: `local-proof`, `remote-proof` |\n"
        )

        self.assertEqual(claims, [(3, "AI-Toolkit", "✅", ["local-proof", "remote-proof"])])

    def test_historical_evidence_requires_a_declared_retirement(self) -> None:
        self.assertTrue(_historical_record_is_retired({"superseded_by": "new-proof"}, set()))
        self.assertTrue(_historical_record_is_retired({"invalidated_by": "contract-change"}, {"contract-change"}))
        self.assertFalse(_historical_record_is_retired({"invalidated_by": "typo"}, {"contract-change"}))


if __name__ == "__main__":
    unittest.main()
