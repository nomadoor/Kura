"""Regression tests for support-to-evidence claim parsing."""

from __future__ import annotations

import unittest

from scripts.check_smoke_evidence import _support_evidence_claims


class SmokeEvidenceCheckTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
