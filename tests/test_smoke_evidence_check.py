"""Regression tests for support-to-evidence claim parsing."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from kura.provenance import adapter_source_identity, executor_source_identity

from scripts.check_smoke_evidence import (
    _executor_identity_reaches_current,
    _historical_record_is_retired,
    _identity_reaches_current,
    _requires_executor_identity,
    _support_evidence_claims,
)


class SmokeEvidenceCheckTests(unittest.TestCase):
    def test_sd1_evidence_migration_reaches_current_ai_toolkit_identity(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        evidence = yaml.safe_load((repository / "docs" / "backend-smoke-evidence.yaml").read_text(encoding="utf-8"))
        migrations = yaml.safe_load((repository / "docs" / "adapter-source-identity-migrations.yaml").read_text(encoding="utf-8"))
        record = next(item for item in evidence["records"] if item["id"] == "ai-toolkit-sd1-publication-docker-2026-09-23")
        migration = next(item for item in migrations["records"] if item["id"] == "ai-toolkit-registered-selector-validation-sd1-2026-09-23")
        latest = next(
            item for item in migrations["records"]
            if item["id"]
            == "ai-toolkit-audio-preflight-transform-ai-toolkit-2026-09-28"
        )

        self.assertEqual(migration["backend"], record["backend"])
        self.assertIs(migration["behavior_changed"], False)
        self.assertEqual(migration["evidence_ids"], [record["id"]])
        self.assertEqual(migration["previous"]["value"], record["adapter_source"]["value"])
        self.assertIn(record["id"], latest["evidence_ids"])
        self.assertEqual(latest["replacement"]["value"], adapter_source_identity("ai-toolkit")["value"])
        self.assertTrue(_identity_reaches_current(record["id"], record["backend"], record["adapter_source"]["value"], migrations["records"]))
        for field, invalid in (("backend", "musubi-tuner"), ("behavior_changed", True), ("evidence_ids", [])):
            with self.subTest(field=field):
                altered = deepcopy(migrations["records"])
                item = next(entry for entry in altered if entry["id"] == latest["id"])
                item[field] = invalid
                self.assertFalse(_identity_reaches_current(record["id"], record["backend"], record["adapter_source"]["value"], altered))

    def test_selected_file_runpod_evidence_binds_the_executor_identity(self) -> None:
        self.assertEqual(
            _requires_executor_identity({"native_path": {"executor": "runpod", "transfer": "selected-files"}}),
            "runpod",
        )
        self.assertIsNone(_requires_executor_identity({"native_path": {"executor": "runpod", "transfer": "legacy-upload"}}))
        self.assertIsNone(_requires_executor_identity({"native_path": {"executor": "local-docker", "transfer": "selected-files"}}))
        # A RunPod record cannot leave the requirement by omitting or misnaming its transfer.
        for native in ({"executor": "runpod"}, {"executor": "runpod", "transport": "selected-files"}, {"executor": "runpod", "transfer": "selected-file"}):
            with self.subTest(native=native), self.assertRaises(ValueError):
                _requires_executor_identity({"native_path": native})

    def test_executor_migration_chain_is_scoped_to_the_executor_and_the_evidence(self) -> None:
        current = executor_source_identity("runpod")["value"]
        migration = {
            "executor": "runpod",
            "behavior_changed": False,
            "evidence_ids": ["proof"],
            "previous": {"value": "old"},
            "replacement": {"value": current},
        }

        self.assertTrue(_executor_identity_reaches_current("proof", "runpod", current, []))
        self.assertTrue(_executor_identity_reaches_current("proof", "runpod", "old", [migration]))
        self.assertFalse(_executor_identity_reaches_current("other-proof", "runpod", "old", [migration]))
        for field, invalid in (("executor", "local-docker"), ("behavior_changed", True), ("evidence_ids", [])):
            with self.subTest(field=field):
                self.assertFalse(_executor_identity_reaches_current("proof", "runpod", "old", [{**migration, field: invalid}]))
        adapter_keyed = {key: value for key, value in migration.items() if key != "executor"} | {"backend": "runpod"}
        self.assertFalse(_executor_identity_reaches_current("proof", "runpod", "old", [adapter_keyed]))

    def test_executor_identity_follows_its_declared_sources(self) -> None:
        baseline = executor_source_identity("runpod", read=lambda relative: relative.encode())
        changed = executor_source_identity(
            "runpod",
            read=lambda relative: relative.encode() + (b"!" if relative == "dataset_transfer.py" else b""),
        )

        self.assertEqual(baseline["executor"], "runpod")
        self.assertEqual(baseline["scope"], "executor-v1")
        self.assertNotEqual(baseline["value"], changed["value"])
        with self.assertRaises(ValueError):
            executor_source_identity("local-docker")

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
