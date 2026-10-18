"""Regression tests for the smoke evidence ledger check."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import scripts.check_smoke_evidence as check


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "proof",
        "backend": "sd-scripts",
        "adapter_source": {"kind": "source-tree-sha256", "value": "a" * 64},
        "runtime_image": {"kind": "docker-image-digest", "value": "sha256:" + "b" * 64},
        "native_path": {"architecture": "sd15", "executor": "docker"},
        "evidence_kind": "real-optimizer-step",
        "outcome": "passed",
        "observed_at": "2026-10-18",
        "artifact": "smoke-evidence/proof.yaml",
    }
    record.update(overrides)
    return record


class SmokeEvidenceCheckTests(unittest.TestCase):
    def _check(self, records: list[dict[str, object]], *, artifact: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "backend-smoke-evidence.yaml"
            ledger.write_text(yaml.safe_dump({"schema_version": 1, "records": records}), encoding="utf-8")
            if artifact:
                (ledger.parent / "smoke-evidence").mkdir()
                (ledger.parent / "smoke-evidence" / "proof.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            with mock.patch.object(check, "PATH", ledger):
                check.main()

    def test_a_record_with_its_artifact_passes_with_or_without_an_executor_source(self) -> None:
        # Old records carry executor_source; new ones do not. The check neither needs nor refuses it.
        runpod = {"architecture": "sd15", "executor": "runpod", "transfer": "selected-files"}
        self._check([_record()], artifact=True)
        self._check([_record(native_path=runpod)], artifact=True)
        self._check([_record(native_path=runpod, executor_source={"kind": "source-tree-sha256", "value": "c" * 64})], artifact=True)

    def test_a_missing_artifact_file_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "artifact does not exist: smoke-evidence/proof.yaml"):
            self._check([_record()], artifact=False)


if __name__ == "__main__":
    unittest.main()
