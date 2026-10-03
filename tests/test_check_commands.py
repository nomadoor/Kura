from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.cli import cmd_check_artifacts, cmd_check_secrets, cmd_workflow_check

# Built at runtime so this file itself does not trip the repository secret scan.
FAKE_TOKEN_LINE = "hf" + "_" + "A" * 24


@contextmanager
def _workspace():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(previous)


def _run(command, **kwargs) -> tuple[int, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        code = command(argparse.Namespace(**kwargs))
    return code, stdout.getvalue() + stderr.getvalue()


class WorkflowCheckCommandTests(unittest.TestCase):
    def test_reports_the_repository_check_findings_for_named_files(self) -> None:
        with _workspace() as root:
            workflows = root / "workflows"
            workflows.mkdir()
            (workflows / "ui.json").write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
            (workflows / "good.json").write_text(json.dumps({"1": {"class_type": "X"}}), encoding="utf-8")
            promptsets = root / "promptsets"
            promptsets.mkdir()
            (promptsets / "dup.jsonl").write_text('{"id": "a"}\n{"id": "a"}\n', encoding="utf-8")

            code, output = _run(cmd_workflow_check, paths=["workflows/ui.json", "promptsets/dup.jsonl"])
            self.assertEqual(code, 1)
            self.assertIn("looks like a UI workflow export", output)
            self.assertIn("duplicate id 'a'", output)

            code, output = _run(cmd_workflow_check, paths=["workflows/good.json"])
            self.assertEqual(code, 0, output)

    def test_without_paths_checks_the_workspace_workflows_and_promptsets(self) -> None:
        with _workspace() as root:
            (root / "workflows").mkdir()
            (root / "workflows" / "empty.json").write_text("{}", encoding="utf-8")
            code, output = _run(cmd_workflow_check, paths=[])
        self.assertEqual(code, 1)
        self.assertIn("workflows/empty.json is empty", output)


class WorkflowCheckStrictnessTests(unittest.TestCase):
    def test_a_named_ui_export_is_refused_wherever_it_lives_unless_it_has_an_api_twin(self) -> None:
        with _workspace() as root:
            elsewhere = root / "downloads"
            elsewhere.mkdir()
            (elsewhere / "export.json").write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
            code, output = _run(cmd_workflow_check, paths=["downloads/export.json"])
            self.assertEqual(code, 1)
            self.assertIn("looks like a UI workflow export", output)
            (elsewhere / "export_api.json").write_text(json.dumps({"1": {}}), encoding="utf-8")
            code, output = _run(cmd_workflow_check, paths=["downloads/export.json"])
            self.assertEqual(code, 0, output)

    def test_unrelated_files_and_empty_selections_are_refused(self) -> None:
        with _workspace() as root:
            (root / "notes.md").write_text("x\n", encoding="utf-8")
            code, output = _run(cmd_workflow_check, paths=["notes.md"])
            self.assertEqual(code, 1)
            self.assertIn("not workflow JSON or promptset JSONL", output)
            code, output = _run(cmd_workflow_check, paths=[])
            self.assertEqual(code, 1)
            self.assertIn("no workflow JSON or promptset JSONL", output)

    def test_default_mode_needs_a_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                code, output = _run(cmd_workflow_check, paths=[])
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertIn("workspace.yaml was not found", output)

    @unittest.skipIf(os.name == "nt", "a colon names an alternate data stream on NTFS, not a sidecar file")
    def test_default_mode_skips_case_queues_and_flags_zone_sidecars(self) -> None:
        with _workspace() as root:
            workflows = root / "workflows"
            workflows.mkdir()
            (workflows / "good.json").write_text(json.dumps({"1": {}}), encoding="utf-8")
            (workflows / "cases.jsonl").write_text('{"values": {}}\n', encoding="utf-8")
            code, output = _run(cmd_workflow_check, paths=[])
            self.assertEqual(code, 0, output)
            (workflows / "good.json:Zone.Identifier").write_text("x", encoding="utf-8")
            code, output = _run(cmd_workflow_check, paths=[])
        self.assertEqual(code, 1)
        self.assertIn("Zone.Identifier sidecar", output)


class SecretAndArtifactCommandTests(unittest.TestCase):
    def test_secret_check_reads_only_named_paths_and_never_env_local(self) -> None:
        with _workspace() as root:
            upload = root / "upload"
            upload.mkdir()
            (upload / "README.md").write_text("model card\n", encoding="utf-8")
            (upload / ".env.local").write_text(FAKE_TOKEN_LINE + "\n", encoding="utf-8")
            (root / "elsewhere.txt").write_text(FAKE_TOKEN_LINE + "\n", encoding="utf-8")

            code, output = _run(cmd_check_secrets, paths=["upload"])
            self.assertEqual(code, 0, output)

            (upload / "notes.txt").write_text(FAKE_TOKEN_LINE + "\n", encoding="utf-8")
            code, output = _run(cmd_check_secrets, paths=["upload"])
            self.assertEqual(code, 1)
            self.assertIn("upload/notes.txt:1", output)
            self.assertNotIn(".env.local", output)
            self.assertNotIn("elsewhere.txt", output)

            code, output = _run(cmd_check_secrets, paths=["upload/.env.local"])
            self.assertEqual(code, 1)
            self.assertIn("never reads", output)
            self.assertNotIn(FAKE_TOKEN_LINE, output)

    @unittest.skipIf(os.name == "nt", "symlinks need extra privileges on Windows")
    def test_a_symlink_to_env_local_is_not_followed(self) -> None:
        with _workspace() as root:
            (root / ".env.local").write_text(FAKE_TOKEN_LINE + "\n", encoding="utf-8")
            upload = root / "upload"
            upload.mkdir()
            (upload / "config.txt").symlink_to(root / ".env.local")
            code, output = _run(cmd_check_secrets, paths=["upload"])
        self.assertEqual(code, 0, output)
        self.assertNotIn(".env.local", output)

    def test_secret_values_are_never_printed_and_binary_or_weight_files_are_skipped(self) -> None:
        with _workspace() as root:
            share = root / "share"
            share.mkdir()
            (share / "adapter.safetensors").write_bytes(FAKE_TOKEN_LINE.encode())
            (share / "blob.dat").write_bytes(b"\0" + FAKE_TOKEN_LINE.encode())
            code, output = _run(cmd_check_secrets, paths=["share"])
            self.assertEqual(code, 0, output)
            (share / "notes.txt").write_text(FAKE_TOKEN_LINE + "\n", encoding="utf-8")
            code, output = _run(cmd_check_secrets, paths=["share"])
        self.assertEqual(code, 1)
        self.assertIn("share/notes.txt:1: looks like a secret value", output)
        self.assertNotIn(FAKE_TOKEN_LINE, output)

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "needs POSIX permissions as a non-root user")
    def test_an_unreadable_file_is_a_finding_not_a_pass(self) -> None:
        with _workspace() as root:
            locked = root / "locked.txt"
            locked.write_text("x\n", encoding="utf-8")
            locked.chmod(0)
            try:
                code, output = _run(cmd_check_secrets, paths=["locked.txt"])
            finally:
                locked.chmod(0o600)
        self.assertEqual(code, 1)
        self.assertIn("locked.txt: cannot be read", output)

    def test_artifact_check_names_model_weights_in_the_given_paths(self) -> None:
        with _workspace() as root:
            share = root / "share"
            share.mkdir()
            (share / "notes.md").write_text("x\n", encoding="utf-8")
            code, output = _run(cmd_check_artifacts, paths=["share"])
            self.assertEqual(code, 0, output)
            (share / "adapter.safetensors").write_bytes(b"x")
            code, output = _run(cmd_check_artifacts, paths=["share"])
            self.assertEqual(code, 1)
            self.assertIn("share/adapter.safetensors", output)

    def test_a_missing_path_is_an_error(self) -> None:
        with _workspace():
            code, output = _run(cmd_check_secrets, paths=["nope"])
        self.assertEqual(code, 1)
        self.assertIn("nope", output)


if __name__ == "__main__":
    unittest.main()
