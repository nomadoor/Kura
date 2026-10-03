from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.cli import cmd_init
from kura.managed import MANIFEST, ensure_current
from kura.shipped import shipped_root

SHIPPED = Path(str(shipped_root()))


@contextmanager
def _workspace_with_source():
    """A fresh workspace plus a private copy of the shipped tree to change."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "ws"
        root.mkdir()
        source = Path(directory) / "shipped"
        shutil.copytree(SHIPPED, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch("kura.managed.shipped_root", return_value=source), patch("kura.init_templates.readiness_gaps", return_value=[]):
                yield root, source
        finally:
            os.chdir(previous)


def _init(**kwargs) -> tuple[int, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch("sys.stdin.isatty", return_value=False):
        code = cmd_init(argparse.Namespace(restore=kwargs.get("restore", False), yes=kwargs.get("yes", False)))
    return code, stdout.getvalue() + stderr.getvalue()


def _refresh(root: Path) -> str:
    stderr = io.StringIO()
    with patch("sys.stderr", stderr):
        ensure_current(root)
    return stderr.getvalue()


class ManagedFileTests(unittest.TestCase):
    def test_init_writes_every_managed_file_and_the_manifest(self) -> None:
        with _workspace_with_source() as (root, _):
            code, output = _init()
            self.assertEqual(code, 0, output)
            self.assertIn("Kura workspace", (root / "AGENTS.md").read_text(encoding="utf-8"))
            for skill in ("dataset-prep", "runpod-lifecycle"):
                self.assertTrue((root / ".agents" / "skills" / skill / "SKILL.md").is_file())
                self.assertTrue((root / ".claude" / "skills" / skill / "SKILL.md").is_file())
            self.assertTrue((root / ".kura" / "knowledge" / "regrets.md").is_file())
            self.assertTrue((root / ".kura" / "reference" / "external-access.md").is_file())
            self.assertTrue((root / "workflows" / "samples" / "README.md").is_file())
            manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
            self.assertIn("AGENTS.md", manifest["files"])
            self.assertFalse((root / ".agents" / "skills" / "dataset-prep" / "SKILL.md").is_symlink())

    def test_a_shipped_change_refreshes_unchanged_files_without_a_version_bump(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (source / "skills" / "dataset-prep" / "SKILL.md").write_text("---\nname: dataset-prep\ndescription: changed\n---\n", encoding="utf-8")
            output = _refresh(root)
            self.assertIn("refreshed", output)
            self.assertIn("changed", (root / ".claude" / "skills" / "dataset-prep" / "SKILL.md").read_text(encoding="utf-8"))
            self.assertEqual(_refresh(root), "")

    def test_an_edited_file_is_kept_and_named_until_restored(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (root / "AGENTS.md").write_text("my rules\n", encoding="utf-8")
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnew shipped rules\n", encoding="utf-8")
            first = _refresh(root)
            second = _refresh(root)
            self.assertEqual((root / "AGENTS.md").read_text(encoding="utf-8"), "my rules\n")
            self.assertIn("AGENTS.md", first)
            self.assertIn("AGENTS.md", second)
            code, output = _init(restore=True, yes=True)
            self.assertEqual(code, 0, output)
            self.assertIn("new shipped rules", (root / "AGENTS.md").read_text(encoding="utf-8"))
            self.assertNotIn("AGENTS.md", _refresh(root))

    def test_a_deleted_file_stays_deleted_until_restored(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            shutil.rmtree(root / ".claude")
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewer\n", encoding="utf-8")
            _refresh(root)
            self.assertFalse((root / ".claude").exists())
            _init()
            self.assertFalse((root / ".claude").exists())
            code, output = _init(restore=True)
            self.assertEqual(code, 1)
            self.assertIn(".claude/skills/dataset-prep/SKILL.md", output)
            self.assertFalse((root / ".claude").exists())
            code, _ = _init(restore=True, yes=True)
            self.assertEqual(code, 0)
            self.assertTrue((root / ".claude" / "skills" / "dataset-prep" / "SKILL.md").is_file())

    def test_deleted_user_files_are_not_recreated_by_init(self) -> None:
        with _workspace_with_source() as (root, _):
            _init()
            (root / "knowledge" / "regrets.md").unlink()
            (root / ".env.local").unlink()
            _init()
            self.assertFalse((root / "knowledge" / "regrets.md").exists())
            self.assertFalse((root / ".env.local").exists())

    def test_refresh_never_touches_user_data(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (root / "knowledge" / "regrets.md").write_text("mine\n", encoding="utf-8")
            config = (root / "workspace.yaml").read_text(encoding="utf-8")
            (source / "knowledge" / "regrets.md").write_text("shipped changed\n", encoding="utf-8")
            _refresh(root)
            self.assertEqual((root / "knowledge" / "regrets.md").read_text(encoding="utf-8"), "mine\n")
            self.assertEqual((root / "workspace.yaml").read_text(encoding="utf-8"), config)
            self.assertEqual((root / ".kura" / "knowledge" / "regrets.md").read_text(encoding="utf-8"), "shipped changed\n")

    def test_a_retired_shipped_file_is_removed_unless_edited(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            shutil.rmtree(source / "workflow-samples" / "anima")
            (source / "reference" / "external-access.md").unlink()
            (root / ".kura" / "reference" / "external-access.md").write_text("my notes\n", encoding="utf-8")
            output = _refresh(root)
            self.assertFalse((root / "workflows" / "samples" / "anima").exists())
            self.assertEqual((root / ".kura" / "reference" / "external-access.md").read_text(encoding="utf-8"), "my notes\n")
            self.assertIn("external-access.md", output)
            self.assertIn("external-access.md", _refresh(root))

    def test_a_previous_agents_stub_is_replaced_with_either_line_ending(self) -> None:
        stub = b"# Repository Guidelines\n\nKura is file-first: use the CLI for mutations and keep secrets out of run artifacts.\n"
        for written in (stub, stub.replace(b"\n", b"\r\n")):
            with self.subTest(crlf=b"\r" in written), _workspace_with_source() as (root, _):
                (root / "AGENTS.md").write_bytes(written)
                _init()
                self.assertIn("Kura workspace", (root / "AGENTS.md").read_text(encoding="utf-8"))

    def test_a_file_the_user_had_before_kura_is_never_overwritten(self) -> None:
        with _workspace_with_source() as (root, _):
            (root / "AGENTS.md").write_text("my own agent rules\n", encoding="utf-8")
            code, output = _init()
            self.assertEqual(code, 0)
            self.assertEqual((root / "AGENTS.md").read_text(encoding="utf-8"), "my own agent rules\n")
            self.assertIn("AGENTS.md", output)

    def test_a_damaged_manifest_or_busy_lock_never_stops_a_command(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (root / MANIFEST).write_text('{"files": {"AGENTS.md": "junk"}, "shipped_identity": 7}', encoding="utf-8")
            self.assertNotIn("Traceback", _refresh(root))
            (root / MANIFEST).write_bytes(b"\xff\xfe not json")
            self.assertNotIn("Traceback", _refresh(root))
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewer\n", encoding="utf-8")
            with patch("kura.managed.sync", side_effect=ValueError("lock busy")):
                self.assertEqual(_refresh(root), "")

    def test_an_unchanged_shipment_names_kept_files_without_writing(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (root / "AGENTS.md").write_text("mine\n", encoding="utf-8")
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewer\n", encoding="utf-8")
            _refresh(root)
            before = (root / MANIFEST).stat().st_mtime_ns
            output = _refresh(root)
            self.assertIn("AGENTS.md", output)
            self.assertEqual((root / MANIFEST).stat().st_mtime_ns, before)

    def test_undoing_an_edit_returns_the_file_to_kura(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            original = (root / "AGENTS.md").read_bytes()
            (root / "AGENTS.md").write_text("mine\n", encoding="utf-8")
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewer\n", encoding="utf-8")
            _refresh(root)
            (root / "AGENTS.md").write_bytes(original)
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewest\n", encoding="utf-8")
            output = _refresh(root)
            self.assertIn("newest", (root / "AGENTS.md").read_text(encoding="utf-8"))
            self.assertNotIn("AGENTS.md", output)

    def test_deleting_an_edited_file_counts_as_a_deletion(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (root / "AGENTS.md").write_text("mine\n", encoding="utf-8")
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewer\n", encoding="utf-8")
            _refresh(root)
            (root / "AGENTS.md").unlink()
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewest\n", encoding="utf-8")
            output = _refresh(root)
            self.assertFalse((root / "AGENTS.md").exists())
            self.assertNotIn("AGENTS.md", output)

    @unittest.skipIf(os.name == "nt", "symlinks need extra privileges on Windows")
    def test_restore_replaces_a_symlinked_managed_file(self) -> None:
        with _workspace_with_source() as (root, _):
            _init()
            target = root / ".claude" / "skills" / "dataset-prep" / "SKILL.md"
            (root / "elsewhere.md").write_text("linked\n", encoding="utf-8")
            target.unlink()
            target.symlink_to(root / "elsewhere.md")
            _init(restore=True, yes=True)
            self.assertFalse(target.is_symlink())
            self.assertIn("dataset-prep", target.read_text(encoding="utf-8"))
            self.assertEqual((root / "elsewhere.md").read_text(encoding="utf-8"), "linked\n")

    def test_a_workspace_without_agent_files_is_told_to_run_init(self) -> None:
        with _workspace_with_source() as (root, _):
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            self.assertIn("run `kura init`", _refresh(root))

    def test_a_kura_checkout_is_never_managed(self) -> None:
        with _workspace_with_source() as (root, _):
            (root / "src" / "kura" / "shipped").mkdir(parents=True)
            code, output = _init()
            self.assertEqual(code, 0, output)
            self.assertFalse((root / "AGENTS.md").exists())
            self.assertEqual(_refresh(root), "")

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "needs POSIX permissions as a non-root user")
    def test_a_read_only_workspace_is_skipped_quietly(self) -> None:
        with _workspace_with_source() as (root, source):
            _init()
            (source / "AGENTS.md").write_text("# Kura workspace\n\nnewer\n", encoding="utf-8")
            os.chmod(root / ".kura", 0o500)
            os.chmod(root, 0o500)
            try:
                output = _refresh(root)
            finally:
                os.chmod(root, 0o700)
                os.chmod(root / ".kura", 0o700)
            self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
