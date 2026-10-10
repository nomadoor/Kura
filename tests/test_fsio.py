from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.fsio import (
    FileLockBusy, append_line_durably, atomic_write_json, atomic_write_text, atomic_write_yaml, create_new_files, file_lock,
)


class FsioTests(unittest.TestCase):
    def test_create_new_files_never_replaces_and_removes_what_it_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            second.write_bytes(b"authored")
            with self.assertRaises(FileExistsError):
                create_new_files({first: b"new", second: b"new"})
            self.assertFalse(first.exists())
            self.assertEqual(second.read_bytes(), b"authored")

    @unittest.skipIf(os.name == "nt", "creating a symlink needs privileges on native Windows")
    def test_create_new_files_does_not_follow_a_dangling_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            link = Path(directory) / "link"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                create_new_files({link: b"new"})
            self.assertFalse(target.exists())
            self.assertTrue(link.is_symlink())

    def test_a_failed_removal_does_not_hide_the_original_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second, third = (Path(directory) / name for name in ("first", "second", "third"))
            third.write_bytes(b"authored")
            real_unlink = Path.unlink

            def unlink(path, *args, **kwargs):
                if path == first:
                    raise PermissionError("locked")
                return real_unlink(path, *args, **kwargs)

            with patch("pathlib.Path.unlink", unlink), self.assertRaises(FileExistsError) as raised:
                create_new_files({first: b"new", second: b"new", third: b"new"})
            self.assertEqual(raised.exception.filename, str(third))
            self.assertEqual(raised.exception.__notes__, [f"could not remove created file(s) {first}"])
            self.assertTrue(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(third.read_bytes(), b"authored")

    def test_append_line_durably_fsyncs_before_returning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with patch("kura.fsio.os.fsync") as fsync, patch("kura.fsio._fsync_directory") as fsync_directory:
                append_line_durably(path, "one\n")
                append_line_durably(path, "two\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")
            self.assertEqual(fsync.call_count, 2)
            fsync_directory.assert_called_once_with(path.parent)

    def test_a_torn_final_line_does_not_swallow_the_next_event(self) -> None:
        from kura.executors.common import append_run_event, run_events

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "logs").mkdir()
            path = run_dir / "logs" / "events.jsonl"
            path.write_text('{"type": "first"}\n{"type": "tor', encoding="utf-8")
            append_run_event(run_dir, {"type": "after_crash"})
            self.assertEqual([event["type"] for event in run_events(run_dir)], ["first", "after_crash"])
            self.assertTrue(path.read_text(encoding="utf-8").endswith('\n'))

    def test_atomic_write_text_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text("old\n", encoding="utf-8")

            atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_atomic_write_json_uses_expected_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"

            atomic_write_json(path, {"state": "compiled", "outputs": ["モデル"]})

            text = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text)["outputs"], ["モデル"])
            self.assertTrue(text.endswith("\n"))

    def test_atomic_write_yaml_uses_workspace_dump_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"

            atomic_write_yaml(path, {"name": "テスト", "items": [1, 2]})

            text = path.read_text(encoding="utf-8")
            self.assertEqual(yaml.safe_load(text)["items"], [1, 2])
            self.assertIn("name: テスト", text)

    def test_atomic_write_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "index.jsonl"

            atomic_write_text(path, "{}\n")

            self.assertEqual([item.name for item in root.iterdir()], ["index.jsonl"])

    def test_atomic_write_cleans_temporary_file_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "status.json"
            path.write_text("old\n", encoding="utf-8")

            with patch("kura.fsio.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual([item.name for item in root.iterdir()], ["status.json"])

    @unittest.skipUnless(os.name == "nt", "msvcrt byte-range locks are Windows-only")
    def test_windows_lock_on_an_empty_file_excludes_a_second_holder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".locks" / "secrets.lock"
            with file_lock(path):
                self.assertEqual(path.stat().st_size, 0)
                with self.assertRaises(FileLockBusy):
                    with file_lock(path, blocking=False):
                        pass
            with file_lock(path, blocking=False):
                pass


if __name__ == "__main__":
    unittest.main()
