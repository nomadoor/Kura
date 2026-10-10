"""Public manifest-v2 validation contract."""

import argparse
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.cli import cmd_dataset_draft, cmd_dataset_validate
from kura.dataset_manifest import draft_manifest, measure_manifest, validate_manifest
from tests.platform_support import NATIVE_WINDOWS, DATASET_IO, posix_only


@posix_only(DATASET_IO)
class DatasetManifestTests(unittest.TestCase):
    def make_dataset(self, root: Path, rows: list[dict], metadata: str = "id: tiny\nitems_schema_version: 2\n") -> Path:
        dataset = root / "tiny"
        dataset.mkdir()
        (dataset / "dataset.yaml").write_text(metadata, encoding="utf-8")
        (dataset / "items.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return dataset

    def validate(self, dataset: Path) -> tuple[int, str]:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            result = cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset)))
        return result, stderr.getvalue()

    def test_typed_reference_and_inline_caption_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.make_dataset(root, [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}],
                 "caption": {"text": "hello"}}
            ])
            (dataset / "a.png").write_bytes(b"image")
            self.assertEqual(self.validate(dataset)[0], 0)

    def test_manifest_jsonl_uses_only_lf_as_the_row_separator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [])
            caption = "first\u2028second"
            row = {
                "id": "a",
                "files": [{"type": "file", "role": "target", "path": "a.png"}],
                "caption": {"text": caption},
            }
            (dataset / "items.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            (dataset / "a.png").write_bytes(b"image")

            measured = measure_manifest(dataset)

            self.assertEqual(measured["identity"]["samples"][0]["caption"], caption)

    def test_declared_id_must_match_the_dataset_directory(self) -> None:
        row = {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}], "caption": {"text": "a"}}
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [row], metadata="id: .tiny.creating\nitems_schema_version: 2\n")
            (dataset / "a.png").write_bytes(b"png")
            with self.assertRaisesRegex(ValueError, "does not match the dataset directory 'tiny'"):
                measure_manifest(dataset)
            (dataset / "dataset.yaml").write_text("items_schema_version: 2\n", encoding="utf-8")
            self.assertEqual(validate_manifest(dataset)[0], 1)  # an omitted id is not a mismatch
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
            link = Path(directory) / "linked"
            link.symlink_to(dataset, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "does not match the dataset directory 'linked'"):
                measure_manifest(link)  # judged by the selected name, not the symlink target

    def test_legacy_row_is_not_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [
                {"id": "a", "path": "a.png", "caption": "hello"}
            ])
            (dataset / "a.png").write_bytes(b"image")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("files", error)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [])
            (dataset / "items.jsonl").write_text(
                '{"id":"a","id":"b","files":[{"type":"file","role":"target","path":"a.png"}],"caption":null}\n',
                encoding="utf-8",
            )
            (dataset / "a.png").write_bytes(b"image")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("duplicate", error)

    def test_hash_assertion_and_path_escape_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.make_dataset(root, [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png", "sha256": "0" * 64}],
                 "caption": None},
            ])
            (dataset / "a.png").write_bytes(b"image")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("sha256", error)
            row = {"id": "a", "files": [{"type": "file", "role": "target", "path": "../outside.png"}], "caption": None}
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("path", error)

    def test_missing_v2_version_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [], "id: tiny\n")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("items_schema_version", error)

    def test_unlisted_media_requires_explicit_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}], "caption": None}
            ])
            (dataset / "a.png").write_bytes(b"a")
            (dataset / "backup.png").write_bytes(b"backup")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("backup.png", error)
            (dataset / "dataset.yaml").write_text(
                "id: tiny\nitems_schema_version: 2\nexcluded_files: [backup.png]\n", encoding="utf-8"
            )
            self.assertEqual(self.validate(dataset)[0], 0)

    def test_excluded_directory_cannot_contain_selected_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "images/a.png"}], "caption": None}
            ], "id: tiny\nitems_schema_version: 2\nexcluded_directories: [images]\n")
            (dataset / "images").mkdir()
            (dataset / "images" / "a.png").write_bytes(b"a")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("excluded", error)

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.make_dataset(root, [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}], "caption": None}
            ])
            (root / "outside.png").write_bytes(b"outside")
            (dataset / "a.png").symlink_to(root / "outside.png")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("path must stay inside", error)

    def test_in_root_symlink_is_measured_by_logical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [{
                "id": "a",
                "files": [{"type": "file", "role": "target", "path": "alias.png"}],
                "caption": None,
            }])
            (dataset / "storage").mkdir()
            (dataset / "storage" / "a.png").write_bytes(b"image")
            (dataset / "alias.png").symlink_to("storage/a.png")
            measured = measure_manifest(dataset)
            self.assertEqual(measured["files"][0]["path"], "alias.png")
            self.assertEqual(
                measured["files"][0]["sha256"],
                "6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d",
            )

    def test_symlink_target_cannot_be_excluded_through_physical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [{
                "id": "a", "files": [{"type": "file", "role": "target", "path": "alias.png"}],
                "caption": None,
            }], "id: tiny\nitems_schema_version: 2\nexcluded_directories: [storage]\n")
            (dataset / "storage").mkdir()
            (dataset / "storage" / "a.png").write_bytes(b"image")
            (dataset / "alias.png").symlink_to("storage/a.png")
            with self.assertRaisesRegex(ValueError, "excluded"):
                measure_manifest(dataset)

    def test_typed_input_hidden_in_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}],
                 "caption": None, "metadata": {"secret_input": {"type": "file", "path": "b.png"}}}
            ])
            (dataset / "a.png").write_bytes(b"a")
            (dataset / "b.png").write_bytes(b"b")
            code, error = self.validate(dataset)
            self.assertEqual(code, 1)
            self.assertIn("metadata", error)

    def test_identity_uses_content_and_effective_caption_not_stat_or_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [
                {"id": "a", "files": [
                    {"type": "file", "role": "target", "path": "a.png"},
                    {"type": "file", "role": "control", "path": "c.png"},
                ], "caption": {"file": {"type": "file", "path": "a.txt"}}, "metadata": {"note": "first"}}
            ])
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "c.png").write_bytes(b"control")
            (dataset / "a.txt").write_text("hello\n", encoding="utf-8")
            first = measure_manifest(dataset)
            (dataset / "a.png").touch()
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["metadata"] = {"note": "second"}
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            second = measure_manifest(dataset)
            self.assertEqual(first["identity_sha256"], second["identity_sha256"])
            self.assertNotEqual(first["files"][0]["stat"], second["files"][0]["stat"])
            (dataset / "a.txt").write_text("different\n", encoding="utf-8")
            third = measure_manifest(dataset)
            self.assertNotEqual(second["identity_sha256"], third["identity_sha256"])

    def test_draft_preview_is_read_only_and_write_creates_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            (dataset / "images").mkdir(parents=True)
            (dataset / "images" / "a.png").write_bytes(b"image")
            (dataset / "images" / "a.txt").write_text("hello\n", encoding="utf-8")
            (dataset / "dataset.yaml").write_text("id: tiny\n", encoding="utf-8")
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=False)), 0)
            preview = json.loads(stdout.getvalue())
            self.assertEqual(preview["items"][0]["files"][0]["path"], "images/a.png")
            self.assertEqual(preview["items"][0]["id"], "a")
            self.assertEqual(preview["items"][0]["caption"], {"file": {"type": "file", "path": "images/a.txt"}})
            self.assertFalse((dataset / "items.v2.candidate.jsonl").exists())
            self.assertFalse((dataset / "items.jsonl").exists())
            self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 0)
            # dataset.yaml lacks items_schema_version: 2, so its proposal stays a candidate.
            self.assertEqual((dataset / "dataset.yaml").read_text(encoding="utf-8"), "id: tiny\n")
            self.assertTrue((dataset / "dataset.v2.candidate.yaml").exists())
            self.assertTrue((dataset / "items.jsonl").exists())
            self.assertFalse((dataset / "items.v2.candidate.jsonl").exists())
            self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 1)

    def test_draft_write_on_a_fresh_v2_dataset_writes_items_jsonl_that_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            dataset.mkdir()
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "a.txt").write_text("hello\n", encoding="utf-8")
            authored = "id: tiny\nitems_schema_version: 2\n"
            (dataset / "dataset.yaml").write_text(authored, encoding="utf-8")
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 0)
            self.assertEqual((dataset / "dataset.yaml").read_text(encoding="utf-8"), authored)
            self.assertFalse((dataset / "dataset.v2.candidate.yaml").exists())
            self.assertFalse((dataset / "items.v2.candidate.jsonl").exists())
            self.assertTrue((dataset / "items.jsonl").exists())
            self.assertEqual(self.validate(dataset), (0, ""))

    def test_draft_write_never_replaces_an_existing_items_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            dataset.mkdir()
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "a.txt").write_text("hello\n", encoding="utf-8")
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
            authored = json.dumps({"id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}],
                                   "caption": {"text": "authored"}}) + "\n"
            (dataset / "items.jsonl").write_text(authored, encoding="utf-8")
            stdout = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", io.StringIO()):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 0)
            self.assertEqual((dataset / "items.jsonl").read_text(encoding="utf-8"), authored)
            self.assertTrue((dataset / "items.v2.candidate.jsonl").exists())
            self.assertFalse((dataset / "dataset.v2.candidate.yaml").exists())
            self.assertIn("review items.v2.candidate.jsonl, then move it over items.jsonl", stdout.getvalue())

    def test_draft_write_with_review_issues_writes_only_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            dataset.mkdir()
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "a.txt").write_text("hello\n", encoding="utf-8")
            (dataset / "a.caption").write_text("other\n", encoding="utf-8")
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
            stderr = io.StringIO()
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 0)
            self.assertIn("multiple same-stem captions", stderr.getvalue())
            self.assertFalse((dataset / "items.jsonl").exists())
            self.assertTrue((dataset / "items.v2.candidate.jsonl").exists())

    def test_draft_write_with_an_ambiguous_image_root_writes_no_items_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            (dataset / "images").mkdir(parents=True)
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "images" / "b.png").write_bytes(b"image")
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 0)
            self.assertFalse((dataset / "items.jsonl").exists())
            self.assertTrue((dataset / "items.v2.candidate.jsonl").exists())

    def test_a_float_items_schema_version_is_not_declared_for_draft_or_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            dataset.mkdir()
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "a.txt").write_text("hello\n", encoding="utf-8")
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2.0\n", encoding="utf-8")
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=True)), 0)
            self.assertTrue((dataset / "dataset.v2.candidate.yaml").exists())
            self.assertIn("review dataset.v2.candidate.yaml, then move it over dataset.yaml", stdout.getvalue())
            self.assertTrue((dataset / "items.jsonl").exists())
            result, error = self.validate(dataset)
            self.assertEqual(result, 1)
            self.assertIn("items_schema_version: 2", error)

    def test_draft_imports_unambiguous_legacy_id_caption_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "Vivi"
            dataset.mkdir()
            (dataset / "001.png").write_bytes(b"image")
            digest = "sha256:6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d"
            (dataset / "dataset.yaml").write_text("id: Vivi\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(
                json.dumps({"id": "001", "path": "001.png", "caption": "Vivi, full body", "hash": digest}) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                self.assertEqual(cmd_dataset_draft(argparse.Namespace(dataset_dir=str(dataset), write=False)), 0)
            preview = json.loads(stdout.getvalue())
            self.assertEqual(preview["items"], [{
                "id": "001",
                "files": [{"type": "file", "role": "target", "path": "001.png", "sha256": digest}],
                "caption": {"text": "Vivi, full body"},
            }])
            self.assertNotIn("not imported", " ".join(preview["issues"]))

    def test_legacy_importer_uses_only_lf_as_the_row_separator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "legacy-unicode-line"
            dataset.mkdir()
            caption = "first\u2028second"
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "dataset.yaml").write_text("id: legacy-unicode-line\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(
                json.dumps({"id": "a", "path": "a.png", "caption": caption}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            preview = draft_manifest(dataset)

            self.assertEqual(preview["items"][0]["caption"], {"text": caption})
            self.assertEqual(preview["issues"], [])

    def test_caption_file_preserves_crlf_after_utf8_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [{
                "id": "a",
                "files": [{"type": "file", "role": "target", "path": "a.png"}],
                "caption": {"file": {"type": "file", "path": "a.txt"}},
            }])
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "a.txt").write_bytes(b"first\r\nsecond\r\n")
            measured = measure_manifest(dataset)
            self.assertEqual(measured["identity"]["samples"][0]["caption"], "first\r\nsecond\r\n")

    def test_draft_imports_explicit_legacy_control_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "paired"
            (dataset / "images").mkdir(parents=True)
            (dataset / "conditioning").mkdir()
            (dataset / "images" / "a.png").write_bytes(b"target")
            (dataset / "conditioning" / "a.png").write_bytes(b"control")
            (dataset / "dataset.yaml").write_text("id: paired\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(json.dumps({
                "id": "a", "path": "images/a.png", "control_path": "conditioning/a.png",
                "caption": "paired caption",
            }) + "\n", encoding="utf-8")
            preview = draft_manifest(dataset)
            self.assertEqual(preview["items"][0]["files"], [
                {"type": "file", "role": "target", "path": "images/a.png"},
                {"type": "file", "role": "control", "path": "conditioning/a.png"},
            ])
            self.assertNotIn("conditioning/a.png", " ".join(preview["issues"]))

    def test_draft_imports_legacy_caption_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "caption-path"
            (dataset / "captions").mkdir(parents=True)
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "captions" / "a.txt").write_text("caption from file", encoding="utf-8")
            (dataset / "dataset.yaml").write_text("id: caption-path\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(json.dumps({
                "id": "a", "path": "a.png", "caption_path": "captions/a.txt",
            }) + "\n", encoding="utf-8")
            preview = draft_manifest(dataset)
            self.assertEqual(preview["items"][0]["caption"], {
                "file": {"type": "file", "path": "captions/a.txt"}
            })
            self.assertEqual(preview["issues"], [])

    def test_draft_refuses_conflicting_inline_and_sidecar_captions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "caption-conflict"
            dataset.mkdir()
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "a.txt").write_text("sidecar caption", encoding="utf-8")
            (dataset / "dataset.yaml").write_text("id: caption-conflict\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(json.dumps({
                "id": "a", "path": "a.png", "caption": "items caption",
            }) + "\n", encoding="utf-8")
            preview = draft_manifest(dataset)
            self.assertIsNone(preview["items"][0]["caption"])
            self.assertIn("caption", " ".join(preview["issues"]))

    def test_draft_never_imports_a_path_repeated_three_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "duplicate-path"
            dataset.mkdir()
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "dataset.yaml").write_text("id: duplicate-path\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(
                "".join(
                    json.dumps({"id": sample_id, "path": "a.png", "caption": caption}) + "\n"
                    for sample_id, caption in (
                        ("first", "first caption"),
                        ("second", "second caption"),
                        ("third", "third caption"),
                    )
                ),
                encoding="utf-8",
            )
            preview = draft_manifest(dataset)
            self.assertEqual(preview["items"][0]["id"], "a")
            self.assertIsNone(preview["items"][0]["caption"])
            self.assertIn("duplicate legacy path", " ".join(preview["issues"]))

    def test_caption_paths_participate_in_casefold_collision_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [{
                "id": "a", "files": [{"type": "file", "role": "target", "path": "A.txt"}],
                "caption": {"file": {"type": "file", "path": "a.TXT"}},
            }])
            (dataset / "A.txt").write_text("target", encoding="utf-8")
            (dataset / "a.TXT").write_text("caption", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "case-folded path collision"):
                measure_manifest(dataset)

    def test_hash_measurement_rejects_file_changed_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [{
                "id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}],
                "caption": None,
            }])
            image = dataset / "a.png"
            image.write_bytes(b"original")
            real_digest = hashlib.sha256()

            class MutatingDigest:
                def update(self, chunk: bytes) -> None:
                    real_digest.update(chunk)
                    image.write_bytes(b"changed")

                def hexdigest(self) -> str:
                    return real_digest.hexdigest()

            with patch("kura.dataset_manifest.hashlib.sha256", return_value=MutatingDigest()):
                with self.assertRaisesRegex(ValueError, "changed during sha256"):
                    measure_manifest(dataset)

    def test_unknown_extension_is_reported_as_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), [{
                "id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}],
                "caption": None,
            }])
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "mystery.data").write_bytes(b"unknown")
            count, warnings = validate_manifest(dataset)
            self.assertEqual(count, 1)
            self.assertIn("mystery.data", " ".join(warnings))

    def test_validate_accepts_dataset_id_and_displays_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            datasets = workspace / "datasets"
            datasets.mkdir()
            dataset = self.make_dataset(datasets, [{
                "id": "a", "files": [{"type": "file", "role": "target", "path": "a.png"}],
                "caption": None,
            }], "id: tiny\nitems_schema_version: 2\nexcluded_files: [backup.png]\n")
            (dataset / "a.png").write_bytes(b"image")
            (dataset / "backup.png").write_bytes(b"backup")
            stdout = io.StringIO()
            with patch("kura.cli._workspace", return_value=workspace), patch("sys.stdout", stdout):
                self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir="tiny")), 0)
            self.assertIn("excluded files (1): backup.png", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()


class NativeWindowsRefusalTests(unittest.TestCase):
    @unittest.skipUnless(NATIVE_WINDOWS, "checks the explicit native-Windows refusal")
    def test_native_windows_refuses_dataset_metadata_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "example"
            root.mkdir()
            (root / "dataset.yaml").write_text("id: example\nitems_schema_version: 2\n", encoding="utf-8")
            (root / "items.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "this platform cannot safely open dataset metadata"):
                measure_manifest(root)
