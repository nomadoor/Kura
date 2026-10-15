"""Dataset files are closed and read in one form."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from kura.backends import validate_backend_config
from kura.dataset_manifest import draft_manifest, measure_manifest
from kura.yamlio import dump_yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend_fixtures import with_required  # noqa: E402


def _dataset(root: Path, metadata: str, reference: dict | None = None) -> Path:
    directory = root / "tiny"
    (directory / "images").mkdir(parents=True)
    (directory / "images" / "a.png").write_bytes(b"png")
    (directory / "dataset.yaml").write_text(metadata, encoding="utf-8")
    row = {"id": "a", "files": [reference or {"type": "file", "role": "target", "path": "images/a.png"}], "caption": {"text": "x"}}
    (directory / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return directory


class DatasetFileTests(unittest.TestCase):
    def test_a_v2_dataset_yaml_refuses_keys_kura_does_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            accepted = _dataset(Path(directory), "items_schema_version: 2\nid: tiny\ndescription: free text\ntrigger_word: vivi\nstats: {count: 1}\n")
            measure_manifest(accepted)
        for metadata, message in (
            ("items_schema_version: 2\nlayout: {image_dir: images}\n", "layout is not a dataset.yaml setting"),
            ("items_schema_version: 2\nmodality: image\n", "modality is not a dataset.yaml setting"),
            ("items_schema_version: 2\ntrigger_word: 3\n", "trigger_word must be a non-empty string"),
            ("items_schema_version: 2\nstats: {count: 1, mean: 2}\n", "stats takes only count"),
        ):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    measure_manifest(_dataset(Path(directory), metadata))

    def test_a_file_digest_has_one_form(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            import hashlib

            digest = hashlib.sha256(b"png").hexdigest()
            reference = {"type": "file", "role": "target", "path": "images/a.png", "sha256": digest}
            measure_manifest(_dataset(Path(directory), "items_schema_version: 2\n", reference))
        with tempfile.TemporaryDirectory() as directory:
            reference = {"type": "file", "role": "target", "path": "images/a.png", "sha256": f"sha256:{digest}"}
            with self.assertRaisesRegex(ValueError, "sha256 must be 64 lowercase hex digits with no prefix"):
                measure_manifest(_dataset(Path(directory), "items_schema_version: 2\n", reference))

    def test_a_draft_candidate_leaves_out_the_keys_v2_does_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proposal = draft_manifest(_dataset(Path(directory), "schema_version: 1\nid: tiny\nlayout: {image_dir: images}\ntrigger_word: vivi\n"))
        self.assertEqual(proposal["dataset_yaml"], {"id": "tiny", "trigger_word": "vivi", "items_schema_version": 2})
        self.assertIn("left out of the candidate: layout, schema_version", " ".join(proposal["issues"]))

    def test_sd_scripts_dataset_settings_have_one_form(self) -> None:
        def run(dataset_config: dict) -> dict:
            return {"backend": {"name": "sd-scripts", "config": with_required("sd-scripts", {
                "architecture": "sd15", "model_paths": {"base": "/models/base"}, "dataset_config": dataset_config,
            })}, "datasets": [{"id": "tiny"}]}

        subsets = [{"dataset_id": "tiny", "num_repeats": 1}]
        for general, message in (
            ({"resolution": [512]}, r"resolution must be a positive integer or \[width, height\]"),
            ({"resolution": None}, "written with no value: resolution"),
        ):
            with self.subTest(general=general), self.assertRaisesRegex(ValueError, message):
                validate_backend_config(run({"general": general, "datasets": [{"subsets": subsets}]}))
        validate_backend_config(run({"general": {"resolution": 512}, "datasets": [{"subsets": subsets}]}))
        for config, message in (
            ({"datasets": [{"subsets": [{"num_repeats": 1}]}]}, r"datasets\[0\].subsets\[0\] must name its manifest dataset with dataset_id"),
            ({"general": None, "datasets": [{"subsets": subsets}]}, "dataset_config.general must be a mapping"),
            ({"datasets": [{"subsets": [None]}]}, r"subsets\[0\] must be a mapping"),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                validate_backend_config(run(config))

    def test_a_drafted_legacy_dataset_validates(self) -> None:
        import hashlib

        digest = hashlib.sha256(b"png").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory), "schema_version: 1\nid: tiny\nlayout: {image_dir: images}\nstats: {count: 1, person_count: 1}\n")
            (dataset / "items.jsonl").write_text(json.dumps({"id": "a", "path": "images/a.png", "caption": "x", "hash": f"sha256:{digest}"}) + "\n", encoding="utf-8")
            proposal = draft_manifest(dataset)
            self.assertEqual(proposal["dataset_yaml"], {"id": "tiny", "stats": {"count": 1}, "items_schema_version": 2})
            self.assertIn("stats.person_count", " ".join(proposal["issues"]))
            self.assertEqual(proposal["items"][0]["files"][0]["sha256"], digest)
            (dataset / "dataset.yaml").write_text(dump_yaml(proposal["dataset_yaml"]), encoding="utf-8")
            (dataset / "items.jsonl").write_text("".join(json.dumps(item) + "\n" for item in proposal["items"]), encoding="utf-8")
            measure_manifest(dataset)

    def test_a_v2_dataset_yaml_with_refused_keys_gets_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proposal = draft_manifest(_dataset(Path(directory), "items_schema_version: 2\nid: tiny\nmodality: image\n"))
        self.assertEqual(proposal["dataset_yaml"], {"items_schema_version": 2, "id": "tiny"})


if __name__ == "__main__":
    unittest.main()
