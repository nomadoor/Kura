from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.yamlio import YAMLError, dump_yaml, load_run_yaml, load_yaml_mapping, load_yaml_text


class YamlReaderTests(unittest.TestCase):
    def test_reader_follows_yaml_1_2_core_scalars(self) -> None:
        cases = {
            "1e-4": 1e-4, "1.0e-4": 1e-4, "-2E3": -2000.0, ".5": 0.5, "12": 12, "010": 10, "0o17": 15,
            "0x1F": 31, "true": True, "False": False, "~": None, ".inf": float("inf"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                value = load_yaml_text(f"value: {text}\n")["value"]
                self.assertEqual(value, expected)
                self.assertIs(type(value), type(expected))

    def test_reader_keeps_yaml_1_1_only_scalars_as_strings(self) -> None:
        for text in ("on", "off", "yes", "no", "Y", "n", "2026-10-14", "1_000", "0b101", "1:30"):
            with self.subTest(text=text):
                self.assertEqual(load_yaml_text(f"value: {text}\n")["value"], text)

    def test_an_explicit_int_tag_keeps_pyyaml_reading(self) -> None:
        self.assertEqual(load_yaml_text("a: !!int 0b101\nb: !!int 1:30\n"), {"a": 5, "b": 90})

    def test_reader_refuses_a_duplicate_mapping_key(self) -> None:
        with self.assertRaisesRegex(YAMLError, "duplicate key 'learning_rate'"):
            load_yaml_text("learning_rate: 1e-4\nlearning_rate: 2e-4\n")
        with self.assertRaisesRegex(YAMLError, "duplicate key 'a'"):
            load_yaml_text("outer:\n  a: 1\n  a: 1\n")

    def test_reader_refuses_the_yaml_1_1_merge_key(self) -> None:
        with self.assertRaisesRegex(YAMLError, "merge key"):
            load_yaml_text("base: &base\n  a: 1\nrun:\n  <<: *base\n  b: 2\n")

    def test_reader_keeps_anchors_and_aliases(self) -> None:
        self.assertEqual(load_yaml_text("a: &x [1, 2]\nb: *x\n"), {"a": [1, 2], "b": [1, 2]})

    def test_load_mapping_refuses_a_non_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text("- a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a YAML mapping"):
                load_yaml_mapping(path)


class RunRecordReaderTests(unittest.TestCase):
    def test_a_compiled_runs_yaml_is_read_as_older_kura_read_it_and_a_draft_strictly(self) -> None:
        from kura.monitor import collect_run_summaries

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "old"
            (run_dir / "resolved").mkdir(parents=True)
            run_yaml = run_dir / "run.yaml"
            run_yaml.write_text("id: old\ntype: train\nexperiment: a\nexperiment: b\n", encoding="utf-8")
            with self.assertRaisesRegex(YAMLError, "duplicate key"):
                load_run_yaml(run_yaml)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: old\ntype: train\n", encoding="utf-8")
            self.assertEqual(load_run_yaml(run_yaml)["experiment"], "b")
            self.assertEqual([summary.experiment for summary in collect_run_summaries(root)], ["b"])


class YamlWriterTests(unittest.TestCase):
    def test_what_the_writer_writes_reads_back_the_same(self) -> None:
        values = [
            "1e-4", "0o17", "on", "off", "yes", "no", "true", "null", "~", "12", "2026-10-14", ".inf", "1:30",
            1e-4, 1e-20, 0.5, 12, -3, True, False, None,
            {"nested": ["1e-4", 1e-4, "yes", {"k": "0x1F"}]},
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(load_yaml_text(dump_yaml({"value": value}))["value"], value)

    def test_a_yaml_1_1_reader_reads_the_writer_output_the_same(self) -> None:
        # Trainers and container scripts read Kura-written files with PyYAML's YAML 1.1 loader.
        for value in ("1e-4", "on", "yes", "2026-10-14", "0o17", 1e-4, True):
            with self.subTest(value=value):
                self.assertEqual(yaml.safe_load(dump_yaml({"value": value}))["value"], value)

    def test_writer_keeps_key_order_and_unicode(self) -> None:
        self.assertEqual(dump_yaml({"b": "日本", "a": 1}), "b: 日本\na: 1\n")

    def test_writer_quotes_dates_so_no_reader_turns_them_into_dates(self) -> None:
        text = dump_yaml({"created": "2026-10-14T12:00:00+09:00"})
        self.assertNotIsInstance(yaml.safe_load(text)["created"], datetime.datetime)


if __name__ == "__main__":
    unittest.main()
