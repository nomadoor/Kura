from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.yamlio import dump_yaml, load_yaml_mapping, load_yaml_text


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

    def test_load_mapping_refuses_a_non_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text("- a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a YAML mapping"):
                load_yaml_mapping(path)


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
