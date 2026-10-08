"""Both executors decide with one function whether a run's dataset handoff gets a terminal postflight."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class HandoffGateTests(unittest.TestCase):
    def test_both_executors_ask_the_same_question(self) -> None:
        from kura import dataset_handoff
        from kura.executors import docker, runpod

        self.assertIs(docker.handoff_was_frozen, dataset_handoff.handoff_was_frozen)
        self.assertIs(runpod.handoff_was_frozen, dataset_handoff.handoff_was_frozen)

    def test_a_frozen_handoff_is_a_schema_2_input_lock_or_one_that_cannot_be_read(self) -> None:
        from kura.dataset_handoff import handoff_was_frozen

        for label, content, expected in (
            ("no lock", None, False),
            ("explicit command lock", {"schema_version": 1}, False),
            ("managed handoff", {"schema_version": 2}, True),
            ("unreadable", "{", True),  # recorded as uncheckable rather than skipped
        ):
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                (run_dir / "resolved").mkdir()
                if content is not None:
                    text = content if isinstance(content, str) else json.dumps(content)
                    (run_dir / "resolved" / "dataset-input.lock.json").write_text(text, encoding="utf-8")
                self.assertEqual(handoff_was_frozen(run_dir), expected)


if __name__ == "__main__":
    unittest.main()
