from __future__ import annotations

import unittest

from scripts.check_no_artifacts import FORBIDDEN_PARTS, allowed


class ArtifactCheckTests(unittest.TestCase):
    def test_workspace_cases_are_forbidden_but_authored_examples_are_allowed(self) -> None:
        self.assertIn("cases", FORBIDDEN_PARTS)
        self.assertFalse(allowed("cases/private-comparison.jsonl"))
        self.assertTrue(allowed("examples/lora-evaluation/cases-comparison.jsonl"))


if __name__ == "__main__":
    unittest.main()
