from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_shipped_text import findings


class ShippedTextCheckTests(unittest.TestCase):
    def test_the_shipped_content_is_workspace_only(self) -> None:
        self.assertEqual(findings(), [])

    def test_repository_only_references_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SKILL.md").write_text(
                "Run `uv run kura doctor`.\nSee docs/commands.md.\nCall scripts/check.py and examples/x.yaml.\n"
                "See `../../../docs/smoke-evidence/x.yaml` and ./scripts/y.py.\n"
                "Read .kura/reference/external-access.md, runs/<id>/scripts/z.py, and https://example.com/docs/a.\n"
                "Read **docs/commands.md** first.\n| step | scripts/check.py |\n",
                encoding="utf-8",
            )
            import scripts.check_shipped_text as module

            previous = module.ROOT
            module.ROOT = root
            try:
                found = findings(root)
            finally:
                module.ROOT = previous
        self.assertEqual(len(found), 8, found)
        self.assertTrue(all("only exists in the repository" in item for item in found))


if __name__ == "__main__":
    unittest.main()
