from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sync_agent_skills import compare_trees, sync_mirror, validate_skills


SKILL = """---
name: example-skill
description: Use for representative skill validation tests.
---

# Example skill

Do the task.
"""

OPENAI = """interface:
  display_name: "Example Skill"
  short_description: "Validate a representative project skill"
  default_prompt: "Use $example-skill to validate this example."
"""


class SkillLayoutTests(unittest.TestCase):
    def _tree(self, root: Path) -> Path:
        skills = root / "skills"
        skill = skills / "example-skill"
        (skill / "agents").mkdir(parents=True)
        (skill / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (skill / "agents" / "openai.yaml").write_text(OPENAI, encoding="utf-8")
        return skills

    def test_validates_skill_and_ui_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(validate_skills(self._tree(Path(directory))), [])

    def test_reports_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            skills = self._tree(Path(directory))
            (skills / "example-skill" / "agents" / "openai.yaml").unlink()
            self.assertTrue(any("missing agents/openai.yaml" in error for error in validate_skills(skills)))

    def test_sync_replaces_stale_mirror_and_parity_check_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = self._tree(root / "canonical")
            mirror = self._tree(root / "mirror")
            stale = mirror / "example-skill" / "SKILL.md"
            stale.write_text(SKILL + "\nStale.\n", encoding="utf-8")
            extra = mirror / "obsolete-skill" / "SKILL.md"
            extra.parent.mkdir()
            extra.write_text(SKILL, encoding="utf-8")
            self.assertTrue(compare_trees(canonical, mirror))
            sync_mirror(canonical, mirror)
            self.assertEqual(compare_trees(canonical, mirror), [])
            self.assertFalse(extra.exists())

    def test_interrupted_sync_keeps_existing_mirror_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = self._tree(root / "canonical")
            mirror = self._tree(root / "mirror")
            existing = mirror / "example-skill" / "SKILL.md"
            with (
                patch("scripts.sync_agent_skills.shutil.copy2", side_effect=RuntimeError("copy interrupted")),
                self.assertRaisesRegex(RuntimeError, "copy interrupted"),
            ):
                sync_mirror(canonical, mirror)
            self.assertEqual(existing.read_text(encoding="utf-8"), SKILL)


if __name__ == "__main__":
    unittest.main()
