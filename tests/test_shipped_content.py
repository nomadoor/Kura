from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from kura.shipped import SHIPPED_SKILLS, shipped_root

ROOT = Path(__file__).resolve().parents[1]


class ShippedContentTests(unittest.TestCase):
    def test_usage_skills_knowledge_and_samples_ship_in_the_package(self) -> None:
        root = shipped_root()
        self.assertEqual(sorted(path.name for path in (root / "skills").iterdir()), sorted(SHIPPED_SKILLS))
        self.assertTrue((root / "knowledge" / "regrets.md").is_file())
        self.assertTrue(any((root / "knowledge" / "model-families").iterdir()))
        self.assertTrue((root / "workflow-samples" / "README.md").is_file())

    def test_agents_md_routes_every_skill_and_the_moved_rules_live_in_skills(self) -> None:
        root = shipped_root()
        agents = (root / "AGENTS.md").read_text(encoding="utf-8")
        routing = agents.split("## Which skill to read first", 1)[1]
        for skill in SHIPPED_SKILLS:
            self.assertIn(f"`{skill}` — before", routing, skill)

        def skill(name: str) -> str:
            return " ".join((root / "skills" / name / "SKILL.md").read_text(encoding="utf-8").split())

        moved = {
            "comfyui-render-workflow": ["Author one explicit `inputs.cases` JSONL queue", "intentional agent-owned presentation task"],
            "lora-evaluation": ["dataset-prep -> training-parameter-planning -> training -> lora-evaluation"],
            "training-parameter-planning": ["Training configuration and compute selection are one plan"],
            "dataset-prep": ["the dataset manifest inventories inputs"],
            "local-disk-safety": ["allow_many_checkpoints", "fix-permissions"],
        }
        for name, phrases in moved.items():
            for phrase in phrases:
                self.assertIn(phrase, skill(name), name)

    def test_development_skills_stay_in_the_repository(self) -> None:
        development = sorted(path.name for path in (ROOT / "dev" / "skills").iterdir())
        self.assertEqual(development, ["backend-upgrade-audit", "kura-core", "monitor-tui", "release-check", "training-backends"])
        self.assertFalse(set(development) & set(SHIPPED_SKILLS))

    @unittest.skipUnless(shutil.which("uv"), "building the wheel needs uv")
    def test_the_built_wheel_contains_the_shipped_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            built = subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", directory, str(ROOT)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr[-2000:])
            wheel = next(Path(directory).glob("kura-*.whl"))
            names = set(zipfile.ZipFile(wheel).namelist())
        source = ROOT / "src" / "kura" / "shipped"
        expected = {
            "kura/shipped/" + path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        self.assertEqual(expected - names, set())
        self.assertFalse([name for name in names if "__pycache__" in name or name.endswith(".pyc")])


if __name__ == "__main__":
    unittest.main()
