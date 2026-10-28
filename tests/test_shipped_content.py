from __future__ import annotations

import shutil
import subprocess
import tempfile
import re
import unittest
import zipfile
from pathlib import Path

from kura.shipped import SHIPPED_SKILLS, shipped_root

ROOT = Path(__file__).resolve().parents[1]


class ShippedContentTests(unittest.TestCase):
    def test_usage_skills_knowledge_and_samples_ship_in_the_package(self) -> None:
        root = shipped_root()
        self.assertEqual(sorted(path.name for path in (root / "skills").iterdir()), sorted(SHIPPED_SKILLS))
        # Kura ships no regrets list; each user keeps their own in knowledge/regrets.md.
        self.assertFalse((root / "knowledge" / "regrets.md").exists())
        self.assertTrue(any((root / "knowledge" / "model-families").iterdir()))
        self.assertTrue((root / "workflow-samples" / "README.md").is_file())

    def test_agents_md_routes_every_skill(self) -> None:
        root = shipped_root()
        agents = (root / "AGENTS.md").read_text(encoding="utf-8")
        for skill in SHIPPED_SKILLS:
            self.assertIn(f"| `{skill}` |", agents, skill)

    def test_every_skill_has_the_parts_an_agent_follows(self) -> None:
        # Each task skill says who decides, and every skill's references exist.
        root = shipped_root()
        for name in SHIPPED_SKILLS:
            text = (root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
            with self.subTest(skill=name):
                if name != "writing":
                    self.assertIn("## Who decides", text)
                for reference in re.findall(r"references/[a-z]+\.md", text):
                    if "skill" not in text.split(reference, 1)[0].rsplit("\n", 1)[-1]:
                        self.assertTrue((root / "skills" / name / reference).is_file(), reference)

    def test_agents_are_told_never_to_print_a_container_environment(self) -> None:
        # Docker keeps a local run's HF_TOKEN in the container's configuration.
        root = shipped_root()
        agents = " ".join((root / "AGENTS.md").read_text(encoding="utf-8").split())
        self.assertIn("Do not show a container's environment either (`docker inspect`, `docker exec`, reading `/proc/<pid>/environ`)", agents)
        mentions = [
            path for path in root.rglob("*.md")
            for _ in range(path.read_text(encoding="utf-8").count("docker inspect"))
        ]
        self.assertEqual(mentions, [root / "AGENTS.md"])

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
