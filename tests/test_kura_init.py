from __future__ import annotations

import argparse
import io
import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.cli import cmd_init
from kura.environment import INTERNAL_VARIABLES, USER_VARIABLES, env_local_template

SRC = Path(__file__).resolve().parents[1] / "src" / "kura"


@contextmanager
def _inside(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _init(missing: list[str] | None = None) -> tuple[int, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("kura.init_templates.readiness_gaps", return_value=missing or []), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        code = cmd_init(argparse.Namespace())
    return code, stdout.getvalue() + stderr.getvalue()


class InitTests(unittest.TestCase):
    def test_creates_the_workspace_layout_and_user_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            code, output = _init()
            root = Path(directory)
            self.assertEqual(code, 0, output)
            for relative in ("workspace.yaml", "datasets", "runs", "cache", "workflows", "knowledge/model-families"):
                self.assertTrue((root / relative).exists(), relative)
            self.assertIn("# Regrets", (root / "knowledge" / "regrets.md").read_text(encoding="utf-8"))
            self.assertIn("# Preferences", (root / "knowledge" / "user-preferences.md").read_text(encoding="utf-8"))
            self.assertIn("kura dataset validate", output)
            self.assertNotIn("uv run", output)

    def test_env_local_template_names_every_user_variable_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            _init()
            template = Path(".env.local").read_text(encoding="utf-8")
        for variable in USER_VARIABLES:
            self.assertRegex(template, rf"(?m)^{variable.name}=$")
        self.assertEqual(template, env_local_template())

    def test_rerunning_init_changes_no_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            _init()
            root = Path(directory)
            (root / ".env.local").write_text("RUNPOD_API_KEY=mine\n", encoding="utf-8")
            (root / "knowledge" / "regrets.md").write_text("my regrets\n", encoding="utf-8")
            config = yaml.safe_load((root / "workspace.yaml").read_text(encoding="utf-8"))
            config["name"] = "renamed"
            (root / "workspace.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            code, _ = _init()
            after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.assertEqual(code, 0)
        self.assertEqual(before, after)

    def test_refuses_inside_or_below_an_existing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _inside(root):
                _init()
            nested = root / "datasets" / "inner"
            nested.mkdir(parents=True)
            with _inside(nested):
                code, output = _init()
            self.assertEqual(code, 1)
            self.assertIn(str(root.resolve()), output)
            self.assertFalse((nested / "workspace.yaml").exists())

    def test_an_older_schema_is_refused_with_the_migrate_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            Path("workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            code, output = _init()
            self.assertFalse(Path(".env.local").exists())
        self.assertEqual(code, 1)
        self.assertIn("kura workspace migrate", output)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_env_local_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            _init()
            self.assertEqual(Path(".env.local").stat().st_mode & 0o777, 0o600)

    def test_reports_only_what_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            code, output = _init(["Docker is not reachable: local training needs it; RunPod training does not"])
        self.assertEqual(code, 0)
        self.assertIn("still needed:", output)
        self.assertIn("Docker is not reachable", output)
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            _, output = _init([])
        self.assertIn("ready", output)
        self.assertNotIn("still needed", output)


class ReadinessTests(unittest.TestCase):
    def test_the_docker_probe_runs_outside_the_workspace(self) -> None:
        from kura.doctor import readiness_gaps

        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            with patch("kura.doctor.shutil.which", return_value="docker"), patch("kura.doctor.subprocess.run") as run:
                run.return_value.returncode = 0
                readiness_gaps(Path(directory))
        cwd = Path(run.call_args.kwargs["cwd"]).resolve()
        self.assertNotEqual(cwd, Path(directory).resolve())
        self.assertNotIn(Path(directory).resolve(), cwd.parents)


class EnvironmentDeclarationTests(unittest.TestCase):
    def test_every_environment_variable_kura_reads_is_declared(self) -> None:
        # Literal reads, plus names read through a loop over a tuple next to the read.
        read = set()
        direct = re.compile(r"(?:os\.environ(?:\.get)?\(|os\.environ\[|os\.getenv\()\s*\"([A-Z0-9_]+)\"")
        looped = re.compile(r"\bfor \w+ in \(([^)]*)\)")
        name = re.compile(r"\"([A-Z][A-Z0-9_]{2,})\"")
        for path in SRC.rglob("*.py"):
            if "container_scripts" in path.parts:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                read.update(direct.findall(line))
                if "os.environ" in line or "os.getenv" in line:
                    for nearby in lines[max(0, index - 2): index + 3]:
                        for group in looped.findall(nearby):
                            read.update(name.findall(group))
        declared = {variable.name for variable in USER_VARIABLES}
        aliases = {alias for variable in USER_VARIABLES for alias in variable.aliases}
        self.assertEqual(read - declared - aliases - INTERNAL_VARIABLES, set())


if __name__ == "__main__":
    unittest.main()
