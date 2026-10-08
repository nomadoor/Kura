"""A new run's files hold only what its author fills in or Kura reads, so an agent is not left guessing."""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.cli import cmd_run_new


class RunTemplateTests(unittest.TestCase):
    def test_a_new_train_run_has_no_field_nothing_reads_or_compile_fills(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    cmd_run_new(argparse.Namespace(experiment="e", slug="s", backend="ai-toolkit", executor="docker", gpu=None))
                run = yaml.safe_load((root / "runs" / stdout.getvalue().strip() / "run.yaml").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous)
        self.assertNotIn("created_by", run)  # nothing reads it, and an agent is not a human
        self.assertEqual(run["datasets"], [{"id": ""}])
        self.assertNotIn("version", run["backend"])  # nothing reads it  # compile fills the digest; role is optional


    def test_a_change_after_compile_starts_from_the_compiled_runs_settings(self) -> None:
        from kura.cli import cmd_run_compile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            source = root / "runs" / "20261008-2355_smoke-test_4a9f"
            source.mkdir(parents=True)
            run = {"schema_version": 2, "id": source.name, "type": "train", "experiment": "vivi", "created": "2026-10-08T23:55:00+09:00",
                   "created_by": "human", "intent": "smoke", "backend": {"name": "ai-toolkit", "config": {"learning_rate": 0.0001}},
                   "model": {"base": "stabilityai/sdxl"}, "datasets": [{"id": "vivi"}], "recipe": {"steps": 50, "seed": 42},
                   "compute": {"executor": "docker"}}
            (source / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (source / "status.json").write_text('{"state": "compiled"}', encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stderr", new_callable=io.StringIO) as refused:
                    self.assertEqual(cmd_run_compile(argparse.Namespace(run_id=source.name)), 1)
                with patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as said:
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment=None, slug="smoke-test-gc", backend=None, executor=None, gpu=None, source=source.name)), 0)
                new_id = stdout.getvalue().strip()
                copied = yaml.safe_load((root / "runs" / new_id / "run.yaml").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous)
        self.assertIn(f"kura run new --from {source.name}", refused.getvalue())
        self.assertIn(f"runs/{new_id}/run.yaml", said.getvalue())
        self.assertEqual((copied["id"], copied["experiment"], copied["backend"], copied["recipe"]), (new_id, "vivi", run["backend"], run["recipe"]))
        self.assertNotEqual(copied["created"], run["created"])
        self.assertNotIn("created_by", copied)

if __name__ == "__main__":
    unittest.main()
