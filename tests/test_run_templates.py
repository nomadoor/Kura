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
        self.assertEqual(run["datasets"], [{"id": ""}])  # compile fills the digest; role is optional


if __name__ == "__main__":
    unittest.main()
