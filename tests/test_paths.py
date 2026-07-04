from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kura.paths import inspect_workspace_symlinks, to_container, to_host


class PathNamespaceTests(unittest.TestCase):
    def test_workspace_relative_helpers_reject_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsafe workspace-relative path"):
                to_host("/etc/passwd", workspace)
            with self.assertRaisesRegex(ValueError, "unsafe workspace-relative path"):
                to_container("/etc/passwd")

    def test_symlink_inspection_ignores_malformed_mount_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            link = workspace / "cache" / "models" / "model.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to("/root/.cache/huggingface/model.safetensors")

            payload = inspect_workspace_symlinks(workspace, mounts=[False, {"source": "cache/huggingface", "target": "/root/.cache/huggingface"}])

            self.assertEqual(payload["unsafe"][0]["workspace_target"], "cache/huggingface/model.safetensors")


if __name__ == "__main__":
    unittest.main()
