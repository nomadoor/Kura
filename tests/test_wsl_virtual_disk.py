"""On WSL2 a disk refusal says when the WSL virtual disk, not the Windows drive behind it, is
the limit, from the one floor check every local gate calls (`storage_floor_problem`)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from kura.cli import cmd_doctor_disk
from kura.run_commands.plan import _local_launch_disk_preflight
from kura.storage import ensure_free_bytes

GIB = 1024**3
HINT = "the WSL virtual disk is the limit"


def _wsl(*, linux_free_gib: int, host_free_gib: int) -> contextlib.ExitStack:
    """A WSL host whose ext4 virtual disk has `linux_free_gib` free, stored on D: with `host_free_gib` free."""
    stack = contextlib.ExitStack()
    stack.enter_context(patch("kura.storage.is_wsl", return_value=True))
    stack.enter_context(patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4", "target": "/", "source": "/dev/sdd"}))
    stack.enter_context(patch("kura.storage._auto_wsl_host_drive", return_value="D:"))
    stack.enter_context(patch("kura.storage._windows_drive_free_bytes", return_value=host_free_gib * GIB))
    stack.enter_context(patch("kura.storage.shutil.disk_usage", return_value=Mock(total=1000 * GIB, used=900 * GIB, free=linux_free_gib * GIB)))
    stack.enter_context(patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu-24.04"}))
    return stack


class WslVirtualDiskHintTests(unittest.TestCase):
    def _launch_problem(self, root: Path) -> str:
        with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
            with self.assertRaises(ValueError) as caught:
                _local_launch_disk_preflight(root, {"type": "train"}, {"docker": {"min_free_gb": 100}})
        return str(caught.exception)

    def test_local_launch_names_the_virtual_disk_when_the_windows_drive_has_room(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _wsl(linux_free_gib=91, host_free_gib=1600):
            problem = self._launch_problem(Path(directory).resolve())
        self.assertIn(f"{HINT} (Linux sees 91 GiB free, D: has 1600 GiB)", problem)
        self.assertIn("wsl --manage Ubuntu-24.04 --resize <size>", problem)
        self.assertIn("free space inside Linux", problem)

    def test_local_launch_gives_no_hint_when_the_windows_drive_is_the_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _wsl(linux_free_gib=900, host_free_gib=50):
            problem = self._launch_problem(Path(directory).resolve())
        self.assertIn("50 GiB effective free on D:", problem)
        self.assertNotIn(HINT, problem)

    def test_runpod_stage_and_pull_gate_names_the_virtual_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with _wsl(linux_free_gib=5, host_free_gib=1600), self.assertRaises(ValueError) as caught:
                ensure_free_bytes(target, 10 * GIB, context="RunPod stage", config={})
            self.assertIn(f"{HINT} (Linux sees 5 GiB free, D: has 1600 GiB)", str(caught.exception))
            self.assertIn("RunPod stage requires at least 10 GiB", str(caught.exception))
            with _wsl(linux_free_gib=900, host_free_gib=5), self.assertRaises(ValueError) as caught:
                ensure_free_bytes(target, 10 * GIB, context="RunPod stage", config={})
            self.assertNotIn(HINT, str(caught.exception))

    def test_doctor_disk_names_the_virtual_disk(self) -> None:
        for linux_free_gib, host_free_gib, hinted in ((50, 1600, True), (900, 50, False)):
            with self.subTest(hinted=hinted), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": [], "min_free_gb": 100}}), encoding="utf-8")
                (root / "cache").mkdir()
                (root / "runs").mkdir()
                previous = Path.cwd()
                os.chdir(root)
                try:
                    with (
                        _wsl(linux_free_gib=linux_free_gib, host_free_gib=host_free_gib),
                        patch("kura.doctor._path_size_bytes", return_value=0),
                        patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                        patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                        patch("sys.stdout", new_callable=io.StringIO) as stdout,
                    ):
                        cmd_doctor_disk(argparse.Namespace())
                finally:
                    os.chdir(previous)
                low = [issue for issue in json.loads(stdout.getvalue())["issues"] if issue["code"] == "workspace_effective_free_low"]
                self.assertEqual(len(low), 1)
                self.assertEqual(HINT in low[0]["message"], hinted)


if __name__ == "__main__":
    unittest.main()
