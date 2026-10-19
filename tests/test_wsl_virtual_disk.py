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


class RequiredFigureTests(unittest.TestCase):
    def test_the_required_figure_is_rounded_up_and_shows_a_fraction(self) -> None:
        from kura.storage import StorageStatus, storage_floor_problem

        status = StorageStatus(path="x", probe="x", backing_id="C:", backing_kind="native", linux_free_bytes=GIB,
                               linux_total_bytes=1000 * GIB, host_free_bytes=None, effective_free_bytes=GIB, confidence="exact", mount={})
        for required, shown in ((20 * GIB, "20 GiB"), (int(1.5 * GIB), "1.5 GiB"), (int(10.31 * GIB), "10.4 GiB"), (20 * GIB + 1, "20.1 GiB")):
            with self.subTest(shown=shown):
                problem = storage_floor_problem(status, "/w", required, action="a local Docker launch")
                self.assertIn(f"a local Docker launch requires at least {shown}", problem)


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


class SharedWindowsDriveTests(unittest.TestCase):
    """The workspace on the WSL virtual disk stored on D:, and the Hugging Face cache on /mnt/d: both grow D:,
    so D: holds the reserve once plus both stores' writes, and the virtual disk still holds its own."""

    def _launch(self, *, linux_free_gib: int, drive_free_gib: int, workspace_gib: int, hf_gib: int) -> None:
        from kura.run_commands import plan
        from kura.storage import StorageStatus

        def status(kind: str) -> StorageStatus:
            linux = linux_free_gib if kind == "wsl2_vhdx" else drive_free_gib
            return StorageStatus(path="x", probe="x", backing_id="D:", backing_kind=kind, linux_free_bytes=linux * GIB,
                                 linux_total_bytes=1000 * GIB, host_free_bytes=drive_free_gib * GIB if kind == "wsl2_vhdx" else None,
                                 effective_free_bytes=min(linux, drive_free_gib) * GIB,
                                 confidence="estimated" if kind == "wsl2_vhdx" else "exact", mount={})

        estimate = {"stores": {"workspace": workspace_gib * GIB, "hf_cache": hf_gib * GIB}, "downloads": {"bytes": hf_gib * GIB},
                    "own_writes": {"checkpoints": 1, "trainer_states": 0, "kura_state_copies": 0, "rank": 16, "weights_files": 1,
                                   "per_file_bytes": GIB, "bytes": GIB}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(plan, "docker_data_location", return_value=(None, "not measured in tests")), \
                patch.object(plan, "local_write_paths", return_value={"workspace": Path(directory), "hf_cache": Path(directory) / "hf"}), \
                patch.object(plan, "probe_storages", return_value={"workspace": status("wsl2_vhdx"), "hf_cache": status("wsl2_windows_drive")}), \
                patch.object(plan, "run_write_estimate", return_value=estimate), \
                patch.object(plan, "own_write_text", return_value="own writes"), \
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            plan._local_launch_disk_preflight(Path(directory), {"type": "train"}, {"docker": {"min_free_gb": 20}})

    def test_the_drive_holds_both_stores_writes(self) -> None:
        # Each store's writes fit alone (20 + 30 < 60), but D: needs 20 + 30 + 30.
        with self.assertRaises(ValueError) as caught:
            self._launch(linux_free_gib=900, drive_free_gib=60, workspace_gib=30, hf_gib=30)
        self.assertIn("the Windows drive D:", str(caught.exception))
        self.assertIn("80 GiB", str(caught.exception))
        self.assertNotIn(HINT, str(caught.exception))

    def test_writes_to_mnt_never_count_against_the_virtual_disk(self) -> None:
        # The virtual disk has 40 GiB and gets 5 GiB; the 100 GiB of /mnt/d writes land on D:, which has room.
        self._launch(linux_free_gib=40, drive_free_gib=900, workspace_gib=5, hf_gib=100)

    def test_the_virtual_disk_keeps_its_own_limit(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._launch(linux_free_gib=40, drive_free_gib=900, workspace_gib=30, hf_gib=1)
        self.assertIn(HINT, str(caught.exception))

    def test_a_virtual_disk_whose_drive_is_unknown_keeps_its_own_key(self) -> None:
        from kura.storage import StorageStatus

        status = StorageStatus(path="x", probe="x", backing_id="/dev/sdd", backing_kind="wsl2_vhdx", linux_free_bytes=GIB,
                               linux_total_bytes=GIB, host_free_bytes=None, effective_free_bytes=GIB, confidence="unknown", mount={})
        self.assertEqual(status.space_key, ("wsl2_vhdx", "/dev/sdd"))


if __name__ == "__main__":
    unittest.main()
