"""A RunPod collection brings home everything the run wrote on the Pod, recovery copies and the trainer's
not-yet-published weights in its `cache/` included, so RunPod keeps what Docker keeps (P7)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura import reclaim
from kura.backends import trainer_cache_outputs
from kura.backends.sd_scripts import _anima_conversion_command
from kura.run_commands import runpod_ssh


def _inventory(workspace: Path, run_id: str) -> list[str]:
    """Run the remote inventory script on this host, as the Pod would, and return the paths it lists."""

    def run_on_this_host(command: list[str], **kwargs):
        script = command[-1]
        body = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        return subprocess.run([sys.executable, "-c", body], check=False, text=True, capture_output=True)

    with patch.object(runpod_ssh, "_ssh_base", return_value=[]), patch.object(runpod_ssh, "_run_bounded", side_effect=run_on_this_host):
        return [item["path"] for item in runpod_ssh._runpod_remote_snapshot_manifest({}, workspace=workspace.as_posix(), run_id=run_id)]


def _write(run: Path, relatives: list[str]) -> None:
    for relative in relatives:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _docker_keeps(run: Path) -> set[str]:
    """The files a finished local Docker run keeps: everything under the run except what reclaim frees."""
    freed = reclaim._scratch_paths(run)
    return {path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()
            and not any(path == free or free in path.parents for free in freed)}


class RecoveryCollectionTests(unittest.TestCase):
    def test_the_terminal_snapshot_lists_the_recovery_copy_and_the_unpublished_weights(self) -> None:
        run_id = "anima"
        spec = json.loads(_anima_conversion_command(run_id, "lora", [])[-1])
        run_root = f"/workspace/runs/{run_id}/"
        recovery = spec["recovery_dir"]
        self.assertTrue(recovery.startswith(run_root))
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run = workspace / "runs" / run_id
            written = {
                f"{recovery[len(run_root):]}/lora.safetensors": True,
                "outputs/lora.safetensors": True,
                "logs/stdout.log": True,
                "realizations/remote-exit-1.json": True,
                # Weights Anima trained before the wrapper converted them: a wrapper killed before
                # conversion leaves the only copy here.
                f"{spec['native_dir'][len(run_root):]}/lora-step00000010.safetensors": True,
                f"{spec['native_dir'][len(run_root):]}/sub/lora.safetensors": True,
                # The trainer's working caches and the upload area never come home.
                f"{spec['staging_dir'][len(run_root):]}/lora.safetensors": False,
                "cache/sd-scripts/notes.txt": False,
                "cache/latents/a.npz": False,
                "transfer/upload.tar": False,
            }
            _write(run, list(written))
            self.assertEqual(sorted(_inventory(workspace, run_id)), sorted(path for path, kept in written.items() if kept))

    def test_runpod_brings_home_from_cache_and_transfer_what_a_docker_run_keeps(self) -> None:
        # The same files in cache/ and transfer/, written by a run on the Pod and by a local Docker run,
        # end up kept on both (tmp/ and .cache/ differ: see docs/backlog.md).
        written = ["outputs/lora.safetensors", "logs/stdout.log", "recovery/sd-scripts/anima-native/lora.safetensors",
                   "cache/sd-scripts/native-output/lora-step00000010.safetensors", "cache/sd-scripts/converted-output/x.safetensors",
                   "cache/latents/a.npz", "transfer/upload.tar"]
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run = workspace / "runs" / "same"
            _write(run, written)
            self.assertEqual(set(_inventory(workspace, "same")), _docker_keeps(run))

    @unittest.skipIf(sys.platform == "win32", "the inventory runs on the Pod's Linux; links need privileges on Windows")
    def test_a_link_in_the_cache_is_passed_over_and_never_fails_the_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run = workspace / "runs" / "linked"
            _write(run, ["outputs/lora.safetensors", "cache/sd-scripts/native-output/a.safetensors", "elsewhere/b.safetensors"])
            (run / "cache" / "sd-scripts" / "native-output" / "link.safetensors").symlink_to(run / "elsewhere" / "b.safetensors")
            (run / "cache" / "sd-scripts" / "native-output" / "dir").symlink_to(run / "elsewhere", target_is_directory=True)
            self.assertEqual(sorted(_inventory(workspace, "linked")),
                             ["cache/sd-scripts/native-output/a.safetensors", "elsewhere/b.safetensors", "outputs/lora.safetensors"])
            other = workspace / "runs" / "cache-linked"
            _write(other, ["outputs/lora.safetensors"])
            (other / "cache").symlink_to(run / "cache", target_is_directory=True)
            self.assertEqual(_inventory(workspace, "cache-linked"), ["outputs/lora.safetensors"])

    def test_both_executors_read_the_one_owner(self) -> None:
        self.assertIn("cache/sd-scripts/native-output", trainer_cache_outputs())
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run = workspace / "runs" / "owner"
            _write(run, ["cache/other-trainer/weights/a.safetensors", "cache/sd-scripts/native-output/b.safetensors"])
            with patch.object(runpod_ssh, "trainer_cache_outputs", return_value=("cache/other-trainer/weights",)), \
                    patch.object(reclaim, "trainer_cache_outputs", return_value=("cache/other-trainer/weights",)):
                self.assertEqual(_inventory(workspace, "owner"), ["cache/other-trainer/weights/a.safetensors"])
                self.assertEqual(_docker_keeps(run), {"cache/other-trainer/weights/a.safetensors"})


if __name__ == "__main__":
    unittest.main()
