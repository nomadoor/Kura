"""A RunPod collection brings home everything the run wrote on the Pod, recovery copies included (P7)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class RecoveryCollectionTests(unittest.TestCase):
    def test_the_terminal_snapshot_lists_the_recovery_copy_and_leaves_out_only_the_caches(self) -> None:
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
                # The trainer's working caches and the upload area never come home.
                f"{spec['native_dir'][len(run_root):]}/lora.safetensors": False,
                "transfer/upload.tar": False,
            }
            for relative in written:
                path = run / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            self.assertEqual(sorted(_inventory(workspace, run_id)), sorted(path for path, kept in written.items() if kept))


if __name__ == "__main__":
    unittest.main()
