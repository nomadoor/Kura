"""A Pod gets its secrets one way, for training and for renders: a 0600 file written over SSH stdin."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.run_commands.runpod_ssh import install_pod_secrets

ROOT = Path(__file__).resolve().parents[1]
DETAILS = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
TOKEN = "hf_" + "y" * 20


class PodSecretTests(unittest.TestCase):
    def test_the_values_travel_on_stdin_into_a_file_only_root_reads(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": TOKEN}), \
                patch("kura.run_commands.runpod_ssh._run_bounded", return_value=subprocess.CompletedProcess(["ssh"], 0, "", "")) as run:
            path = install_pod_secrets(DETAILS, "run-1")
        self.assertEqual(path, "/tmp/kura-secrets/run-1.env")
        argv, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertNotIn(TOKEN, " ".join(argv))
        self.assertIn(TOKEN, kwargs["input"])
        self.assertIn("umask 077", argv[-1])
        self.assertIn("chmod 600 /tmp/kura-secrets/run-1.env", argv[-1])

    def test_nothing_is_written_without_a_secret_and_a_failure_hides_values(self) -> None:
        clean = {key: value for key, value in os.environ.items() if key not in ("HF_TOKEN", "KURA_NTFY_TOPIC")}
        with patch.dict(os.environ, clean, clear=True), patch("kura.run_commands.runpod_ssh._run_bounded") as run:
            self.assertEqual(install_pod_secrets(DETAILS, "run-1"), "/tmp/kura-secrets/run-1.env")
        run.assert_not_called()
        failed = subprocess.CompletedProcess(["ssh"], 1, "", f"cannot write {TOKEN}")
        with patch.dict(os.environ, {"HF_TOKEN": TOKEN}), patch("kura.run_commands.runpod_ssh._run_bounded", return_value=failed), \
                self.assertRaises(ValueError) as caught:
            install_pod_secrets(DETAILS, "run-1")
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertIn("exit code 1: cannot write ***", str(caught.exception))

    def test_training_and_render_both_install_through_the_one_function(self) -> None:
        sources = [path for path in (ROOT / "src" / "kura").rglob("*.py") if "mkdir -p /tmp/kura-secrets" in path.read_text(encoding="utf-8")]
        self.assertEqual(sources, [ROOT / "src" / "kura" / "run_commands" / "runpod_ssh.py"])
        self.assertIn("install_pod_secrets(details, run_id)", (ROOT / "src" / "kura" / "run_commands" / "render_runpod.py").read_text(encoding="utf-8"))
        self.assertEqual((ROOT / "src" / "kura" / "run_commands" / "runpod_ssh.py").read_text(encoding="utf-8").count("mkdir -p /tmp/kura-secrets"), 1)


if __name__ == "__main__":
    unittest.main()
