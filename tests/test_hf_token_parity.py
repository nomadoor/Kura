"""Every place that sends the Hugging Face token reads it the same way, whichever declared name holds it."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.environment import USER_VARIABLES
from kura.executors.docker import docker_command
from kura.run_commands.plan import _hf_file_size_probe
from kura.run_commands.runpod_ssh import _runpod_secret_env_payload

HF_NAMES = ("HF_TOKEN", *next(variable.aliases for variable in USER_VARIABLES if variable.name == "HF_TOKEN"))


def _environment_with(name: str) -> dict[str, str]:
    clean = {key: value for key, value in os.environ.items() if key not in HF_NAMES}
    return {**clean, name: "example-value"}


class HfTokenParityTests(unittest.TestCase):
    def test_docker_runpod_and_plan_all_see_the_token_under_any_declared_name(self) -> None:
        for name in HF_NAMES:
            with self.subTest(name=name), patch.dict(os.environ, _environment_with(name), clear=True), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                _, docker_env, _ = docker_command(workspace, workspace / "runs" / "r", {"cwd": "/opt", "argv": ["x"], "env": {}}, "image", [], "r1")
                self.assertEqual(docker_env.get("HF_TOKEN"), "example-value")
                self.assertIn("export HF_TOKEN=example-value", _runpod_secret_env_payload() or "")
                seen = {}

                def head(request, timeout):
                    seen["authorization"] = request.headers.get("Authorization")
                    raise OSError("offline")

                with patch("kura.run_commands.plan.urllib.request.urlopen", side_effect=head):
                    _hf_file_size_probe({"repo_id": "org/model", "filename": "a.safetensors"})
                scheme, _, token = str(seen.get("authorization")).partition(" ")
                self.assertEqual((scheme, token), ("Bearer", "example-value"))


if __name__ == "__main__":
    unittest.main()
