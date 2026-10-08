"""Every backend refuses the same secret names in a command's env, by the one rule in kura.secrets."""

from __future__ import annotations

import unittest


def _explicit(backend: str, env: dict[str, str]) -> dict:
    return {"id": "native", "type": "train", "backend": {"name": backend, "config": {"command": {"cwd": "/opt/x", "argv": ["python", "train.py"], "env": env}}}}


class SecretNameParityTests(unittest.TestCase):
    def test_every_backend_refuses_every_secret_name_kura_knows(self) -> None:
        from kura.backends.ai_toolkit import command_ai_toolkit
        from kura.backends.musubi_command import command_musubi_tuner
        from kura.backends.sd_scripts import command_sd_scripts
        from kura.secrets import SECRET_NAME_PARTS

        commands = {"ai-toolkit": command_ai_toolkit, "musubi-tuner": command_musubi_tuner, "sd-scripts": command_sd_scripts}
        for backend, command in commands.items():
            for part in SECRET_NAME_PARTS:
                name = f"MY_{part}_ID"
                with self.subTest(backend=backend, name=name):
                    with self.assertRaisesRegex(ValueError, "must not contain secrets"):
                        command(_explicit(backend, {name: "x"}))

    def test_musubi_generated_env_uses_the_same_rule(self) -> None:
        from kura.backends.musubi_command import _backend_env
        from kura.secrets import SECRET_NAME_PARTS

        for part in SECRET_NAME_PARTS:
            with self.subTest(part=part), self.assertRaisesRegex(ValueError, "must not contain secrets"):
                _backend_env("Musubi Tuner", {"env": {f"MY_{part}_ID": "x"}})

    def test_a_plain_name_passes_on_every_backend(self) -> None:
        from kura.backends.ai_toolkit import command_ai_toolkit
        from kura.backends.musubi_command import command_musubi_tuner
        from kura.backends.sd_scripts import command_sd_scripts

        for backend, command in {"ai-toolkit": command_ai_toolkit, "musubi-tuner": command_musubi_tuner, "sd-scripts": command_sd_scripts}.items():
            with self.subTest(backend=backend):
                self.assertEqual(command(_explicit(backend, {"PYTORCH_CUDA_ALLOC_CONF": "x"}))["env"].get("PYTORCH_CUDA_ALLOC_CONF"), "x")


if __name__ == "__main__":
    unittest.main()
