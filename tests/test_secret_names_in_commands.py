"""Every backend refuses the same secret names in a command's env, by the one rule in kura.secrets."""

from __future__ import annotations

import unittest

SECRET_NAMES = ("HF_TOKEN", "MY_SECRET", "DB_PASSWORD", "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "SSH_PRIVATE_KEY", "apiKey")


def _explicit(backend: str, env: dict[str, str]) -> dict:
    return {"id": "native", "type": "train", "backend": {"name": backend, "config": {"command": {"cwd": "/opt/x", "argv": ["python", "train.py"], "env": env}}}}


class SecretNameParityTests(unittest.TestCase):
    def test_every_backend_refuses_every_secret_name_kura_knows(self) -> None:
        from kura.backends.ai_toolkit import command_ai_toolkit
        from kura.backends.musubi_command import command_musubi_tuner
        from kura.backends.sd_scripts import command_sd_scripts
        commands = {"ai-toolkit": command_ai_toolkit, "musubi-tuner": command_musubi_tuner, "sd-scripts": command_sd_scripts}
        for backend, command in commands.items():
            for name in SECRET_NAMES:
                with self.subTest(backend=backend, name=name):
                    with self.assertRaisesRegex(ValueError, "must not contain secrets"):
                        command(_explicit(backend, {name: "x"}))

    def test_musubi_generated_env_uses_the_same_rule(self) -> None:
        from kura.backends.musubi_command import _backend_env

        for name in SECRET_NAMES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "must not contain secrets"):
                _backend_env("Musubi Tuner", {"env": {name: "x"}})

    def test_a_plain_name_passes_on_every_backend(self) -> None:
        from kura.backends.ai_toolkit import command_ai_toolkit
        from kura.backends.musubi_command import command_musubi_tuner
        from kura.backends.sd_scripts import command_sd_scripts

        for backend, command in {"ai-toolkit": command_ai_toolkit, "musubi-tuner": command_musubi_tuner, "sd-scripts": command_sd_scripts}.items():
            with self.subTest(backend=backend):
                self.assertEqual(command(_explicit(backend, {"TOKENIZERS_PARALLELISM": "false"}))["env"].get("TOKENIZERS_PARALLELISM"), "false")


if __name__ == "__main__":
    unittest.main()
