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


    def test_every_refusal_says_only_the_declared_secrets_reach_a_trainer(self) -> None:
        # No executor forwards the process environment, so the message never suggests it.
        from kura.backends.ai_toolkit import command_ai_toolkit
        from kura.backends.musubi_command import _backend_env, command_musubi_tuner
        from kura.backends.sd_scripts import _validate_native_env, command_sd_scripts
        from kura.secrets import undeclared_secret_refusal

        refusals = []
        for call in (
            lambda: command_ai_toolkit(_explicit("ai-toolkit", {"WANDB_API_KEY": "x"})),
            lambda: command_musubi_tuner(_explicit("musubi-tuner", {"WANDB_API_KEY": "x"})),
            lambda: command_sd_scripts(_explicit("sd-scripts", {"WANDB_API_KEY": "x"})),
            lambda: _backend_env("Musubi Tuner", {"env": {"WANDB_API_KEY": "x"}}),
            lambda: _validate_native_env({"env": {"WANDB_API_KEY": "x"}}),
        ):
            with self.assertRaises(ValueError) as caught:
                call()
            refusals.append(str(caught.exception))
        for refusal in refusals:
            self.assertNotIn("process environment", refusal)
            self.assertIn(undeclared_secret_refusal("x", ["WANDB_API_KEY"]).split(": ", 1)[1], refusal)


def _with_extra_args(backend: str, extra_args: list[str]) -> dict:
    config = {"musubi-tuner": {"architecture": "wan"}, "sd-scripts": {"architecture": "sdxl", "mode": "lora"}}[backend]
    return {"id": "native", "type": "train", "backend": {"name": backend, "config": {**config, "extra_args": extra_args}}}


class SecretFlagTests(unittest.TestCase):
    def test_a_secret_flag_with_a_secret_value_is_refused_in_extra_args_on_every_backend(self) -> None:
        from kura.backends.registry import validate_backend_config

        for backend in ("musubi-tuner", "sd-scripts"):
            for arguments in (["--wandb_api_key", "0123456789abcdef"], ["--wandb_api_key=0123456789abcdef"], ["--huggingface_token", "hf_0123456789"]):
                with self.subTest(backend=backend, arguments=arguments), self.assertRaisesRegex(ValueError, "extra_args must not contain secrets"):
                    validate_backend_config(_with_extra_args(backend, arguments))

    def test_a_secret_value_in_an_explicit_command_argv_is_refused(self) -> None:
        from kura.backends.registry import validate_backend_config

        for backend in ("ai-toolkit", "musubi-tuner", "sd-scripts"):
            run = _explicit(backend, {})
            run["backend"]["config"]["command"]["argv"].extend(["--wandb_api_key", "0123456789abcdef"])
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "command.argv must not contain secrets"):
                validate_backend_config(run)

    def test_a_short_value_or_a_bare_flag_is_a_setting_not_a_secret(self) -> None:
        from kura.secrets import secret_flags

        # sd-scripts' --max_token_length 225 and --token_warmup_step 100 hold the word TOKEN.
        self.assertEqual(secret_flags(["--max_token_length", "225", "--token_warmup_step=100", "--use_secret_sauce", "--seed", "123456789"]), [])
        self.assertEqual(secret_flags(["--wandb_api_key", "0123456789abcdef", "--x"]), ["--wandb_api_key"])
        # One item holding flag and value, or an `env NAME=value` assignment, is found; the value
        # never appears in what is reported.
        self.assertEqual(secret_flags(["--wandb_api_key 0123456789abcdef", "/some/long/path"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["--wandb_api_key\t0123456789abcdef", "/some/long/path"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["env", "WANDB_API_KEY=0123456789abcdef", "python"]), ["WANDB_API_KEY"])
        self.assertEqual(secret_flags(["env", "TOKENIZERS_PARALLELISM=false", "MAX_TOKEN_LENGTH=225"]), [])

    def test_a_frozen_manifest_is_read_as_it_is(self) -> None:
        from kura.backends.registry import validate_backend_config

        validate_backend_config(_with_extra_args("sd-scripts", ["--wandb_api_key", "0123456789abcdef"]), frozen=True)

if __name__ == "__main__":
    unittest.main()
