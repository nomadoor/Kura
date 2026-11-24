"""Every backend refuses the same secret names in a command's env, by the one rule in kura.secrets."""

from __future__ import annotations

import unittest

# Built at runtime so this file itself does not trip the repository secret scan.
VALUE = "Kq7" + "zz81" * 4

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

    def test_every_command_env_is_checked_by_one_owner(self) -> None:
        from unittest.mock import patch

        from kura.backends import shared
        from kura.backends.ai_toolkit import command_ai_toolkit
        from kura.backends.musubi_command import _backend_env, command_musubi_tuner
        from kura.backends.sd_scripts import _validate_native_env, command_sd_scripts

        modules = {"ai-toolkit": "kura.backends.ai_toolkit", "musubi-tuner": "kura.backends.musubi_command", "sd-scripts": "kura.backends.sd_scripts",
                   "musubi generated": "kura.backends.musubi_command", "sd-scripts generated": "kura.backends.sd_scripts"}
        calls = {
            "ai-toolkit": lambda: command_ai_toolkit(_explicit("ai-toolkit", {"A": "1"})),
            "musubi-tuner": lambda: command_musubi_tuner(_explicit("musubi-tuner", {"A": "1"})),
            "sd-scripts": lambda: command_sd_scripts(_explicit("sd-scripts", {"A": "1"})),
            "musubi generated": lambda: _backend_env("Musubi Tuner", {"env": {"A": "1"}}),
            "sd-scripts generated": lambda: _validate_native_env({"env": {"A": "1"}}),
        }
        for name, call in calls.items():
            with self.subTest(command=name), patch(f"{modules[name]}.validated_command_env", wraps=shared.validated_command_env) as owner:
                call()
            owner.assert_called_once()
        with self.assertRaises(ValueError) as caught:
            shared.validated_command_env({"Z_TOKEN": "value-z", "A_SECRET": "value-a", "PLAIN": "x"}, label="X")
        self.assertTrue(str(caught.exception).startswith("X env must not contain secrets (A_SECRET, Z_TOKEN): "))
        self.assertNotIn("value-", str(caught.exception))
        self.assertNotIn("PLAIN", str(caught.exception))
        for env in ({"A": 1}, ["A"], None):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "env must be a mapping of names to string values"):
                shared.validated_command_env(env, label="X")

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


def _with_native_config(native_config: dict) -> dict:
    return {"id": "native", "type": "train", "model": {"base": "owner/model"}, "backend": {"name": "ai-toolkit", "config": {"model_arch": "sd1", "native_config": native_config}}}


def _with_extra_args(backend: str, extra_args: list[str]) -> dict:
    config = {"musubi-tuner": {"architecture": "wan"}, "sd-scripts": {"architecture": "sdxl", "mode": "lora"}}[backend]
    return {"id": "native", "type": "train", "backend": {"name": backend, "config": {**config, "extra_args": extra_args}}}


class SecretFlagTests(unittest.TestCase):
    def assert_no_secret_found(self, run: dict) -> None:
        """These short runs lack values a compile requires; what matters is that the refusal,
        when there is one, is not the secret one, which comes before those."""
        from kura.backends.registry import validate_backend_config

        try:
            validate_backend_config(run)
        except ValueError as refusal:
            self.assertNotIn("secret", str(refusal))

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
        # The value inside the item is what is found, not the argument after it: last, or before a flag.
        self.assertEqual(secret_flags(["--wandb_api_key 0123456789abcdef"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["--wandb_api_key 0123456789abcdef", "--seed"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["env", "WANDB_API_KEY=0123456789abcdef", "python"]), ["WANDB_API_KEY"])
        self.assertEqual(secret_flags(["env", "TOKENIZERS_PARALLELISM=false", "MAX_TOKEN_LENGTH=225"]), [])

    def test_a_secret_inside_a_shell_string_or_after_a_single_dash_flag_is_found(self) -> None:
        from kura.secrets import secret_flags

        self.assertEqual(secret_flags(["bash", "-c", f"python train.py --seed 1 --wandb_api_key {VALUE}"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["bash", "-c", f"WANDB_API_KEY='{VALUE}' python train.py"]), ["WANDB_API_KEY"])
        # Shell quoting around the value does not hide it, in a shell string or in one item.
        self.assertEqual(secret_flags(["sh", "-c", f'python train.py --wandb_api_key="{VALUE}"']), ["--wandb_api_key"])
        self.assertEqual(secret_flags([f"--wandb_api_key '{VALUE}'"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["--wandb_api_key", VALUE, "--wandb_api_key", VALUE]), ["--wandb_api_key"])
        # A string the shell rules cannot read (an unclosed quote) is split at whitespace.
        self.assertEqual(secret_flags(["bash", "-c", f"echo it's --wandb_api_key {VALUE}"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["-wandb_api_key", VALUE]), ["-wandb_api_key"])
        self.assertEqual(secret_flags([f"-api_key={VALUE}"]), ["-api_key"])
        # A value that begins with a dash is still a value; a `--` flag after a flag never is.
        self.assertEqual(secret_flags(["--wandb_api_key", "-" + VALUE]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["-use_auth_token", "--other_long_flag", "--no_token_padding", "--another_long_flag"]), [])
        self.assertEqual(secret_flags(["bash", "-c", "python train.py --max_token_length 225 -x 12345678", "", "-1"]), [])

    def test_an_argv_item_is_one_value_with_its_spaces(self) -> None:
        from kura.secrets import secret_flags

        spaced = "Bearer " + VALUE[:6]
        # Each word alone is too short to be a secret; the item is the value.
        self.assertEqual(secret_flags(["--auth_token", spaced]), ["--auth_token"])
        self.assertEqual(secret_flags([f"--auth_token={spaced}"]), ["--auth_token"])
        self.assertEqual(secret_flags([f"--auth_token {spaced}"]), ["--auth_token"])
        self.assertEqual(secret_flags(["env", f"AUTH_TOKEN={spaced}", "python"]), ["AUTH_TOKEN"])
        self.assertEqual(secret_flags(["bash", "-c", f"python train.py --auth_token '{spaced}'"]), ["--auth_token"])
        # Only the string a shell is given (`-c`) is read as shell text: a value with spaces is not
        # searched for flags.
        self.assertEqual(secret_flags(["--prompt", "a -token-based illustration"]), [])
        self.assertEqual(secret_flags(["bash", "-c", "python gen.py --prompt 'a -token-based illustration'"]), [])

    def test_a_shell_string_is_read_at_any_depth_and_across_separators(self) -> None:
        from kura.secrets import secret_flags

        self.assertEqual(secret_flags(["bash", "-c", f"sh -c 'python train.py --wandb_api_key {VALUE}'"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["bash", "-lc", f"cd /workspace;WANDB_API_KEY={VALUE} python train.py"]), ["WANDB_API_KEY"])
        self.assertEqual(secret_flags(["bash", "-c", f"cd /workspace&&python train.py --wandb_api_key {VALUE}|tee log"]), ["--wandb_api_key"])
        self.assertEqual(secret_flags(["bash", "-c", "cd /workspace;TOKENIZERS_PARALLELISM=false python train.py --max_token_length 225"]), [])
        # A shell keeps `#` inside a word, so the flag after it is still read.
        self.assertEqual(secret_flags(["bash", "-c", f"python train.py --run_name trial#1 --hf_token {VALUE}"]), ["--hf_token"])
        # The command string is the first argument after the shell's options, not the one right after `-c`.
        self.assertEqual(secret_flags(["bash", "-c", "-e", f"python train.py --hf_token {VALUE}"]), ["--hf_token"])
        self.assertEqual(secret_flags(["/bin/sh", "-e", "-o", "pipefail", "-c", f"python train.py --hf_token {VALUE}"]), ["--hf_token"])
        # Another program's `-c` takes code, not a shell string.
        self.assertEqual(secret_flags(["python", "-c", "print('--max_token_length 2048 --resolution 1024')"]), [])

    def test_a_secret_in_a_shell_string_or_after_a_single_dash_flag_is_refused_on_every_backend(self) -> None:
        from kura.backends.registry import validate_backend_config

        forms = (["bash", "-c", f"python train.py --wandb_api_key {VALUE}"], ["python", "train.py", f'--wandb_api_key="{VALUE}"'], ["python", "train.py", "-wandb_api_key", VALUE])
        for backend in ("ai-toolkit", "musubi-tuner", "sd-scripts"):
            for argv in forms:
                run = _explicit(backend, {})
                run["backend"]["config"]["command"]["argv"] = argv
                with self.subTest(backend=backend, argv=argv[:2]), self.assertRaisesRegex(ValueError, "command.argv must not contain secrets") as caught:
                    validate_backend_config(run)
                self.assertNotIn(VALUE, str(caught.exception))
        for backend in ("musubi-tuner", "sd-scripts"):
            for arguments in ([f"--wandb_api_key '{VALUE}'"], ["-wandb_api_key", VALUE], [f'-wandb_api_key="{VALUE}"']):
                with self.subTest(backend=backend, arguments=arguments[:1]), self.assertRaisesRegex(ValueError, "extra_args must not contain secrets") as caught:
                    validate_backend_config(_with_extra_args(backend, arguments))
                self.assertNotIn(VALUE, str(caught.exception))

    def test_a_secret_under_a_key_of_backend_config_is_refused_by_its_path(self) -> None:
        from kura.backends.registry import validate_backend_config
        from kura.secrets import secret_keys

        self.assertEqual(
            secret_keys({"a": {"wandb_api_key": VALUE, "list": [{"x": 1}, [{"hf_token": VALUE}]]}, "password": VALUE, "max_token_length": 225, "access_token": "short"}),
            ["a.wandb_api_key", "a.list[1][0].hf_token", "password"],
        )
        for native_config in (
            {"logging": {"wandb_api_key": VALUE}},
            {"datasets": [{"folder_path": "/x"}, {"hf_token": VALUE}]},
            {"model": {"model_kwargs": {"nested": [[{"apiKey": VALUE}]]}}},
        ):
            run = _with_native_config(native_config)
            with self.subTest(native_config=sorted(native_config)), self.assertRaisesRegex(ValueError, r"ai-toolkit backend\.config\.native_config must not contain secrets \((logging|datasets|model)[.\[]") as caught:
                validate_backend_config(run)
            self.assertNotIn(VALUE, str(caught.exception))
        # Only native_config is walked: every other key of backend.config is declared, and a
        # command's env is refused by name.
        for backend in ("ai-toolkit", "musubi-tuner", "sd-scripts"):
            with self.subTest(backend=backend):
                run = _explicit(backend, {})
                run["backend"]["config"]["command"]["eos_token"] = "<|endoftext|>"
                self.assert_no_secret_found(run)

    def test_an_ordinary_setting_is_still_accepted_on_every_backend(self) -> None:
        from kura.secrets import secret_keys

        ordinary = {"train": {"max_token_length": 225, "tokenizer_path": "/workspace/models/tokenizer"}, "datasets": [{"keep_tokens_separator": "|||||||||", "token_dropout": "0.1"}]}
        self.assertEqual(secret_keys(ordinary), [])
        self.assert_no_secret_found(_with_native_config(ordinary))
        for backend in ("musubi-tuner", "sd-scripts"):
            with self.subTest(backend=backend):
                self.assert_no_secret_found(_with_extra_args(backend, ["--max_token_length", "225", "--keep_tokens_separator", "|||||||||", "--tokenizer_cache_dir=/workspace/cache/tokenizers", "-1"]))
        for backend in ("ai-toolkit", "musubi-tuner", "sd-scripts"):
            run = _explicit(backend, {"TOKENIZERS_PARALLELISM": "false"})
            run["backend"]["config"]["command"]["argv"] = ["bash", "-c", "python train.py --max_token_length 225 --output_dir '/workspace/out put'"]
            with self.subTest(backend=backend):
                self.assert_no_secret_found(run)

    def test_every_value_check_asks_holds_secret(self) -> None:
        from unittest.mock import patch

        from kura import secrets
        from kura.backends.registry import validate_backend_config

        runs = {
            "ai-toolkit native_config": _with_native_config({"logging": [{"wandb_api_key": VALUE}]}),
            "musubi-tuner extra_args": _with_extra_args("musubi-tuner", ["-wandb_api_key", VALUE]),
            "sd-scripts extra_args": _with_extra_args("sd-scripts", [f"--wandb_api_key '{VALUE}'"]),
        }
        for backend in ("ai-toolkit", "musubi-tuner", "sd-scripts"):
            runs[f"{backend} command.argv"] = _explicit(backend, {})
            runs[f"{backend} command.argv"]["backend"]["config"]["command"]["argv"] = ["bash", "-c", f"python train.py --wandb_api_key {VALUE}"]
        for name, run in runs.items():
            with self.subTest(run=name), patch("kura.secrets.holds_secret", wraps=secrets.holds_secret) as owner:
                with self.assertRaisesRegex(ValueError, "must not contain secrets"):
                    validate_backend_config(run)
            self.assertTrue(any(VALUE in call.args[1] for call in owner.call_args_list))
            # With the owner answering no, nothing else finds the value.
            with self.subTest(run=name, owner="no"), patch("kura.secrets.holds_secret", return_value=False):
                self.assert_no_secret_found(run)

    def test_a_frozen_manifest_is_read_as_it_is(self) -> None:
        from kura.backends.registry import validate_backend_config

        validate_backend_config(_with_extra_args("sd-scripts", ["--wandb_api_key", "0123456789abcdef"]), frozen=True)
        validate_backend_config(_with_native_config({"logging": {"wandb_api_key": VALUE}}), frozen=True)

if __name__ == "__main__":
    unittest.main()
