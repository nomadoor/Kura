from __future__ import annotations

import argparse
import io
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura import secrets
from kura.cli import cmd_check_secrets
from kura.secrets import cmd_secrets_set, load_secrets, sources

NAMES = ("RUNPOD_API_KEY", "HF_TOKEN", "KURA_NTFY_TOPIC", "KURA_NTFY_TOKEN", "MY_RUNPOD_KEY", "MY_R2_KEY")


@contextmanager
def _setup(workspace_yaml: str = "schema_version: 2\n"):
    """A workspace, a user secrets file location, and a clean environment."""
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root = base / "ws"
        root.mkdir()
        (root / "workspace.yaml").write_text(workspace_yaml, encoding="utf-8")
        user_file = base / "config" / "kura" / "secrets.env"
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.object(secrets, "user_secrets_path", return_value=user_file), patch.dict(os.environ, {}, clear=False), \
                    patch.dict(secrets._loaded_from, {}, clear=True):
                for name in NAMES:
                    os.environ.pop(name, None)
                secrets._resolved_user_secrets_path.cache_clear()
                try:
                    yield root, user_file
                finally:
                    secrets._resolved_user_secrets_path.cache_clear()
        finally:
            os.chdir(previous)


def _set(name: str, value: str | None = None, *, workspace: bool = False, tty: bool = True) -> tuple[int, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    stdin = io.StringIO(value + "\n" if value is not None and not tty else "")
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch("sys.stdin", stdin), \
            patch.object(stdin, "isatty", return_value=tty), patch("kura.secrets.getpass.getpass", return_value=value or ""):
        code = cmd_secrets_set(argparse.Namespace(name=name, workspace=workspace, stdin=not tty and value is not None))
    return code, stdout.getvalue() + stderr.getvalue()


class LoadingTests(unittest.TestCase):
    def test_environment_then_workspace_then_user_file(self) -> None:
        with _setup() as (root, user_file):
            user_file.parent.mkdir(parents=True)
            user_file.write_text("RUNPOD_API_KEY=from-user\nHF_TOKEN=hf-user\nKURA_NTFY_TOPIC=topic-user\n", encoding="utf-8")
            (root / ".env.local").write_text("HF_TOKEN=hf-workspace\nKURA_NTFY_TOPIC=topic-workspace\n", encoding="utf-8")
            os.environ["KURA_NTFY_TOPIC"] = "topic-env"
            load_secrets()
            self.assertEqual(os.environ["RUNPOD_API_KEY"], "from-user")
            self.assertEqual(os.environ["HF_TOKEN"], "hf-workspace")
            self.assertEqual(os.environ["KURA_NTFY_TOPIC"], "topic-env")
            found = sources()
            self.assertEqual(found["RUNPOD_API_KEY"], "user secrets file")
            self.assertEqual(found["HF_TOKEN"], ".env.local")
            self.assertEqual(found["KURA_NTFY_TOPIC"], "environment")
            self.assertEqual(found["KURA_NTFY_TOKEN"], "unset")

    def test_an_empty_template_line_does_not_hide_the_user_file(self) -> None:
        with _setup() as (root, user_file):
            user_file.parent.mkdir(parents=True)
            user_file.write_text("RUNPOD_API_KEY=from-user\n", encoding="utf-8")
            (root / ".env.local").write_text("# template\nRUNPOD_API_KEY=\nHF_TOKEN=\n", encoding="utf-8")
            load_secrets()
            self.assertEqual(os.environ["RUNPOD_API_KEY"], "from-user")
            self.assertNotIn("HF_TOKEN", os.environ)

    def test_the_workspace_file_is_found_from_a_subdirectory(self) -> None:
        with _setup() as (root, _):
            (root / ".env.local").write_text("export KURA_NTFY_TOPIC='root-topic'\n", encoding="utf-8")
            nested = root / "datasets" / "tiny"
            nested.mkdir(parents=True)
            os.chdir(nested)
            load_secrets()
            self.assertEqual(os.environ["KURA_NTFY_TOPIC"], "root-topic")

    def test_secrets_set_and_file_checks_see_only_the_real_environment(self) -> None:
        from kura import cli

        for argv, loads in ((["kura", "secrets", "set", "HF_TOKEN"], False), (["kura", "doctor", "secrets"], True)):
            with self.subTest(argv=argv), patch("sys.argv", argv), patch.object(cli, "_load_secrets") as load, \
                    patch.object(cli, "cmd_secrets_set", return_value=0), patch.object(cli, "cmd_doctor_secrets", return_value=0), \
                    patch.object(cli, "_refresh_managed_files"):
                with self.assertRaises(SystemExit):
                    cli.main()
                self.assertEqual(load.called, loads)


class SetTests(unittest.TestCase):
    def test_hidden_input_stores_the_value_in_the_user_file_only(self) -> None:
        with _setup() as (root, user_file):
            code, output = _set("RUNPOD_API_KEY", "rk-secret-value")
            self.assertEqual(code, 0, output)
            self.assertIn("RUNPOD_API_KEY=rk-secret-value", user_file.read_text(encoding="utf-8"))
            self.assertNotIn("rk-secret-value", output)
            self.assertFalse((root / ".env.local").exists())
            if os.name != "nt":
                self.assertEqual(user_file.stat().st_mode & 0o777, 0o600)

    def test_setting_again_replaces_only_that_line(self) -> None:
        with _setup() as (_, user_file):
            user_file.parent.mkdir(parents=True)
            user_file.write_text("# mine\nHF_TOKEN=hf-old\nRUNPOD_API_KEY=old\nRUNPOD_API_KEY=older\n", encoding="utf-8")
            code, output = _set("RUNPOD_API_KEY", "new value")
            self.assertEqual(code, 0, output)
            self.assertEqual(user_file.read_text(encoding="utf-8"), "# mine\nHF_TOKEN=hf-old\nRUNPOD_API_KEY='new value'\n")
            load_secrets()
            self.assertEqual(os.environ["RUNPOD_API_KEY"], "new value")

    def test_without_a_terminal_nothing_is_written_and_the_user_is_told_where_to_run_it(self) -> None:
        with _setup() as (_, user_file):
            code, output = _set("RUNPOD_API_KEY", None, tty=False)
            self.assertEqual(code, 1)
            self.assertIn("your own terminal", output)
            self.assertIn("Nothing was written", output)
            self.assertFalse(user_file.exists())

    def test_stdin_mode_reads_a_pipe(self) -> None:
        with _setup() as (_, user_file):
            code, output = _set("HF_TOKEN", "hf-piped", tty=False)
            self.assertEqual(code, 0, output)
            self.assertIn("HF_TOKEN=hf-piped", user_file.read_text(encoding="utf-8"))

    def test_workspace_mode_writes_env_local_and_it_wins(self) -> None:
        with _setup() as (root, user_file):
            _set("RUNPOD_API_KEY", "user-account")
            code, output = _set("RUNPOD_API_KEY", "other-account", workspace=True)
            self.assertEqual(code, 0, output)
            self.assertIn("RUNPOD_API_KEY=other-account", (root / ".env.local").read_text(encoding="utf-8"))
            load_secrets()
            self.assertEqual(os.environ["RUNPOD_API_KEY"], "other-account")

    def test_unknown_names_and_empty_values_are_refused(self) -> None:
        with _setup() as (_, user_file):
            code, output = _set("SOMETHING_ELSE", "x")
            self.assertEqual(code, 1)
            self.assertIn("RUNPOD_API_KEY", output)
            code, output = _set("HF_TOKEN", "   ")
            self.assertEqual(code, 1)
            self.assertFalse(user_file.exists())

    def test_the_configured_runpod_key_name_is_accepted(self) -> None:
        with _setup("schema_version: 2\nrunpod:\n  api_key_env: MY_RUNPOD_KEY\n") as (_, user_file):
            code, output = _set("MY_RUNPOD_KEY", "rk")
            self.assertEqual(code, 0, output)
            self.assertIn("MY_RUNPOD_KEY", sources())

    def test_configured_object_store_key_names_are_accepted(self) -> None:
        with _setup("schema_version: 2\nrunpod:\n  object_store: {access_key_env: MY_R2_KEY}\n"):
            code, output = _set("MY_R2_KEY", "r2")
            self.assertEqual(code, 0, output)

    def test_a_value_spanning_lines_is_refused(self) -> None:
        with _setup() as (_, user_file):
            code, _ = _set("HF_TOKEN", "hf-a\u2028hf-b")
            self.assertEqual(code, 1)
            self.assertFalse(user_file.exists())

    @unittest.skipIf(os.name == "nt", "symlinks need extra privileges on Windows")
    def test_a_symlinked_secrets_file_is_updated_where_it_points(self) -> None:
        with _setup() as (_, user_file):
            synced = user_file.parent.parent / "dotfiles" / "kura.env"
            synced.parent.mkdir(parents=True)
            synced.write_text("HF_TOKEN=old\n", encoding="utf-8")
            user_file.parent.mkdir(parents=True)
            user_file.symlink_to(synced)
            code, output = _set("HF_TOKEN", "new")
            self.assertEqual(code, 0, output)
            self.assertTrue(user_file.is_symlink())
            self.assertEqual(synced.read_text(encoding="utf-8"), "HF_TOKEN=new\n")

    def test_concurrent_writes_keep_every_name(self) -> None:
        import threading

        with _setup() as (_, user_file):
            names = [variable.name for variable in secrets.USER_VARIABLES]
            threads = [threading.Thread(target=secrets.write_secret, args=(user_file, name, f"value-{index}")) for index, name in enumerate(names)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            text = user_file.read_text(encoding="utf-8")
            for index, name in enumerate(names):
                self.assertIn(f"{name}=value-{index}", text)

    def test_a_shell_value_is_named_as_taking_precedence(self) -> None:
        with _setup():
            os.environ["HF_TOKEN"] = "from-shell"
            code, output = _set("HF_TOKEN", "hf-new")
            self.assertEqual(code, 0, output)
            self.assertIn("takes precedence", output)


class CheckTests(unittest.TestCase):
    def test_check_secrets_never_reads_the_user_file(self) -> None:
        with _setup() as (_, user_file):
            _set("HF_TOKEN", "hf_" + "A" * 24)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = cmd_check_secrets(argparse.Namespace(paths=[str(user_file)]))
            self.assertEqual(code, 1)
            self.assertIn("never reads", stderr.getvalue())
            self.assertNotIn("A" * 24, stdout.getvalue() + stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
