"""Opening a folder or URL on the user's desktop: one rule per OS, for the monitor and `kura open`."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import yaml

from kura import opener


@contextmanager
def _os(name: str):
    """Pretend to run on `name`: wsl, windows, darwin, or linux."""
    with patch.object(opener, "is_wsl", return_value=name == "wsl"), \
            patch.object(opener.platform, "system", return_value={"windows": "Windows", "darwin": "Darwin"}.get(name, "Linux")), \
            patch.object(opener.shutil, "which", side_effect=lambda command: command), \
            patch.object(opener.subprocess, "Popen") as popen:
        yield popen


class OpenerTests(unittest.TestCase):
    def test_each_os_opens_a_folder_its_own_way(self) -> None:
        folder = Path(tempfile.gettempdir())
        with _os("darwin") as popen:
            self.assertTrue(opener.open_path(folder))
        self.assertEqual(popen.call_args.args[0], ["open", str(folder)])
        with _os("linux") as popen:
            self.assertTrue(opener.open_path(folder))
        self.assertEqual(popen.call_args.args[0], ["xdg-open", str(folder)])
        with _os("wsl") as popen, patch.object(opener.subprocess, "run") as run:
            run.return_value.returncode, run.return_value.stdout = 0, "\\\\wsl.localhost\\Ubuntu\\tmp\n"
            self.assertTrue(opener.open_path(folder))
        self.assertEqual(popen.call_args.args[0], ["explorer.exe", "\\\\wsl.localhost\\Ubuntu\\tmp"])

    def test_native_windows_opens_with_the_shell(self) -> None:
        folder = Path(tempfile.gettempdir())
        with _os("windows") as popen, patch.object(opener.os, "startfile", create=True) as startfile:
            self.assertTrue(opener.open_path(folder))
            self.assertTrue(opener.open_url("https://console.runpod.io/pods?id=pod-1"))
        self.assertEqual([call.args[0] for call in startfile.call_args_list], [str(folder), "https://console.runpod.io/pods?id=pod-1"])
        popen.assert_not_called()

    def test_a_url_on_wsl_goes_to_the_windows_browser(self) -> None:
        with _os("wsl") as popen:
            self.assertTrue(opener.open_url("https://console.runpod.io/pods?id=pod-1"))
        self.assertEqual(popen.call_args.args[0][:4], ["cmd.exe", "/c", "start", ""])

    def test_wsl_without_a_windows_bridge_opens_nothing(self) -> None:
        with _os("wsl") as popen, patch.object(opener, "_windows_command", return_value=None), patch.object(opener.subprocess, "run") as run:
            run.return_value.returncode, run.return_value.stdout = 0, "\\\\wsl.localhost\\Ubuntu\\tmp\n"
            self.assertFalse(opener.open_path(Path(tempfile.gettempdir())))
        popen.assert_not_called()

    def test_a_missing_path_opens_nothing(self) -> None:
        with _os("linux") as popen:
            self.assertFalse(opener.open_path(Path(tempfile.gettempdir()) / "no-such-folder-kura"))
        popen.assert_not_called()

    def test_the_monitor_opens_through_the_same_rule(self) -> None:
        from kura import tui

        self.assertIs(tui.open_path, opener.open_path)
        self.assertIs(tui.open_url, opener.open_url)


@contextmanager
def _workspace():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(previous)


def _run(root: Path, run_id: str, run: dict, *, manifest: dict | None = None) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(yaml.safe_dump({"id": run_id, **run}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    if manifest is not None:
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return run_dir


def _kura_open(target: str) -> tuple[int, list[Path], str]:
    from kura.cli import cmd_open

    opened: list[Path] = []
    stderr = io.StringIO()
    with patch("kura.cli.open_path", side_effect=lambda path: opened.append(path) or True), redirect_stderr(stderr), patch("sys.stdout", io.StringIO()):
        code = cmd_open(argparse.Namespace(target=target))
    return code, opened, stderr.getvalue()


class KuraOpenTests(unittest.TestCase):
    def test_a_training_run_opens_its_comparisons_or_else_its_outputs(self) -> None:
        with _workspace() as root:
            train = _run(root, "t1", {"type": "train"})
            (train / "outputs").mkdir()
            (train / "outputs" / "lora.safetensors").write_bytes(b"x")
            self.assertEqual(_kura_open("t1")[:2], (0, [train / "outputs"]))
            (train / "comparisons").mkdir()
            self.assertEqual(_kura_open("t1")[:2], (0, [train / "comparisons"]))

    def test_a_render_run_opens_the_folder_it_wrote_images_to(self) -> None:
        with _workspace() as root:
            render = _run(root, "r1", {"type": "render"}, manifest={"type": "render", "render": {"output_dir": "samples/grid"}})
            (render / "samples" / "grid").mkdir(parents=True)
            (render / "samples" / "grid" / "a.png").write_bytes(b"x")
            self.assertEqual(_kura_open("r1")[:2], (0, [render / "samples" / "grid"]))

    def test_a_path_is_opened_as_given(self) -> None:
        with _workspace() as root:
            self.assertEqual(_kura_open(str(root))[:2], (0, [root]))

    def test_what_cannot_be_opened_says_why_and_where(self) -> None:
        from kura.cli import cmd_open

        with _workspace() as root:
            code, opened, text = _kura_open("no-such-run")
            self.assertEqual((code, opened), (1, []))
            self.assertIn("no-such-run", text)
            _run(root, "t1", {"type": "train"})
            stderr = io.StringIO()
            with patch("kura.cli.open_path", return_value=False), redirect_stderr(stderr), patch("sys.stdout", io.StringIO()):
                code = cmd_open(argparse.Namespace(target="t1"))
            self.assertEqual(code, 1)
            self.assertIn(str(root / "runs" / "t1"), stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
