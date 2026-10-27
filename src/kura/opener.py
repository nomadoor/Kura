"""Opening a folder, file, or URL on the user's desktop, for the monitor and `kura open`.

One rule per OS: on WSL through Windows (`explorer.exe` or `cmd /c start`, a path converted with
`wslpath -w`), on Windows with the shell (`os.startfile`), on macOS with `open`, elsewhere with
`xdg-open`. Opening shows the user the file; nothing reaches an agent.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from kura.storage import is_wsl


def _windows_command(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    candidates = {
        "explorer.exe": ("/mnt/c/Windows/explorer.exe",),
        "cmd.exe": ("/mnt/c/Windows/System32/cmd.exe",),
    }.get(name, ())
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _open(target: str, *, url: bool) -> bool:
    """Hand `target` to the desktop; whether a launcher was found and started."""
    if is_wsl():
        # A folder opens in Explorer; a URL in the default browser, which `start` reaches.
        order = ("cmd.exe", "explorer.exe") if url else ("explorer.exe", "cmd.exe")
        command = None
        for name in order:
            found = _windows_command(name)
            if found:
                command = [found, "/c", "start", "", target] if name == "cmd.exe" else [found, target]
                break
        if command is None:
            return False
    elif platform.system() == "Windows":
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except OSError:
            return False
        return True
    elif platform.system() == "Darwin":
        command = ["open", target]
    else:
        if not shutil.which("xdg-open"):
            return False
        command = ["xdg-open", target]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return True


def open_path(path: Path) -> bool:
    """Open a folder or file that exists; False when it is missing or nothing could open it."""
    if not path.exists():
        return False
    target = str(path)
    if is_wsl():
        converted = subprocess.run(["wslpath", "-w", target], text=True, capture_output=True, check=False)
        target = converted.stdout.strip()
        if converted.returncode or not target:
            return False
    return _open(target, url=False)


def open_url(url: str) -> bool:
    """Open a URL in the user's browser; False when nothing could open it."""
    return _open(url, url=True)
