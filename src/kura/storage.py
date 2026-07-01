"""Storage backing probes for local safety checks.

Kura's disk checks must account for the physical backing store, not only the
logical filesystem reported by ``statvfs``. This mainly matters on WSL2, where a
Linux ext4 filesystem may report the virtual disk's maximum free space while the
actual Windows drive that grows the VHDX has much less room.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageStatus:
    path: str
    probe: str
    backing_id: str
    backing_kind: str
    linux_free_bytes: int
    linux_total_bytes: int
    host_free_bytes: int | None
    effective_free_bytes: int
    confidence: str
    mount: dict[str, Any]
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _existing_probe(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _findmnt_for(path: Path) -> dict[str, Any]:
    findmnt = shutil.which("findmnt")
    if not findmnt:
        return {"available": False, "reason": "findmnt not found"}
    try:
        result = subprocess.run(
            [findmnt, "-T", str(path), "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}
    if result.returncode != 0:
        return {"available": False, "reason": result.stderr.strip() or result.stdout.strip()}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"available": False, "reason": str(exc)}
    filesystems = payload.get("filesystems") if isinstance(payload, dict) else None
    if not isinstance(filesystems, list) or not filesystems or not isinstance(filesystems[0], dict):
        return {"available": False, "reason": "findmnt returned no filesystem"}
    item = filesystems[0]
    return {
        "available": True,
        "target": item.get("target"),
        "source": item.get("source"),
        "fstype": item.get("fstype"),
        "options": item.get("options"),
    }


def _config_drive(config: dict[str, Any] | None, key: str) -> str | None:
    storage = config.get("storage") if isinstance(config, dict) else None
    if not isinstance(storage, dict):
        return None
    value = storage.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    drive = value.strip().upper()
    if len(drive) == 1 and drive.isalpha():
        drive = f"{drive}:"
    return drive


def _normalize_drive(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("\\\\?\\"):
        text = text[4:]
    if len(text) >= 2 and text[0].isalpha() and text[1] == ":":
        return f"{text[0].upper()}:"
    if len(text) == 1 and text.isalpha():
        return f"{text.upper()}:"
    return None


def _drive_from_mnt_path(path: Path) -> str | None:
    parts = path.resolve(strict=False).parts
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
        return f"{parts[2].upper()}:"
    return None


def _windows_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    candidates = {
        "powershell.exe": [
            Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"),
            Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.EXE"),
        ],
        "cmd.exe": [
            Path("/mnt/c/Windows/System32/cmd.exe"),
            Path("/mnt/c/Windows/System32/cmd.EXE"),
        ],
    }.get(name.lower(), [])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@lru_cache(maxsize=16)
def _powershell_json(script: str) -> Any:
    powershell = _windows_executable("powershell.exe")
    if not powershell:
        return None
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", script],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


@lru_cache(maxsize=1)
def _auto_wsl_host_drive() -> str | None:
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return None
    script = (
        "$items = Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | "
        "ForEach-Object { Get-ItemProperty $_.PSPath | Select-Object DistributionName,BasePath }; "
        "$items | ConvertTo-Json -Compress"
    )
    payload = _powershell_json(script)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("DistributionName") == distro:
            return _normalize_drive(item.get("BasePath"))
    return None


@lru_cache(maxsize=16)
def _windows_drive_free_bytes(drive: str) -> int | None:
    powershell = _windows_executable("powershell.exe")
    if not powershell:
        return None
    safe_drive = drive.upper()
    if len(safe_drive) == 1 and safe_drive.isalpha():
        safe_drive = f"{safe_drive}:"
    if not (len(safe_drive) == 2 and safe_drive[0].isalpha() and safe_drive[1] == ":"):
        return None
    script = (
        f"$d = Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='{safe_drive}'\"; "
        "if ($null -eq $d) { exit 2 }; "
        "[Console]::Out.Write($d.FreeSpace)"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", script],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def probe_storage(path: Path, config: dict[str, Any] | None = None, *, role: str | None = None) -> StorageStatus:
    probe = _existing_probe(path)
    usage = shutil.disk_usage(probe)
    mount = _findmnt_for(probe)
    linux_free = usage.free
    linux_total = usage.total
    host_free: int | None = None
    effective_free = linux_free
    confidence = "exact"
    backing_kind = "native"
    backing_id = str(mount.get("source") or probe)
    warning: str | None = None

    if is_wsl():
        fstype = str(mount.get("fstype") or "").lower()
        drive = _drive_from_mnt_path(probe)
        if drive:
            backing_kind = "wsl2_windows_drive"
            backing_id = drive
            confidence = "exact"
        elif fstype == "ext4":
            backing_kind = "wsl2_vhdx"
            drive = _config_drive(config, "host_drive") or _auto_wsl_host_drive()
            backing_id = drive or str(mount.get("source") or "wsl2-vhdx")
            host_free = _windows_drive_free_bytes(drive) if drive else None
            if host_free is None:
                confidence = "unknown"
                warning = (
                    f"{role or path} is on WSL Linux ext4; Linux free space may not reflect "
                    "the Windows drive that stores the WSL virtual disk"
                )
            else:
                confidence = "estimated"
                effective_free = min(linux_free, host_free)
        else:
            backing_kind = "wsl2"
            confidence = "unknown"
            warning = (
                f"{role or path} is on WSL, but Kura could not identify the physical backing store; "
                "Linux free space may not reflect the host drive that stores Docker/WSL data"
            )

    return StorageStatus(
        path=str(path),
        probe=str(probe),
        backing_id=backing_id,
        backing_kind=backing_kind,
        linux_free_bytes=linux_free,
        linux_total_bytes=linux_total,
        host_free_bytes=host_free,
        effective_free_bytes=effective_free,
        confidence=confidence,
        mount=mount,
        warning=warning,
    )


def probe_storages(paths: dict[str, Path], config: dict[str, Any] | None = None) -> dict[str, StorageStatus]:
    return {name: probe_storage(path, config, role=name) for name, path in paths.items()}
