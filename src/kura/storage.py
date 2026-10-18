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


def _mount_point(path: Path) -> Path:
    # Without findmnt (macOS, native Windows), the enclosing mount point names the backing store,
    # so paths on one filesystem are reported together.
    point = path.resolve()
    while not os.path.ismount(point) and point != point.parent:
        point = point.parent
    return point


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
    return value if isinstance(value, str) else None


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


# Answers PowerShell gave. A failure (a timeout, say) is not kept, so the next check asks again.
_powershell_cache: dict[str, Any] = {}


def _powershell_json(script: str) -> Any:
    if script in _powershell_cache:
        return _powershell_cache[script]
    value = _powershell_json_uncached(script)
    if value is not None:
        _powershell_cache[script] = value
    return value


def _powershell_json_uncached(script: str) -> Any:
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


# The `drive_memo` key of the drive detected behind the WSL virtual disk; a drive letter never looks like this.
_AUTO_DRIVE = "auto-detected"


def probe_storage(
    path: Path, config: dict[str, Any] | None = None, *, role: str | None = None, drive_memo: dict[str, Any] | None = None,
) -> StorageStatus:
    """The effective free space behind `path`. `drive_memo` keeps, for one round of probes, the
    detected drive behind the WSL virtual disk (under `_AUTO_DRIVE`, a failed lookup included) and
    each Windows drive's free space, so stores on one drive start PowerShell once per round."""
    probe = _existing_probe(path)
    usage = shutil.disk_usage(probe)
    mount = _findmnt_for(probe)
    linux_free = usage.free
    linux_total = usage.total
    host_free: int | None = None
    effective_free = linux_free
    confidence = "exact"
    backing_kind = "native"
    backing_id = str(mount.get("source") or _mount_point(probe))
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
            drive = _config_drive(config, "host_drive")
            if drive is None:
                if drive_memo is not None and _AUTO_DRIVE in drive_memo:
                    drive = drive_memo[_AUTO_DRIVE]
                else:
                    drive = _auto_wsl_host_drive()
                    if drive_memo is not None:
                        drive_memo[_AUTO_DRIVE] = drive
            backing_id = drive or str(mount.get("source") or "wsl2-vhdx")
            if drive and drive_memo is not None and drive in drive_memo:
                host_free = drive_memo[drive]
            else:
                host_free = _windows_drive_free_bytes(drive) if drive else None
                if drive and drive_memo is not None:
                    drive_memo[drive] = host_free
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


def storage_floor_problem(
    status: StorageStatus, path: Path | str, required_bytes: int, *, action: str, minimum: str | None = None,
    risk_setting: str | None = None, risk_accepted: bool = False,
) -> str | None:
    """Whether a store holds `required_bytes`, for every local disk gate (a launch, an image build,
    a RunPod stage or pull, and `kura doctor disk`).

    On WSL2 a store whose Windows backing drive is unknown cannot be measured, so it is refused
    unless the caller's risk setting accepts it. When the WSL virtual disk is full while the
    Windows drive behind it has room, the problem says so: Linux seeing less free space than the
    drive is normal on WSL, so this is said only when it is what stops the gate."""
    gib = 1024**3
    required_display_gib = (required_bytes + gib - 1) // gib
    needed = f"{action} requires at least {required_display_gib} GiB" + (f" ({minimum})" if minimum else "")
    if status.confidence == "unknown" and not risk_accepted:
        return (
            f"{path} is on storage with unknown physical backing free space; {needed}. "
            "Set storage.host_drive in workspace.yaml" + (f" or set {risk_setting} if this is intentional" if risk_setting else "")
        )
    if status.effective_free_bytes >= required_bytes:
        return None
    problem = f"{path} has only {status.effective_free_bytes // gib} GiB effective free on {status.backing_id}; {needed}"
    host_free = status.host_free_bytes
    if host_free is not None and status.linux_free_bytes < required_bytes <= host_free:
        distro = os.environ.get("WSL_DISTRO_NAME") or "<distro>"
        problem += (
            f"; the WSL virtual disk is the limit (Linux sees {status.linux_free_bytes // gib} GiB free, "
            f"{status.backing_id} has {host_free // gib} GiB): enlarge it from Windows with "
            f"`wsl --manage {distro} --resize <size>` (recent WSL) or free space inside Linux"
        )
    return problem


def ensure_free_bytes(path: Path, required_bytes: int, *, context: str, config: dict[str, Any] | None) -> None:
    """The RunPod stage, download, and pull gate. A store whose backing is unknown falls back to
    Linux free space here, where a local launch refuses it."""
    path.mkdir(parents=True, exist_ok=True)
    problem = storage_floor_problem(probe_storage(path, config), path, required_bytes, action=context, risk_accepted=True)
    if problem:
        raise ValueError(problem)
