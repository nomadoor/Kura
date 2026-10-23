"""Environment diagnostics for Kura workspaces."""

from __future__ import annotations

import argparse
import json
import http.client
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from kura.comfyui_models import endpoint_fingerprint, visible_model_refs
from kura.yamlio import YAMLError, load_yaml
from kura.container_scripts import script_source

from kura.backends import MUSUBI_ADAPTER_SCRIPTS
from kura.executors import _redact_secret_text, _redact_secrets
from kura.secrets import declared_secret, is_secret_name
from kura.executors.docker import DOCKER_INFO_TIMEOUT_SEC, docker_daemon_problem, docker_data_location
from kura.executors.runpod import runpod_network_volumes, runpod_pods
from kura.images import effective_image, image_names, mutable_override_warning
from kura.paths import inspect_workspace_symlinks, local_docker_mounts, local_hf_cache
from kura.storage import is_wsl as _is_wsl
from kura.storage import RECLAIM_POINTER, probe_storage, probe_storages, storage_floor_problem
from kura.run_commands.plan import local_min_free_gib
from kura.workspace import comfyui_endpoint, comfyui_endpoint_problem, validate_workspace_config, workspace_schema_description
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.workspace import workspace_relative_path as _workspace_relative_path


def _image_reference(name: str) -> str:
    return effective_image(_workspace_config(), name)["reference"]


# A container the doctor starts (the GPU probe) loads torch, so it gets longer than a daemon query.
DOCKER_PROBE_TIMEOUT_SEC = 300


def _docker_run(command: list[str], *, capture: bool = False, timeout: int = DOCKER_INFO_TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
    """A docker command for a doctor report; one that does not answer within `timeout` returns
    exit code 124 with the reason as its stderr, so a hung daemon is reported, never waited on, and
    a host without the docker command returns 127."""
    try:
        return subprocess.run(command, text=True, capture_output=capture, check=False, timeout=timeout)
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, 127, "", "the docker command was not found on PATH")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command, 124, "", f"`{' '.join(command[:3])}` did not answer within {timeout} seconds; the Docker daemon may be hung or still starting",
        )


def _docker_failure(result: subprocess.CompletedProcess[str]) -> str:
    """Why a doctor docker command gave no answer, in Docker's words (or the timeout's)."""
    said = _redact_secret_text((result.stderr or result.stdout or "").strip())[-500:]
    return f"`{' '.join(result.args[:3])}` exited with {result.returncode}" + (f": {said}" if said else "")


def _docker_root_dir(report: dict[str, Any]) -> None:
    """Where the engine keeps its data, from the one place Kura asks (`docker_data_location`):
    `root_dir` when this host can measure it, else `root_dir_not_measured` with why."""
    root, unseen = docker_data_location()
    if root is not None:
        report["root_dir"] = str(root)
    else:
        report["root_dir_not_measured"] = unseen


# A doctor report's conclusion leads, so a reader that stops early has read the answer.
CONCLUSION_KEYS = ("diagnosis", "checks", "configuration_error", "warnings", "reclaimable", "issues")


def conclusion_first(payload: dict[str, Any]) -> dict[str, Any]:
    """The same report with its conclusion keys first; every key and value is kept."""
    return {**{key: payload[key] for key in CONCLUSION_KEYS if key in payload}, **payload}


def _print_report(payload: dict[str, Any]) -> None:
    print(json.dumps(conclusion_first(payload), indent=2))


def _safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def _docker_json_lines(command: list[str]) -> list[dict[str, Any]]:
    result = _docker_run(command, capture=True)
    if result.returncode != 0:
        return []
    items: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            items.append(value)
    return items


def kura_docker_containers() -> list[dict[str, Any]]:
    return _docker_json_lines(["docker", "ps", "-a", "--filter", "label=io.kura.managed=true", "--format", "{{json .}}"])


def stopped_kura_containers(containers: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Kura's containers on this Docker host that are no longer running, of any workspace; `kura cleanup`
    offers only those of this workspace's finished runs (`kura.reclaim`)."""
    return [
        item
        for item in (kura_docker_containers() if containers is None else containers)
        if not str(item.get("State") or item.get("Status") or "").lower().startswith(("running", "up"))
    ]


def kura_docker_volumes() -> list[dict[str, Any]]:
    return _docker_json_lines(["docker", "volume", "ls", "--filter", "label=io.kura.managed=true", "--format", "{{json .}}"])


def _docker_managed_resources() -> dict[str, Any]:
    containers = kura_docker_containers()
    stopped = stopped_kura_containers(containers)
    volumes = kura_docker_volumes()
    # Counts only: a workspace that has run for a while has hundreds of these.
    summary: dict[str, Any] = {"containers": len(containers), "stopped_containers": len(stopped), "volumes": len(volumes)}
    if stopped:
        # Stopped Kura containers are in the free class (docs/adr/reclaimable-data.md): Kura removes them
        # after the user's yes, so no agent runs a state-changing Docker command itself.
        summary["remove_stopped"] = (
            "`kura cleanup plan` lists, as free, the stopped containers of this workspace's finished runs, and "
            "`kura cleanup --yes` removes them after the user's yes, and `kura run prune` removes a run's containers with "
            "the run; a container of an unfinished run, or one no run of this workspace recorded, is left"
        )
    return summary


def _bytes_from_docker_size(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    number = ""
    unit = ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
        elif not char.isspace():
            unit += char
    if not number:
        return None
    try:
        amount = float(number)
    except ValueError:
        return None
    unit = unit.lower().rstrip("b")
    scale = {
        "": 1,
        "k": 1000,
        "kb": 1000,
        "ki": 1024,
        "m": 1000**2,
        "mb": 1000**2,
        "mi": 1024**2,
        "g": 1000**3,
        "gb": 1000**3,
        "gi": 1024**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ti": 1024**4,
    }.get(unit)
    if scale is None:
        return None
    return int(amount * scale)


def _path_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return 0
    du = shutil.which("du")
    if du:
        try:
            result = subprocess.run([du, "-sb", str(path)], text=True, capture_output=True, check=False, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.split()[0])
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    total = 0
    try:
        for root, dirs, files in os.walk(path, followlinks=False):
            root_path = Path(root)
            for name in dirs + files:
                try:
                    total += (root_path / name).lstat().st_size
                except OSError:
                    continue
        return total
    except OSError:
        return None


def _root_owned_files(paths: list[Path], *, sample_limit: int = 20, scan_limit: int = 10000) -> dict[str, Any]:
    if os.name == "nt":
        return {"supported": False, "count": 0, "samples": []}
    samples: list[str] = []
    seen: set[Path] = set()
    count = 0
    scanned = 0
    truncated = False
    for base in paths:
        if not base.exists():
            continue
        try:
            iterator = os.walk(base, followlinks=False)
            for root, dirs, files in iterator:
                names = [".", *dirs, *files]
                root_path = Path(root)
                for name in names:
                    path = root_path if name == "." else root_path / name
                    try:
                        normalized = path.resolve(strict=False)
                    except OSError:
                        normalized = path
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    scanned += 1
                    if scanned > scan_limit:
                        truncated = True
                        raise StopIteration
                    try:
                        if path.lstat().st_uid == 0:
                            count += 1
                            if len(samples) < sample_limit:
                                samples.append(str(path))
                    except OSError:
                        continue
        except StopIteration:
            break
        except OSError:
            continue
    return {"supported": True, "count": count, "samples": samples, "scan_limit": scan_limit, "truncated": truncated}


def _docker_storage_summary() -> dict[str, Any]:
    docker_path = shutil.which("docker")
    summary: dict[str, Any] = {"docker_path": docker_path, "daemon_reachable": False}
    if not docker_path:
        summary["diagnosis"] = "Docker CLI was not found on PATH."
        return summary
    problem = docker_daemon_problem()
    summary["daemon_reachable"] = problem is None
    if problem:
        summary["diagnosis"] = problem
        return summary
    _docker_root_dir(summary)
    usage = _docker_run(["docker", "system", "df", "--format", "{{json .}}"], capture=True)
    items: list[dict[str, Any]] = []
    if usage.returncode == 0:
        for line in usage.stdout.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                size_bytes = _bytes_from_docker_size(item.get("Size"))
                reclaimable_bytes = _bytes_from_docker_size(str(item.get("Reclaimable", "")).split()[0])
                if size_bytes is not None:
                    item["size_bytes"] = size_bytes
                if reclaimable_bytes is not None:
                    item["reclaimable_bytes"] = reclaimable_bytes
                items.append(item)
    else:
        summary["usage_not_measured"] = _docker_failure(usage)
    summary["usage"] = items
    summary["kura_managed"] = _docker_managed_resources()
    return summary


def _disk_issue(*, code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    issue.update(details)
    return issue


def cmd_doctor_disk(_: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        config = _workspace_config()
        docker_config = config.get("docker") if isinstance(config.get("docker"), dict) else {}
        min_free_gib = local_min_free_gib(docker_config)
    except (OSError, ValueError, YAMLError) as exc:
        print(f"disk: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1

    paths = {
        "workspace": workspace_root,
        "cache": workspace_root / "cache",
        "huggingface_cache": local_hf_cache(workspace_root, config),
        "model_cache": workspace_root / "cache" / "models",
        "runs": workspace_root / "runs",
        "datasets": workspace_root / "datasets",
        "outputs": workspace_root / "outputs",
        "downloads": workspace_root / "downloads",
        "tmp": Path(os.environ.get("TMPDIR") or "/tmp"),
    }
    docker_mounts = local_docker_mounts(workspace_root, config)

    sizes = {name: {"path": str(path), "size_bytes": _path_size_bytes(path)} for name, path in paths.items()}
    storage_statuses = probe_storages(paths, config)
    docker_storage = _docker_storage_summary()
    root_owned = _root_owned_files([paths["cache"], paths["runs"]])
    symlinks = inspect_workspace_symlinks(workspace_root, mounts=[item for item in docker_mounts if isinstance(item, dict)])
    env = {
        name: os.environ.get(name)
        for name in ("KURA_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")
        if os.environ.get(name)
    }

    gib = 1024**3
    issues: list[dict[str, Any]] = []
    workspace_storage = storage_statuses["workspace"]
    # The floor a local launch keeps; an unknown backing is reported by its probe warning.
    low = storage_floor_problem(
        workspace_storage, workspace_root, min_free_gib * gib, action="a local Docker launch",
        minimum="docker.min_free_gb", risk_accepted=True,
    )
    if low:
        low = f"{low}. {RECLAIM_POINTER}"
        issues.append(_disk_issue(
            code="workspace_effective_free_low",
            severity="warning",
            message=low,
            backing_id=workspace_storage.backing_id,
            backing_kind=workspace_storage.backing_kind,
            effective_free_bytes=workspace_storage.effective_free_bytes,
            threshold_bytes=min_free_gib * gib,
        ))
    if root_owned.get("count"):
        issues.append(_disk_issue(
            code="root_owned_cache_runs",
            severity="warning",
            message="cache/runs contain root-owned files; cleanup may require permission repair",
            count=root_owned.get("count"),
        ))
    unsafe_links = symlinks.get("unsafe")
    if isinstance(unsafe_links, list) and unsafe_links:
        issues.append(_disk_issue(
            code="unsafe_workspace_symlinks",
            severity="warning",
            message="workspace contains symlinks with container-private or workspace-external absolute targets",
            count=len(unsafe_links),
        ))
    warned_backings: set[tuple[str, str]] = set()
    for status in storage_statuses.values():
        if not status.warning:
            continue
        key = (status.backing_kind, status.backing_id)
        if key in warned_backings:
            continue
        warned_backings.add(key)
        issues.append(_disk_issue(
            code="storage_backing_warning",
            severity="warning",
            message=status.warning,
            backing_id=status.backing_id,
            backing_kind=status.backing_kind,
            confidence=status.confidence,
        ))

    warnings = [issue["message"] for issue in issues if issue.get("severity") == "warning"]
    # What can be freed, by class, from the owner `kura cleanup plan` lists (docs/adr/reclaimable-data.md).
    from kura.reclaim import reclaim_inventory

    reclaimable = {**reclaim_inventory(workspace_root, config, docker=docker_storage)["totals"], "list": "kura cleanup plan"}

    # Kept short: the paths that share a backing store are listed once with its free space,
    # Docker usage is one size per type, and long lists keep their count and a few samples.
    storage: dict[str, dict[str, Any]] = {}
    for name, status in storage_statuses.items():
        entry = storage.setdefault(status.backing_id, {
            "kind": status.backing_kind, "paths": [],
            "effective_free_bytes": status.effective_free_bytes, "linux_free_bytes": status.linux_free_bytes,
            "linux_total_bytes": status.linux_total_bytes, "host_free_bytes": status.host_free_bytes,
            "confidence": status.confidence,
        })
        entry["paths"].append(name)
    docker_summary = {key: value for key, value in docker_storage.items() if key != "usage"}
    docker_summary["usage"] = {
        str(item.get("Type")): {key: item[key] for key in ("size_bytes", "reclaimable_bytes") if key in item}
        for item in docker_storage.get("usage", [])
    }
    unsafe = symlinks.get("unsafe") if isinstance(symlinks.get("unsafe"), list) else []
    samples = 5

    payload = {
        "workspace_root": str(workspace_root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "wsl": _is_wsl(),
        },
        "sizes": sizes,
        "storage": storage,
        "docker_storage": docker_summary,
        # `scan_stopped_at_limit`: the scan stopped after scan_limit paths, so count is a lower bound.
        "root_owned": {"count": root_owned.get("count", 0), "samples": root_owned.get("samples", [])[:samples], "scan_stopped_at_limit": root_owned.get("truncated", False)},
        "symlinks": {"scanned": symlinks.get("scanned"), "unsafe_count": len(unsafe), "unsafe": unsafe[:samples], "truncated": symlinks.get("truncated", False)},
        "cache_environment": env,
        "issues": issues,
        "warnings": warnings,
        "reclaimable": reclaimable,
        "diagnosis": "Disk diagnostics completed. This command is read-only.",
    }
    _print_report(_redact_secrets(payload))
    return 1 if warnings else 0


def cmd_doctor_docker(_: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        config = _workspace_config()
        images = {name: effective_image(config, name)["reference"] for name in image_names()}
    except (OSError, ValueError, YAMLError) as exc:
        print(f"docker: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    docker_path = shutil.which("docker")
    docker_exe_path = shutil.which("docker.exe")
    checks: dict[str, Any] = {"docker_command": bool(docker_path), "daemon_reachable": False, "local_images": False, "gpu_available": False}
    runtime: dict[str, Any] = {}
    docker_storage: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"docker_path": docker_path, "docker_exe_path": docker_exe_path, "wsl": _is_wsl()}
    diagnosis = ""
    if not checks["docker_command"]:
        diagnosis = "Docker CLI was not found on PATH."
    else:
        problem = docker_daemon_problem()
        checks["daemon_reachable"] = problem is None
        diagnostics["docker_daemon_problem"] = problem
        version = _docker_run(["docker", "version"], capture=True)
        diagnostics["docker_version_returncode"] = version.returncode
        diagnostics["docker_version_stdout"] = version.stdout.strip()
        diagnostics["docker_version_stderr"] = version.stderr.strip()
        if not checks["daemon_reachable"]:
            if _looks_like_process_permission_denial(problem):
                diagnosis = "This process could not access the Docker daemon because the OS denied permission. The same Kura command may work outside this process's permission context. See external-access.md, shipped with Kura as .kura/reference/external-access.md."
            else:
                diagnosis = f"Docker CLI is available but the daemon is unreachable ({problem}). If Docker Desktop settings look correct, restart Docker Desktop and this WSL distro; confirm the active Docker context points at the Desktop Linux engine."
        if checks["daemon_reachable"]:
            _docker_root_dir(docker_storage)
            usage = _docker_run(["docker", "system", "df", "--format", "{{json .}}"], capture=True)
            if usage.returncode == 0:
                docker_storage["usage"] = [json.loads(line) for line in usage.stdout.splitlines() if line.strip()]
            else:
                docker_storage["usage_not_measured"] = _docker_failure(usage)
            docker_storage["kura_managed"] = _docker_managed_resources()
            present = [name for name, image in images.items() if _docker_run(["docker", "image", "inspect", image], capture=True).returncode == 0]
            # A pinned image that is not here yet is pulled by the first run that needs it.
            checks["local_images"] = {name: "present" if name in present else "will be pulled" for name in images}
            if not present:
                checks["gpu_available"] = None
                diagnostics["images"] = images
                diagnosis = f"Docker is ready. No Kura image is pulled yet, so the GPU was not probed; the first local run pulls the image it needs, or run, for example: docker pull {images['ai-toolkit']}"
            if present:
                # Every Kura image has python with torch; the entrypoint differs, so name python.
                probe_image = images[present[0]]
                diagnostics["gpu_probe_image"] = probe_image
                gpu_probe = _docker_run(["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "python", probe_image, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"], capture=True, timeout=DOCKER_PROBE_TIMEOUT_SEC)
                checks["gpu_available"] = gpu_probe.returncode == 0
                if gpu_probe.returncode:
                    # What the probe said is the only evidence of why; a later probe may pass.
                    diagnostics["gpu_probe_error"] = _redact_secret_text((gpu_probe.stderr or gpu_probe.stdout).strip())[-2000:]
                if "ai-toolkit" in present:
                    # Only the AI-Toolkit image records its runtime here.
                    runtime_result = _docker_run(["docker", "run", "--rm", "--entrypoint", "cat", images["ai-toolkit"], "/opt/kura-runtime.json"], capture=True, timeout=DOCKER_PROBE_TIMEOUT_SEC)
                    if runtime_result.returncode == 0:
                        try:
                            runtime = json.loads(runtime_result.stdout)
                        except json.JSONDecodeError:
                            runtime = {"runtime_metadata": "invalid"}
                if not checks["gpu_available"]:
                    diagnosis = "The local image cannot access a GPU. In WSL, confirm Docker Desktop WSL integration and NVIDIA Container Toolkit support."
    cache_path = local_hf_cache(workspace_root, config)
    cache: dict[str, Any] = {"path": str(cache_path), "exists": cache_path.exists()}
    try:
        cache["free_bytes"] = probe_storage(cache_path, config).effective_free_bytes
    except OSError:
        cache["free_bytes"] = None
    if not cache_path.exists():
        cache["note"] = "Kura creates this cache directory before a local Docker launch. Set docker.hf_cache in workspace.yaml to keep it elsewhere."
    if checks["daemon_reachable"] and not diagnosis:
        diagnosis = "Docker is ready. Keep Docker Desktop and WSL updated; configure global memory/swap limits outside Kura only when the host requires them."
    _print_report({**checks, "workspace_root": str(workspace_root), "runtime": runtime, "huggingface_cache": cache, "docker_storage": docker_storage, "diagnostics": diagnostics, "diagnosis": diagnosis})
    return 0 if all(value is not False for value in checks.values()) else 1


def _musubi_probe_items() -> list[tuple[str, str]]:
    return [
        (adapter, script)
        for adapter, scripts in MUSUBI_ADAPTER_SCRIPTS.items()
        for script in scripts
    ]


def cmd_doctor_musubi(args: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        image = args.image or _image_reference("musubi-tuner")
    except (OSError, ValueError, YAMLError) as exc:
        print(f"musubi: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    docker = shutil.which("docker")
    checks: dict[str, Any] = {
        "docker_command": bool(docker),
        "local_image": False,
        "adapter_scripts_exist": False,
        "adapter_help_smoke": False if not args.skip_help else None,
    }
    diagnostics: dict[str, Any] = {
        "workspace_root": str(workspace_root),
        "image": image,
        "script_root": "/opt/musubi-tuner/src/musubi_tuner",
        "help_smoke": not args.skip_help,
        "gpu": not args.no_gpu,
        "script_timeout_seconds": args.script_timeout,
        "parallel_help_workers": 4 if not args.skip_help else 0,
    }
    if not docker:
        diagnosis = "Docker CLI was not found on PATH."
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis})
        return 1
    image_check = subprocess.run([docker, "image", "inspect", image], text=True, capture_output=True, check=False, timeout=30)
    checks["local_image"] = image_check.returncode == 0
    if not checks["local_image"]:
        diagnostics["image_inspect_stderr"] = _redact_secret_text(image_check.stderr.strip())
        diagnosis = f"The Musubi Tuner image is not pulled yet. Pull it with: docker pull {image}"
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis})
        return 1

    probe_code = script_source("musubi_probe.py")

    command = [
        docker,
        "run",
        "--rm",
    ]
    if not args.no_gpu:
        command.extend(["--gpus", "all"])
    command.extend([
        "--entrypoint",
        "python",
        image,
        "-c",
        probe_code,
        json.dumps(_musubi_probe_items()),
        "0" if args.skip_help else "1",
        str(args.script_timeout),
    ])
    try:
        probe = subprocess.run(command, text=True, capture_output=True, check=False, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        diagnostics["probe_error"] = f"timed out after {exc.timeout}s"
        diagnosis = "Musubi adapter probe timed out; inspect the local image and try a larger --timeout."
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis})
        return 1
    diagnostics["probe_returncode"] = probe.returncode
    if probe.stderr.strip():
        diagnostics["probe_stderr"] = _redact_secret_text(probe.stderr.strip()[-4000:])
    payload = None
    for line in reversed(probe.stdout.splitlines()):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            diagnostics["scripts"] = results
            checks["adapter_scripts_exist"] = all(isinstance(item, dict) and item.get("exists") is True for item in results)
            if not args.skip_help:
                checks["adapter_help_smoke"] = all(isinstance(item, dict) and item.get("help_returncode") == 0 for item in results)
    else:
        diagnostics["probe_stdout_tail"] = _redact_secret_text(probe.stdout[-4000:])
    if checks["adapter_scripts_exist"] and (args.skip_help or checks["adapter_help_smoke"]):
        diagnosis = "Musubi adapter scripts are present in the configured image and the smoke check passed."
    else:
        diagnosis = "Musubi adapter smoke failed. The configured image/ref may not contain all Kura adapter scripts or their imports may not start."
    _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis})
    return 0 if checks["adapter_scripts_exist"] and (args.skip_help or checks["adapter_help_smoke"]) else 1


def cmd_doctor_sd_scripts(args: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        image = args.image or _image_reference("sd-scripts")
    except (OSError, ValueError, YAMLError) as exc:
        print(f"sd-scripts: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    docker = shutil.which("docker")
    checks: dict[str, Any] = {"docker_command": bool(docker), "local_image": False, "probe_exit": False, "tier1_scripts": False, "imports": False, "sd_checkpoint_symlink_compat": False, "sdxl_checkpoint_symlink_compat": False, "gpu_available": None if args.no_gpu else False}
    diagnostics: dict[str, Any] = {"workspace_root": str(workspace_root), "image": image, "script_root": "/opt/sd-scripts", "gpu": not args.no_gpu}
    if not docker:
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": "Docker CLI was not found on PATH."})
        return 1
    try:
        inspected = subprocess.run([docker, "image", "inspect", image], text=True, capture_output=True, check=False, timeout=30)
    except subprocess.TimeoutExpired:
        diagnostics["image_inspect_error"] = "timed out after 30s"
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": "Configured sd-scripts image inspection timed out."})
        return 1
    checks["local_image"] = inspected.returncode == 0
    if not checks["local_image"]:
        diagnostics["image_inspect_stderr"] = _redact_secret_text(inspected.stderr.strip())
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": f"The sd-scripts image is not pulled yet. Pull it with: docker pull {image}"})
        return 1
    command = [docker, "run", "--rm"]
    if not args.no_gpu:
        command.extend(["--gpus", "all"])
    command.extend(["--entrypoint", "python", image, "-c", script_source("sd_scripts_probe.py")])
    try:
        probe = subprocess.run(command, text=True, capture_output=True, check=False, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        diagnostics["probe_error"] = f"timed out after {exc.timeout}s"
        _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": "sd-scripts image probe timed out."})
        return 1
    diagnostics["probe_returncode"] = probe.returncode
    checks["probe_exit"] = probe.returncode == 0
    if probe.stderr.strip():
        diagnostics["probe_stderr"] = _redact_secret_text(probe.stderr.strip()[-4000:])
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError:
        payload = None
        diagnostics["probe_stdout_tail"] = _redact_secret_text(probe.stdout[-4000:])
    if isinstance(payload, dict):
        diagnostics["probe"] = payload
        scripts = payload.get("scripts") if isinstance(payload.get("scripts"), dict) else {}
        imports = payload.get("imports") if isinstance(payload.get("imports"), dict) else {}
        compatibility = payload.get("compatibility") if isinstance(payload.get("compatibility"), dict) else {}
        checks["tier1_scripts"] = bool(scripts) and all(
            isinstance(value, dict)
            and value.get("exists") is True
            and value.get("help_exit_code") == 0
            and all(option is True for option in (value.get("required_options") or {}).values())
            for value in scripts.values()
        )
        checks["imports"] = bool(imports) and all(not isinstance(value, dict) for value in imports.values())
        checks["sd_checkpoint_symlink_compat"] = compatibility.get("sd_checkpoint_symlink_safe") is True
        checks["sdxl_checkpoint_symlink_compat"] = compatibility.get("sdxl_checkpoint_symlink_safe") is True
        if not args.no_gpu:
            checks["gpu_available"] = payload.get("torch", {}).get("cuda_available") is True if isinstance(payload.get("torch"), dict) else False
    passed = all(value is True or value is None for value in checks.values())
    diagnosis = "sd-scripts Tier 1 scripts, imports, and runtime passed." if passed else "sd-scripts image probe failed; inspect the per-script and import diagnostics."
    _print_report({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis})
    return 0 if passed else 1


def cmd_doctor_runpod(_: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        workspace_config = _workspace_config()
        config = workspace_config.get("runpod", {})
    except (OSError, ValueError, YAMLError) as exc:
        print(f"runpod: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    api_key = declared_secret("RUNPOD_API_KEY")
    checks: dict[str, Any] = {
        "api_key": bool(api_key),
        "pod_list": False,
        "pods_empty": None,
        "network_volume_list": None,
        "network_volumes_empty": None,
        "images_pinned": True,
    }
    diagnostics: dict[str, Any] = {"config": {key: value for key, value in config.items() if not is_secret_name(key)}}
    images = [effective_image(workspace_config, name) for name in image_names()]
    mutable_images = {image["name"]: image["reference"] for image in images if mutable_override_warning(image)}
    checks["images_pinned"] = not mutable_images
    if mutable_images:
        diagnostics["mutable_images"] = mutable_images
    if api_key:
        # Kura talks to RunPod only through its API, with RUNPOD_API_KEY; no local RunPod CLI is needed.
        try:
            pods = runpod_pods(api_key)
            checks["pod_list"] = True
            checks["pods_empty"] = pods == []
            diagnostics["pods"] = _redact_secrets(pods)
        except Exception as exc:  # read-only doctor; keep diagnosis broad.
            diagnostics["pods_error"] = _redact_secret_text(str(exc))
        try:
            volumes = runpod_network_volumes(api_key)
            checks["network_volume_list"] = True
            checks["network_volumes_empty"] = volumes == []
            diagnostics["network_volumes"] = _redact_secrets(volumes)
        except Exception as exc:  # a listing RunPod refuses is reported, never a doctor failure
            diagnostics["network_volumes_error"] = "could not list network volumes: " + _redact_secret_text(str(exc))
    ok = bool(checks["api_key"] and checks["pod_list"] and checks["pods_empty"] is not False and checks["network_volumes_empty"] is not False)
    if ok and mutable_images:
        diagnosis = "The RunPod API is ready, but images.<name> overrides with mutable tags should be pinned before reproducible runs."
    elif ok:
        diagnosis = "The RunPod API is ready."
    elif checks["pods_empty"] is False:
        diagnosis = "RunPod has Pods remaining; delete stopped/exited Pods if they should not persist."
    elif checks["network_volumes_empty"] is False:
        diagnosis = "RunPod has Network Volumes remaining; delete volumes that should not persist."
    elif _looks_like_process_permission_denial(diagnostics.get("pods_error")):
        diagnosis = "This process could not reach the RunPod API because the OS denied the connection before RunPod responded. The same Kura command may work outside this process's permission context. See external-access.md, shipped with Kura as .kura/reference/external-access.md."
    else:
        diagnosis = "RunPod is not fully ready; inspect checks and diagnostics."
    _print_report(_redact_secrets({"workspace_root": str(workspace_root), "checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}))
    return 0 if ok else 1


def _comfyui_lora_names(object_info: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for class_type in ("LoraLoader", "LoraLoaderModelOnly"):
        loader = object_info.get(class_type)
        if not isinstance(loader, dict):
            continue
        required = loader.get("input", {}).get("required") if isinstance(loader.get("input"), dict) else None
        lora_name = required.get("lora_name") if isinstance(required, dict) else None
        if isinstance(lora_name, list) and lora_name and isinstance(lora_name[0], list):
            names.update(str(item) for item in lora_name[0])
    return names


def _comfyui_lora_count(object_info: dict[str, Any]) -> int | None:
    names = _comfyui_lora_names(object_info)
    return len(names) if names else None


def _fetch_comfyui_object_info(endpoint: str, *, timeout: float = 5) -> dict[str, Any]:
    with urllib.request.urlopen(f"{endpoint}/object_info", timeout=timeout) as response:
        object_info = json.loads(response.read().decode("utf-8"))
    return object_info if isinstance(object_info, dict) else {}


def _probe_comfyui_lora_stage(endpoint: str, stage_dir: Path, stage_subdir: str) -> tuple[bool, str]:
    stage_dir_created = not stage_dir.exists()
    stage_dir.mkdir(parents=True, exist_ok=True)
    probe_name = f"kura-doctor-probe-{uuid.uuid4().hex}.safetensors"
    probe_path = stage_dir / probe_name
    lora_name = f"{stage_subdir}/{probe_name}"
    try:
        with probe_path.open("xb") as handle:
            handle.write(b"kura doctor ComfyUI LoRA visibility probe\n")
        object_info = _fetch_comfyui_object_info(endpoint)
        return lora_name in _comfyui_lora_names(object_info), lora_name
    finally:
        try:
            probe_path.unlink()
        except OSError:
            pass
        if stage_dir_created:
            try:
                stage_dir.rmdir()
            except OSError:
                pass


def _kura_stage_files(stage_dir: Path | None) -> list[str]:
    if stage_dir is None or not stage_dir.is_dir():
        return []
    try:
        return sorted(item.name for item in stage_dir.iterdir() if item.is_file() or item.is_symlink())
    except OSError:
        return []


def cmd_doctor_comfyui(args: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        config = _workspace_config()
    except (OSError, ValueError, YAMLError) as exc:
        print(f"comfyui: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    endpoint = str(getattr(args, "endpoint", None) or comfyui_endpoint(config)).rstrip("/")
    if problem := comfyui_endpoint_problem(endpoint, where="--endpoint"):
        print(f"comfyui: configuration error: {problem}", file=sys.stderr)
        return 1
    parsed_endpoint = urllib.parse.urlparse(endpoint)
    lora_dir = _workspace_relative_path(str(comfyui["lora_dir"])) if isinstance(comfyui.get("lora_dir"), str) and comfyui.get("lora_dir") else None
    stage_subdir = str(comfyui.get("lora_stage_subdir") or "Kura_tmp").strip("/\\")
    stage_dir = lora_dir / stage_subdir if lora_dir is not None and stage_subdir else None
    checks = {
        "endpoint_reachable": False,
        "object_info": False,
        "lora_dir_configured": lora_dir is not None,
        "lora_dir_exists": bool(lora_dir and lora_dir.is_dir()),
        "stage_dir_exists": bool(stage_dir and stage_dir.is_dir()),
        "stage_dir_writable": bool(stage_dir and stage_dir.is_dir() and os.access(stage_dir, os.W_OK)),
        "lora_stage_visible": None,
        "core_model_patch_loader": None,
        "core_anima_lllite_apply": None,
    }
    diagnostics: dict[str, Any] = {
        "endpoint": endpoint,
        "lora_dir": str(lora_dir) if lora_dir else None,
        "stage_dir": str(stage_dir) if stage_dir else None,
    }
    warnings: list[str] = []
    lora_dir_inside_workspace = False
    if lora_dir is not None:
        try:
            lora_dir.resolve().relative_to(workspace_root.resolve())
            lora_dir_inside_workspace = True
        except ValueError:
            pass
    checks["lora_dir_outside_workspace"] = not lora_dir_inside_workspace if lora_dir is not None else None
    if lora_dir_inside_workspace:
        warnings.append(
            "configured comfyui.lora_dir is inside the Kura workspace; this looks like a managed/smoke directory, not a user ComfyUI model directory"
        )
    stage_files = _kura_stage_files(stage_dir)
    diagnostics["kura_stage_file_count"] = len(stage_files)
    if stage_files:
        diagnostics["kura_stage_file_samples"] = stage_files[:10]
    if parsed_endpoint.scheme not in ("http", "https"):
        diagnostics["object_info_error"] = f"unsupported comfyui.endpoint scheme: {parsed_endpoint.scheme or '(none)'}"
        diagnosis = "ComfyUI endpoint is not ready; comfyui.endpoint must start with http:// or https://."
        _print_report(_redact_secrets({"workspace_root": str(workspace_root), "checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}))
        return 1
    object_info: dict[str, Any] | None = None
    try:
        object_info = _fetch_comfyui_object_info(endpoint)
        checks["endpoint_reachable"] = True
        checks["object_info"] = isinstance(object_info, dict)
        if isinstance(object_info, dict):
            diagnostics["lora_loader_count"] = _comfyui_lora_count(object_info)
            diagnostics["endpoint_identity"] = endpoint_fingerprint(object_info)
            checks["core_model_patch_loader"] = "ModelPatchLoader" in object_info
            checks["core_anima_lllite_apply"] = "AnimaLLLiteApply" in object_info
    except Exception as exc:
        diagnostics["object_info_error"] = _redact_secret_text(str(exc))
        configured_default = endpoint == "http://127.0.0.1:8188"
        if not configured_default:
            try:
                candidate_info = _fetch_comfyui_object_info("http://127.0.0.1:8188", timeout=1)
            except Exception:
                pass
            else:
                diagnostics["candidate_endpoint"] = "http://127.0.0.1:8188"
                diagnostics["candidate_endpoint_identity"] = endpoint_fingerprint(candidate_info)
                warnings.append(
                    "another ComfyUI is responding at http://127.0.0.1:8188; verify it with the user, but do not retarget automatically"
                )
    workflow_arg = getattr(args, "workflow", None)
    if workflow_arg:
        checks["workflow_loaded"] = False
        checks["workflow_models_visible"] = None
        workflow_path = Path(str(workflow_arg))
        if not workflow_path.is_absolute():
            workflow_path = workspace_root / workflow_path
        diagnostics["workflow"] = str(workflow_path)
        try:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            if not isinstance(workflow, dict):
                raise ValueError("workflow must be an API-format JSON object")
            checks["workflow_loaded"] = True
            if not isinstance(object_info, dict):
                raise ValueError("configured endpoint object_info is unavailable")
            visible, missing = visible_model_refs(workflow, object_info)
            diagnostics["workflow_required_models"] = visible + missing
            diagnostics["workflow_missing_models"] = missing
            checks["workflow_models_visible"] = not missing
        except Exception as exc:
            if checks["workflow_loaded"]:
                checks["workflow_models_visible"] = False
            diagnostics["workflow_check_error"] = _redact_secret_text(str(exc))
    if getattr(args, "probe_stage", False):
        if not (stage_dir and lora_dir and lora_dir.is_dir()):
            diagnostics["lora_stage_probe_error"] = "comfyui.lora_dir must exist before probing stage visibility"
            checks["lora_stage_visible"] = False
        elif not checks["endpoint_reachable"]:
            diagnostics["lora_stage_probe_error"] = "ComfyUI endpoint must be reachable before probing stage visibility"
            checks["lora_stage_visible"] = False
        else:
            try:
                visible, lora_name = _probe_comfyui_lora_stage(endpoint, stage_dir, stage_subdir)
                checks["stage_dir_exists"] = bool(stage_dir.is_dir())
                checks["stage_dir_writable"] = bool(stage_dir.is_dir() and os.access(stage_dir, os.W_OK))
                checks["lora_stage_visible"] = visible
                diagnostics["lora_stage_probe_name"] = lora_name
            except Exception as exc:
                checks["lora_stage_visible"] = False
                diagnostics["lora_stage_probe_error"] = _redact_secret_text(str(exc))
    if stage_dir and not stage_dir.exists() and lora_dir and lora_dir.is_dir():
        diagnostics["stage_parent_writable"] = os.access(lora_dir, os.W_OK)
    if checks["endpoint_reachable"] and checks["object_info"] and checks["lora_stage_visible"] is not False:
        diagnosis = "ComfyUI endpoint is reachable."
    elif checks["endpoint_reachable"] and checks["lora_stage_visible"] is False:
        stage_error = diagnostics.get("lora_stage_probe_error")
        if not checks["lora_dir_configured"]:
            diagnosis = "ComfyUI is reachable, but comfyui.lora_dir is not configured."
        elif not checks["lora_dir_exists"]:
            diagnosis = "ComfyUI is reachable, but configured comfyui.lora_dir does not exist."
        elif _looks_like_process_permission_denial(stage_error) or (
            checks["stage_dir_exists"] and not checks["stage_dir_writable"]
        ) or diagnostics.get("stage_parent_writable") is False:
            diagnosis = "ComfyUI is reachable, but this process could not write to the configured LoRA staging directory. The same Kura command may work outside this process's permission context. See external-access.md, shipped with Kura as .kura/reference/external-access.md."
        else:
            diagnosis = "ComfyUI endpoint is reachable, but configured comfyui.lora_dir is not visible to that endpoint."
    else:
        diagnosis = (
            "ComfyUI endpoint is not reachable. Ask the user to start their local ComfyUI or correct comfyui.endpoint. "
            "Do not start a Docker ComfyUI: Kura's managed image is only for RunPod and disposable smoke tests."
        )
    if checks.get("workflow_models_visible") is False and checks["endpoint_reachable"]:
        diagnosis = (
            "ComfyUI is reachable, but it cannot see every workflow-required model. Verify the intended endpoint and the user's "
            "ComfyUI model paths. Local render never downloads models."
        )
    _print_report(_redact_secrets({"workspace_root": str(workspace_root), "checks": checks, "diagnostics": diagnostics, "warnings": warnings, "diagnosis": diagnosis}))
    ok = checks["endpoint_reachable"] and checks["object_info"] and checks["lora_stage_visible"] is not False and checks.get("workflow_models_visible") is not False
    return 0 if ok else 1


def _looks_like_process_permission_denial(value: Any) -> bool:
    """Recognize OS denial wording without guessing which sandbox produced it."""

    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(
        phrase in lowered
        for phrase in (
            "operation not permitted",
            "permission denied",
            "read-only file system",
            "access is denied",
        )
    )


def cmd_doctor_secrets(_: argparse.Namespace) -> int:
    config = Path.home() / ".docker" / "config.json"
    registries: list[str] = []
    try:
        registries = sorted(json.loads(config.read_text(encoding="utf-8")).get("auths", {}).keys())
    except (OSError, json.JSONDecodeError):
        pass
    from kura.secrets import sources, user_secrets_path

    _print_report({"secrets": sources(), "user_secrets_file": str(user_secrets_path()), "docker_login_registries": registries})
    return 0


def _workspace_image_diagnostics(workspace: Path) -> tuple[dict[str, Any], list[str]]:
    """The image each Kura backend runs in this workspace, and warnings for mutable overrides."""
    workspace_yaml = workspace / "workspace.yaml"
    try:
        config = load_yaml(workspace_yaml) if workspace_yaml.is_file() else {}
    except (OSError, YAMLError) as exc:
        return {}, [f"workspace.yaml could not be read: {_safe_error(exc)}"]
    if not isinstance(config, dict):
        return {}, ["workspace.yaml must contain a YAML mapping"]
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    for name in image_names():
        try:
            image = effective_image(config, name)
        except ValueError as exc:
            warnings.append(_safe_error(exc))
            continue
        diagnostics[name] = {"image": image["reference"], "origin": image["origin"]}
        warning = mutable_override_warning(image)
        if warning:
            warnings.append(warning)
    return diagnostics, warnings


def cmd_doctor_workspace(_: argparse.Namespace) -> int:
    workspace = _workspace()
    image_diagnostics, warnings = _workspace_image_diagnostics(workspace)
    configuration_error: str | None = None
    workspace_yaml = workspace / "workspace.yaml"
    if workspace_yaml.is_file():
        try:
            _require_workspace()
            validate_workspace_config(load_yaml(workspace_yaml))
        except (OSError, ValueError, YAMLError) as exc:
            configuration_error = _safe_error(exc)
            warnings.append(configuration_error)
    from kura.runner import logout_stops_runner

    logout_warning = logout_stops_runner()
    if logout_warning:
        warnings.append(logout_warning)
    subdirs = {name: (workspace / name).is_dir() for name in ("datasets", "runs", "workflows", "promptsets")}
    _print_report({
        "workspace_root": str(workspace),
        "workspace_yaml": (workspace / "workspace.yaml").is_file(),
        "subdirs": subdirs,
        "docker_images": image_diagnostics,
        "settings": workspace_schema_description(),
        "configuration_error": configuration_error,
        "warnings": warnings,
    })
    return 1 if warnings else 0


def readiness_gaps(root: Path) -> list[str]:
    """What this machine still needs before Kura can train or render; empty when ready.

    A quick subset of the doctor checks for `kura init`. Each line names the
    command that diagnoses it in full.
    """

    gaps: list[str] = []
    if not shutil.which("docker"):
        gaps.append("Docker CLI was not found: local training needs Docker; RunPod training does not. See `kura doctor docker`.")
    else:
        if problem := docker_daemon_problem():
            gaps.append(f"The Docker daemon is not reachable ({problem}): local training needs it; RunPod training does not. See `kura doctor docker`.")
    config_path = root / "workspace.yaml"
    try:
        config = load_yaml(config_path) if config_path.is_file() else {}
    except (OSError, YAMLError):
        config = {}
    config = config if isinstance(config, dict) else {}
    if not declared_secret("RUNPOD_API_KEY"):
        gaps.append("RUNPOD_API_KEY is not set: RunPod training and renders need it. Run `kura secrets set RUNPOD_API_KEY` in your own terminal.")
    endpoint = comfyui_endpoint(config)
    if problem := comfyui_endpoint_problem(endpoint, where="comfyui.endpoint in workspace.yaml"):
        gaps.append(f"{problem}. See `kura doctor workspace`.")
        return gaps
    try:
        with urllib.request.urlopen(f"{endpoint}/system_stats", timeout=2):
            pass
    except (OSError, ValueError, http.client.HTTPException):
        gaps.append(f"ComfyUI is not reachable at {endpoint}: local renders need it; set comfyui.endpoint in workspace.yaml. See `kura doctor comfyui`.")
    return gaps
