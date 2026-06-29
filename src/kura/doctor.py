"""Read-only environment diagnostics for Kura workspaces."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from kura.backends import MUSUBI_ADAPTER_SCRIPTS
from kura.executors import _redact_secret_text, _redact_secrets
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.workspace import workspace_relative_path as _workspace_relative_path


def _image_config(name: str) -> dict[str, Any]:
    try:
        image = _workspace_config()["docker"]["images"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"workspace.yaml has no docker.images.{name} configuration") from exc
    if not isinstance(image, dict) or not all(isinstance(image.get(key), str) for key in ("local", "remote", "dockerfile", "context")):
        raise ValueError(f"docker.images.{name} requires local, remote, dockerfile, and context strings")
    return image


def _docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=capture, check=False)


def _secret_state() -> dict[str, str]:
    return {name: "present" if os.environ.get(name) else "absent" for name in ("HF_TOKEN", "RUNPOD_API_KEY", "GHCR_TOKEN", "DOCKERHUB_TOKEN")}


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


def _docker_managed_resources() -> dict[str, Any]:
    containers = _docker_json_lines(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=io.kura.managed=true",
            "--format",
            "{{json .}}",
        ]
    )
    volumes = _docker_json_lines(
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            "label=io.kura.managed=true",
            "--format",
            "{{json .}}",
        ]
    )
    stopped = [
        item
        for item in containers
        if not str(item.get("State") or item.get("Status") or "").lower().startswith(("running", "up"))
    ]
    return {"containers": containers, "stopped_containers": stopped, "volumes": volumes}


def cmd_doctor_docker(_: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        image = _image_config("ai-toolkit")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"docker: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    docker_path = shutil.which("docker")
    docker_exe_path = shutil.which("docker.exe")
    checks: dict[str, Any] = {"docker_command": bool(docker_path), "daemon_reachable": False, "local_image": False, "gpu_available": False}
    runtime: dict[str, Any] = {}
    docker_storage: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"docker_path": docker_path, "docker_exe_path": docker_exe_path, "wsl": "microsoft" in platform.uname().release.lower()}
    diagnosis = ""
    if not checks["docker_command"]:
        diagnosis = "Docker CLI was not found on PATH."
    else:
        info = _docker_run(["docker", "info"], capture=True)
        checks["daemon_reachable"] = info.returncode == 0
        diagnostics["docker_info_returncode"] = info.returncode
        diagnostics["docker_info_stderr"] = info.stderr.strip()
        version = _docker_run(["docker", "version"], capture=True)
        diagnostics["docker_version_returncode"] = version.returncode
        diagnostics["docker_version_stdout"] = version.stdout.strip()
        diagnostics["docker_version_stderr"] = version.stderr.strip()
        if not checks["daemon_reachable"]:
            diagnosis = "Docker CLI is available but the daemon is unreachable. If Docker Desktop settings look correct, restart Docker Desktop and this WSL distro; confirm the active Docker context points at the Desktop Linux engine."
        if checks["daemon_reachable"]:
            root_dir = _docker_run(["docker", "info", "--format", "{{.DockerRootDir}}"], capture=True)
            if root_dir.returncode == 0:
                docker_storage["root_dir"] = root_dir.stdout.strip()
            usage = _docker_run(["docker", "system", "df", "--format", "{{json .}}"], capture=True)
            if usage.returncode == 0:
                docker_storage["usage"] = [json.loads(line) for line in usage.stdout.splitlines() if line.strip()]
            docker_storage["kura_managed"] = _docker_managed_resources()
            checks["local_image"] = _docker_run(["docker", "image", "inspect", image["local"]], capture=True).returncode == 0
            if not checks["local_image"]:
                diagnosis = "Docker daemon is reachable but local image is missing. Run: kura image build ai-toolkit"
            if checks["local_image"]:
                gpu_probe = _docker_run(["docker", "run", "--rm", "--gpus", "all", image["local"], "python", "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"], capture=True)
                checks["gpu_available"] = gpu_probe.returncode == 0
                runtime_result = _docker_run(["docker", "run", "--rm", "--entrypoint", "cat", image["local"], "/opt/kura-runtime.json"], capture=True)
                if runtime_result.returncode == 0:
                    try:
                        runtime = json.loads(runtime_result.stdout)
                    except json.JSONDecodeError:
                        runtime = {"runtime_metadata": "invalid"}
                if not checks["gpu_available"]:
                    diagnosis = "The local image cannot access a GPU. In WSL, confirm Docker Desktop WSL integration and NVIDIA Container Toolkit support."
    mounts = _workspace_config().get("docker", {}).get("mounts", [])
    cache_path = next((_workspace_relative_path(item["source"]) for item in mounts if isinstance(item, dict) and isinstance(item.get("source"), str) and item.get("target") == "/root/.cache/huggingface"), None)
    cache: dict[str, Any] = {"path": str(cache_path) if cache_path else None, "exists": bool(cache_path and cache_path.exists()), "default_path": str(_workspace_relative_path("./cache/huggingface"))}
    if cache_path:
        try:
            filesystem_path = cache_path
            while not filesystem_path.exists() and filesystem_path != filesystem_path.parent:
                filesystem_path = filesystem_path.parent
            usage = shutil.disk_usage(filesystem_path)
            cache["free_bytes"] = usage.free
        except OSError:
            cache["free_bytes"] = None
        if not cache_path.exists():
            cache["note"] = "Kura will create this cache directory before local Docker launch. Change docker.mounts[].source in workspace.yaml if you want another location."
    else:
        cache["note"] = "No Hugging Face cache mount is configured. Add docker.mounts source ./cache/huggingface target /root/.cache/huggingface to reuse downloads across local Docker runs."
    if checks["daemon_reachable"] and not diagnosis:
        diagnosis = "Docker is ready. Keep Docker Desktop and WSL updated; configure global memory/swap limits outside Kura only when the host requires them."
    print(json.dumps({**checks, "workspace_root": str(workspace_root), "runtime": runtime, "huggingface_cache": cache, "docker_storage": docker_storage, "diagnostics": diagnostics, "diagnosis": diagnosis}, indent=2))
    return 0 if all(checks.values()) else 1


def _musubi_probe_items() -> list[tuple[str, str]]:
    return [
        (adapter, script)
        for adapter, scripts in MUSUBI_ADAPTER_SCRIPTS.items()
        for script in scripts
    ]


def cmd_doctor_musubi(args: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        image = _image_config("musubi-tuner")["local"]
    except (OSError, ValueError, yaml.YAMLError) as exc:
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
    }
    if not docker:
        diagnosis = "Docker CLI was not found on PATH."
        print(json.dumps({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}, indent=2))
        return 1
    image_check = subprocess.run([docker, "image", "inspect", image], text=True, capture_output=True, check=False, timeout=30)
    checks["local_image"] = image_check.returncode == 0
    if not checks["local_image"]:
        diagnostics["image_inspect_stderr"] = _redact_secret_text(image_check.stderr.strip())
        diagnosis = "Configured Musubi local image is missing. Build it with: kura image build musubi-tuner"
        print(json.dumps({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}, indent=2))
        return 1

    probe_code = r"""
import json
import os
import subprocess
import sys

root = "/opt/musubi-tuner/src/musubi_tuner"
items = json.loads(sys.argv[1])
do_help = sys.argv[2] == "1"
script_timeout = float(sys.argv[3])
results = []
ok = True
for adapter, script in items:
    path = os.path.join(root, script)
    item = {"adapter": adapter, "script": script, "exists": os.path.isfile(path)}
    if not item["exists"]:
        ok = False
    elif do_help:
        try:
            proc = subprocess.run(
                [sys.executable, path, "--help"],
                cwd="/opt/musubi-tuner",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=script_timeout,
            )
            item["help_returncode"] = proc.returncode
            if proc.returncode != 0:
                ok = False
                item["help_output_tail"] = proc.stdout[-2000:]
        except Exception as exc:
            ok = False
            item["help_error"] = str(exc)
    results.append(item)
print(json.dumps({"script_root": root, "results": results}))
raise SystemExit(0 if ok else 1)
""".strip()
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
        print(json.dumps({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}, indent=2))
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
    print(json.dumps({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}, indent=2))
    return 0 if checks["adapter_scripts_exist"] and (args.skip_help or checks["adapter_help_smoke"]) else 1


def cmd_doctor_runpod(_: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        config = _workspace_config().get("runpod", {})
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"runpod: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    api_key_env = config.get("api_key_env", "RUNPOD_API_KEY")
    api_key_present = isinstance(api_key_env, str) and bool(os.environ.get(api_key_env))
    checks: dict[str, Any] = {
        "runpodctl_command": bool(shutil.which("runpodctl")),
        "api_key": api_key_present,
        "pod_list": False,
        "rest_pods": False,
        "pods_empty": None,
        "network_volumes_empty": None,
    }
    diagnostics: dict[str, Any] = {"runpodctl_path": shutil.which("runpodctl"), "api_key_env": api_key_env, "config": {key: value for key, value in config.items() if "key" not in key.lower() and "secret" not in key.lower()}}
    if checks["runpodctl_command"]:
        version = subprocess.run(["runpodctl", "version"], text=True, capture_output=True, check=False)
        diagnostics["runpodctl_version"] = _redact_secret_text((version.stdout or version.stderr).strip())
    if checks["runpodctl_command"] and api_key_present:
        pods = subprocess.run(["runpodctl", "pod", "list"], text=True, capture_output=True, check=False)
        checks["pod_list"] = pods.returncode == 0
        diagnostics["pod_list_returncode"] = pods.returncode
        diagnostics["pod_list_stdout"] = _redact_secret_text(pods.stdout.strip())
        diagnostics["pod_list_stderr"] = _redact_secret_text(pods.stderr.strip())
        try:
            request = urllib.request.Request("https://rest.runpod.io/v1/pods", headers={"Authorization": "Bearer " + os.environ[str(api_key_env)], "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=20) as response:
                all_pods = json.loads(response.read().decode("utf-8"))
            checks["rest_pods"] = True
            diagnostics["pods"] = _redact_secrets(all_pods)
            checks["pods_empty"] = all_pods == []
        except Exception as exc:
            diagnostics["pods_error"] = _redact_secret_text(str(exc))
        try:
            request = urllib.request.Request("https://rest.runpod.io/v1/networkvolumes", headers={"Authorization": "Bearer " + os.environ[str(api_key_env)], "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=20) as response:
                volumes = json.loads(response.read().decode("utf-8"))
            diagnostics["network_volumes"] = _redact_secrets(volumes)
            checks["network_volumes_empty"] = volumes == []
        except Exception as exc:  # read-only doctor; keep diagnosis broad.
            diagnostics["network_volumes_error"] = _redact_secret_text(str(exc))
    ok = bool(checks["runpodctl_command"] and checks["api_key"] and checks["pod_list"] and checks["rest_pods"] and checks["pods_empty"] is not False and checks["network_volumes_empty"] is True)
    if ok:
        diagnosis = "RunPod CLI/API are ready."
    elif checks["pods_empty"] is False:
        diagnosis = "RunPod has Pods remaining; delete stopped/exited Pods if they should not persist."
    elif checks["network_volumes_empty"] is False:
        diagnosis = "RunPod has Network Volumes remaining; delete volumes that should not persist."
    else:
        diagnosis = "RunPod is not fully ready; inspect checks and diagnostics."
    print(json.dumps(_redact_secrets({"workspace_root": str(workspace_root), "checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}), indent=2))
    return 0 if ok else 1


def _comfyui_lora_count(object_info: dict[str, Any]) -> int | None:
    loader = object_info.get("LoraLoader")
    if not isinstance(loader, dict):
        return None
    required = loader.get("input", {}).get("required") if isinstance(loader.get("input"), dict) else None
    lora_name = required.get("lora_name") if isinstance(required, dict) else None
    if isinstance(lora_name, list) and lora_name and isinstance(lora_name[0], list):
        return len(lora_name[0])
    return None


def _redact_url_userinfo(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    replacement = {"query": "", "fragment": ""}
    if "@" not in parsed.netloc:
        return urllib.parse.urlunparse(parsed._replace(**replacement))
    host = parsed.netloc.rsplit("@", 1)[1]
    return urllib.parse.urlunparse(parsed._replace(netloc=f"***@{host}", **replacement))


def cmd_doctor_comfyui(_: argparse.Namespace) -> int:
    try:
        workspace_root = _require_workspace()
        config = _workspace_config()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"comfyui: configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 1
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    endpoint = str(comfyui.get("endpoint") or "http://127.0.0.1:8188").rstrip("/")
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
    }
    diagnostics: dict[str, Any] = {
        "endpoint": _redact_url_userinfo(endpoint),
        "lora_dir": str(lora_dir) if lora_dir else None,
        "stage_dir": str(stage_dir) if stage_dir else None,
    }
    if parsed_endpoint.scheme not in ("http", "https"):
        diagnostics["object_info_error"] = f"unsupported comfyui.endpoint scheme: {parsed_endpoint.scheme or '(none)'}"
        diagnosis = "ComfyUI endpoint is not ready; comfyui.endpoint must start with http:// or https://."
        print(json.dumps(_redact_secrets({"workspace_root": str(workspace_root), "checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}), indent=2))
        return 1
    try:
        with urllib.request.urlopen(f"{endpoint}/object_info", timeout=5) as response:
            object_info = json.loads(response.read().decode("utf-8"))
        checks["endpoint_reachable"] = True
        checks["object_info"] = isinstance(object_info, dict)
        if isinstance(object_info, dict):
            diagnostics["lora_loader_count"] = _comfyui_lora_count(object_info)
    except Exception as exc:
        diagnostics["object_info_error"] = _redact_secret_text(str(exc))
    if stage_dir and not stage_dir.exists() and lora_dir and lora_dir.is_dir():
        diagnostics["stage_parent_writable"] = os.access(lora_dir, os.W_OK)
    if checks["endpoint_reachable"] and checks["object_info"]:
        diagnosis = "ComfyUI endpoint is reachable."
    else:
        diagnosis = "ComfyUI endpoint is not ready; start ComfyUI or check comfyui.endpoint in workspace.yaml."
    print(json.dumps(_redact_secrets({"workspace_root": str(workspace_root), "checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}), indent=2))
    return 0 if checks["endpoint_reachable"] and checks["object_info"] else 1


def cmd_doctor_secrets(_: argparse.Namespace) -> int:
    config = Path.home() / ".docker" / "config.json"
    registries: list[str] = []
    try:
        registries = sorted(json.loads(config.read_text(encoding="utf-8")).get("auths", {}).keys())
    except (OSError, json.JSONDecodeError):
        pass
    print(json.dumps({"secrets": _secret_state(), "docker_login_registries": registries}, indent=2))
    return 0


def cmd_doctor_workspace(_: argparse.Namespace) -> int:
    workspace = _workspace()
    subdirs = {name: (workspace / name).is_dir() for name in ("datasets", "runs", "workflows", "promptsets", "docker")}
    print(json.dumps({
        "workspace_root": str(workspace),
        "workspace_yaml": (workspace / "workspace.yaml").is_file(),
        "subdirs": subdirs,
    }, indent=2))
    return 0
