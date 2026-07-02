"""RunPod ComfyUI render orchestration."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from kura.executors import launch_runpod_session
from kura.notifications import notify as _notify
from kura.render import _safe_stage_name, launch_render
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.run_commands.common import _image_config, _safe_error
from kura.run_commands.plan import stop_run
from kura.run_commands.runpod_ssh import _free_local_port, _runpod_secret_env_payload, _runpod_ssh_details, _scp_to_runpod, _ssh_base, _start_runpod_session_lease_guard, _sync_runpod_remote_stdout, _wait_http_ready


def _render_runpod_config(config: dict[str, Any]) -> dict[str, Any]:
    source_runpod_config = config.get("runpod", {})
    runpod_config = dict(source_runpod_config) if isinstance(source_runpod_config, dict) else {}
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    comfy_runpod = comfyui.get("runpod") if isinstance(comfyui.get("runpod"), dict) else {}
    runpod_config.update(comfy_runpod)
    runpod_config.pop("template_id", None)
    backend_ports = runpod_config.get("backend_ports")
    if isinstance(backend_ports, dict) and isinstance(backend_ports.get("comfyui"), list):
        runpod_config["ports"] = backend_ports["comfyui"]
    else:
        runpod_config["ports"] = runpod_config.get("ports") if isinstance(runpod_config.get("ports"), list) else ["22/tcp"]
    return runpod_config


def _render_runpod_lora(workspace: Path, run_dir: Path, frozen: dict[str, Any]) -> tuple[Path | None, str | None]:
    if "lora" not in frozen.get("workflow_patches", {}) and not frozen.get("lora_insert"):
        return None, None
    checkpoint = frozen.get("inputs", {}).get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None, None
    path_value = checkpoint.get("path")
    if not isinstance(path_value, str) or not path_value:
        return None, None
    source = Path(path_value).expanduser()
    if not source.is_absolute():
        source = workspace / source
    source = source.resolve()
    if not source.is_file() or source.suffix != ".safetensors":
        raise ValueError("runpod ComfyUI render with LoRA requires checkpoint.path to point to a local .safetensors file")
    return source, "Kura_tmp/" + _safe_stage_name(run_dir.name, source)


def _start_runpod_comfyui(details: dict[str, Any], *, workspace: str, run_id: str, workflow_remote: str, registry_remote: str, lora_remote_name: str | None, lora_remote_path: str | None, max_lease_sec: int = 12 * 3600) -> None:
    secret_payload = _runpod_secret_env_payload(remote_notify=False)
    remote_secret_path = f"/tmp/kura-secrets/{run_id}.env"
    if secret_payload is not None:
        install_secret_script = f"""
set -euo pipefail
umask 077
mkdir -p /tmp/kura-secrets
cat > {shlex.quote(remote_secret_path)}
chmod 600 {shlex.quote(remote_secret_path)}
""".strip()
        installed = subprocess.run([*_ssh_base(details), install_secret_script], input=secret_payload, text=True, check=False)
        if installed.returncode:
            raise ValueError(f"ssh secret preparation failed with exit code {installed.returncode}")
    pod_id = details.get("pod_id")
    pod_id_value = pod_id if isinstance(pod_id, str) else ""
    lease_guard = ""
    if max_lease_sec > 0:
        lease_guard = f"""
(
  sleep {int(max_lease_sec)}
  echo "Kura render max lease expired after {int(max_lease_sec)} seconds; attempting to delete RunPod pod" >> "$KURA_LOG_PATH" 2>&1 || true
  if command -v runpodctl >/dev/null 2>&1 && [ -n {shlex.quote(pod_id_value)} ]; then
    runpodctl pod delete {shlex.quote(pod_id_value)} >> "$KURA_LOG_PATH" 2>&1 || true
  fi
) </dev/null >/dev/null 2>&1 &
""".strip()
    lora_line = ""
    if lora_remote_name and lora_remote_path:
        lora_line = f"printf '%s\\n' {shlex.quote('Kura LoRA staged: ' + lora_remote_name + ' -> ' + lora_remote_path)} >> \"$KURA_LOG_PATH\""
    script = f"""
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
export KURA_WORKSPACE={shlex.quote(workspace)}
export KURA_RUN_ID={shlex.quote(run_id)}
export KURA_LOG_PATH={shlex.quote(workspace.rstrip('/') + '/runs/' + run_id + '/logs/stdout.log')}
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
touch "$KURA_LOG_PATH"
secret_file={shlex.quote(remote_secret_path)}
cleanup() {{
  rm -f "$secret_file"
}}
trap cleanup EXIT
if [ -f "$secret_file" ]; then
  . "$secret_file"
fi
{lease_guard}
python /opt/kura_comfy_prepare.py {shlex.quote(workflow_remote)} --registry-json {shlex.quote(registry_remote)} --comfyui-root /opt/ComfyUI >> "$KURA_LOG_PATH" 2>&1
{lora_line}
cd /opt/ComfyUI
nohup python main.py --listen 127.0.0.1 --port 8188 >> "$KURA_LOG_PATH" 2>&1 &
""".strip()
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False)
    if result.returncode:
        detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "remote ComfyUI start failed")
        raise ValueError(f"remote ComfyUI start failed with exit code {result.returncode}: {detail}")


def launch_render_runpod(run_id: str, *, dry_run: bool, image: str | None = None, notify_channels: Any = None) -> int:
    workspace = _workspace()
    run_dir = _run_path(run_id)
    launched = False
    ssh_details: dict[str, Any] | None = None
    remote_workspace = "/workspace"
    try:
        frozen = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        if frozen.get("type") != "render":
            raise ValueError("run is not a render run")
        if frozen.get("generator", {}).get("name") != "comfyui":
            raise ValueError("runpod render currently requires generator.name=comfyui")
        current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if current_status.get("state") != "compiled":
            raise ValueError("render must be compiled before launch")
        config = _workspace_config()
        image_config = _image_config("comfyui")
        runpod_config = _render_runpod_config(config)
        default_image = runpod_config.get("default_image")
        remote_image = image_config["remote"]
        if isinstance(default_image, dict) and isinstance(default_image.get("comfyui"), str):
            remote_image = default_image["comfyui"]
        if image:
            remote_image = image
        compute = frozen.get("compute") if isinstance(frozen.get("compute"), dict) else {}
        gpu_override = compute.get("gpu") if isinstance(compute, dict) else None
        if isinstance(gpu_override, str) and gpu_override and gpu_override.lower() not in {"true", "false", "gpu", "cpu"}:
            runpod_config["gpu_type_ids"] = [gpu_override]
            runpod_config["gpu_type_priority"] = "custom"
        model_specs = frozen.get("comfyui_models")
        model_registry = frozen.get("comfyui_model_registry")
        if not isinstance(model_specs, list) or not isinstance(model_registry, dict):
            raise ValueError("runpod render requires a manifest compiled for executor.name=runpod; set executor.name=runpod in run.yaml and recompile before launching on RunPod")
        lora_source, lora_name = _render_runpod_lora(workspace, run_dir, frozen)
        plan = {"executor": "runpod", "image": remote_image, "models": model_specs, "lora_name": lora_name, "ports": runpod_config.get("ports"), "gpu_type_ids": runpod_config.get("gpu_type_ids")}
        if dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        launch_runpod_session(run_dir=run_dir, image=remote_image, config=runpod_config, purpose="comfyui-render", dry_run=False)
        launched = True
        details = _runpod_ssh_details(run_dir, timeout_sec=300, interval_sec=5)
        ssh_details = details
        remote_workspace = str(runpod_config.get("workspace_path") or "/workspace")
        remote_run_dir = f"{remote_workspace.rstrip('/')}/runs/{run_dir.name}"
        _start_runpod_session_lease_guard(details, workspace=remote_workspace, run_id=run_dir.name)
        prepared = subprocess.run([*_ssh_base(details), f"mkdir -p {shlex.quote(remote_run_dir + '/resolved')} /opt/ComfyUI/models/loras/Kura_tmp"], check=False)
        if prepared.returncode:
            raise ValueError(f"ssh workspace preparation failed with exit code {prepared.returncode}")
        workflow_path = run_dir / "resolved" / "workflow_used.json"
        remote_workflow = f"{remote_run_dir}/resolved/workflow_used.json"
        _scp_to_runpod(details, workflow_path, remote_workflow)
        registry_path = run_dir / "resolved" / "comfyui_model_registry.json"
        remote_registry = f"{remote_run_dir}/resolved/comfyui_model_registry.json"
        _scp_to_runpod(details, registry_path, remote_registry)
        lora_remote_path = None
        if lora_source and lora_name:
            lora_remote_path = "/opt/ComfyUI/models/loras/" + lora_name
            _scp_to_runpod(details, lora_source, lora_remote_path)
        _start_runpod_comfyui(details, workspace=remote_workspace, run_id=run_dir.name, workflow_remote=remote_workflow, registry_remote=remote_registry, lora_remote_name=lora_name, lora_remote_path=lora_remote_path, max_lease_sec=0)
        local_port = _free_local_port()
        tunnel = subprocess.Popen([
            "ssh",
            "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:8188",
            "-i", str(details["key"]),
            "-p", str(details["port"]),
            f"root@{details['ip']}",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            endpoint = f"http://127.0.0.1:{local_port}"
            ready_timeout = int(frozen.get("render", {}).get("timeout_sec", 600) or 600)
            _wait_http_ready(endpoint, timeout_sec=max(ready_timeout, 180))
            code = launch_render(workspace, run_dir, endpoint_override=endpoint, lora_name_override=lora_name, executor_name="runpod", manage_lora_stage=False)
            state_word = "completed" if code == 0 else "failed"
            _notify(notify_channels, subject=f"Kura render {state_word}: {run_id}", body=f"RunPod render {run_id} {state_word} with exit code {code}.", priority="3")
            return code
        finally:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
    except KeyboardInterrupt:
        if ssh_details is not None:
            try:
                _sync_runpod_remote_stdout(run_dir, ssh_details, workspace=remote_workspace, run_id=run_dir.name, timeout_sec=15)
            except BaseException:
                pass
        print("runpod render interrupted; stopping pod now", file=sys.stderr)
        return 130
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError, subprocess.TimeoutExpired) as exc:
        if ssh_details is not None:
            _sync_runpod_remote_stdout(run_dir, ssh_details, workspace=remote_workspace, run_id=run_dir.name, timeout_sec=15)
        message = _safe_error(exc)
        print(f"cannot launch runpod render: {message}", file=sys.stderr)
        if not dry_run:
            _notify(notify_channels, subject=f"Kura render failed: {run_id}", body=f"RunPod render {run_id} failed before completion:\n{message}", priority="3")
        return 1
    finally:
        if launched:
            if ssh_details is not None:
                try:
                    _sync_runpod_remote_stdout(run_dir, ssh_details, workspace=remote_workspace, run_id=run_dir.name, timeout_sec=15)
                except BaseException:
                    pass
            try:
                stop_run(run_id)
            except Exception as exc:
                print(f"warning: could not stop RunPod render pod automatically: {_safe_error(exc)}", file=sys.stderr)
