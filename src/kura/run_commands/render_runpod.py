"""RunPod ComfyUI render orchestration."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from kura.executors import launch_runpod_session, runpod_gpu_availability
from kura.executors.common import DEFAULT_MAX_LEASE_SEC, check_stop
from kura.executors.runpod import confirm_runpod_billing, stop_runpod, unresolved_create_intents, unstopped_recovered_pod
from kura.fsio import file_lock
from kura.notifications import notify as _notify
from kura.render import _safe_stage_name, digest, image_patch_names, launch_render, load_resolved_cases
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.images import image_cuda_version, runpod_min_cuda_version
from kura.run_commands.common import _effective_image, _safe_error, requested_gpu_types
from kura.run_commands.runpod_ssh import record_pod_lease_deadline
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


def _format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "disabled"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _render_runpod_billing_plan(runpod_config: dict[str, Any], *, max_lease_sec: int, image: str) -> dict[str, Any]:
    gpu_type_ids = runpod_config.get("gpu_type_ids")
    if not isinstance(gpu_type_ids, list) or not all(isinstance(item, str) and item for item in gpu_type_ids):
        raise ValueError("runpod.gpu_type_ids must be configured before showing a RunPod render plan")
    min_cuda_version = runpod_min_cuda_version(image)
    measurement = runpod_gpu_availability(runpod_config, gpu_type_ids, min_cuda_version=min_cuda_version)
    return {
        "host_cuda": f"{min_cuda_version} or newer" + ("" if image_cuda_version(image) else " (Kura does not know this image's CUDA version)"),
        "gpu_count": measurement.get("gpu_count", runpod_config.get("gpu_count", 1)),
        "gpu_candidates": measurement.get("candidates", []),
        "price_status": measurement.get("status", "unavailable"),
        "price_checked_at": measurement.get("checked_at"),
        "price_reason": measurement.get("reason"),
        "maximum_lease": _format_duration(max_lease_sec),
        "maximum_lease_sec": max_lease_sec,
    }


def _render_runpod_loras(workspace: Path, run_dir: Path, frozen: dict[str, Any]) -> list[dict[str, Any]]:
    if "lora" not in frozen.get("workflow_patches", {}) and not frozen.get("lora_insert"):
        return []
    cases_path = run_dir / "resolved" / "cases.jsonl"
    if cases_path.is_file():
        cases = load_resolved_cases(cases_path)
    else:
        checkpoint = frozen.get("inputs", {}).get("checkpoint")
        cases = [{"id": "legacy", "checkpoint": checkpoint}] if isinstance(checkpoint, dict) else []
    items: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for case in cases:
        checkpoint = case.get("checkpoint")
        if not isinstance(checkpoint, dict):
            continue
        path_value = checkpoint.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        source = Path(path_value).expanduser()
        if not source.is_absolute():
            source = workspace / source
        source = source.resolve()
        if not source.is_file() or source.suffix != ".safetensors":
            raise ValueError(f"runpod ComfyUI render case {case['id']!r} requires checkpoint.path to point to a local .safetensors file")
        identity = (checkpoint.get("id"), str(source))
        if identity in seen:
            continue
        seen.add(identity)
        items.append({
            "id": checkpoint.get("id"),
            "source": source,
            "name": "Kura_tmp/" + _safe_stage_name(run_dir.name, source),
        })
    return items


def _render_runpod_images(run_dir: Path, frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve and verify compile-frozen image inputs before any Pod is created."""
    names = image_patch_names(frozen.get("workflow_patches", {}))
    if not names:
        return []
    cases_path = run_dir / "resolved" / "cases.jsonl"
    if not cases_path.is_file():
        raise ValueError("runpod image render requires resolved/cases.jsonl; compile the render again")
    cases = load_resolved_cases(cases_path)
    raw_records = frozen.get("promptset_images")
    if not isinstance(raw_records, list):
        raise ValueError("runpod image render manifest has no frozen image records; compile the render again")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError("runpod image render manifest contains an invalid frozen image record")
        patch_name = record.get("patch")
        resolved = record.get("resolved")
        if not isinstance(patch_name, str) or not isinstance(resolved, str):
            raise ValueError("runpod image render manifest contains an invalid frozen image record")
        key = (patch_name, resolved)
        if key in records:
            raise ValueError(f"runpod image render manifest contains a duplicate frozen image record: {resolved}")
        records[key] = record

    frozen_root = (run_dir / "resolved" / "images").resolve()
    items: dict[str, dict[str, Any]] = {}
    expected_records: set[tuple[str, str]] = set()
    for case in cases:
        values = case.get("values") if isinstance(case.get("values"), dict) else {}
        for name in names:
            resolved = values.get(name)
            if not isinstance(resolved, str) or not resolved:
                raise ValueError(f"runpod render case {case.get('id')!r} has no frozen image for workflow_patches.{name}")
            expected_records.add((name, resolved))
            relative = Path(resolved)
            expected_prefix = Path("resolved") / "images" / name
            if relative.is_absolute() or ".." in relative.parts or "\\" in resolved or relative.parent != expected_prefix:
                raise ValueError(f"runpod render case {case.get('id')!r} has an unsafe frozen image path: {resolved}")
            source = (run_dir / relative).resolve()
            if source.parent != (frozen_root / name).resolve() or not source.is_file():
                raise ValueError(f"runpod render frozen image is missing or not a regular file: {resolved}")
            record = records.get((name, resolved))
            expected_digest = record.get("digest") if isinstance(record, dict) else None
            if not isinstance(expected_digest, str) or digest(source) != expected_digest:
                raise ValueError(f"runpod render frozen image does not match its manifest digest: {resolved}")
            items.setdefault(resolved, {
                "patch": name,
                "frozen": resolved,
                "source": source,
                "name": "Kura_tmp/" + _safe_stage_name(run_dir.name, source),
                "bytes": source.stat().st_size,
            })
    if set(records) != expected_records:
        missing = sorted(expected_records - set(records))
        unexpected = sorted(set(records) - expected_records)
        raise ValueError(
            "runpod image render manifest does not match the frozen case queue; "
            f"missing={missing} unexpected={unexpected}"
        )
    return list(items.values())


def _start_runpod_comfyui(details: dict[str, Any], *, workspace: str, run_id: str, workflow_remote: str, registry_remote: str, lora_remote_name: str | None, lora_remote_path: str | None, lora_remote_files: list[dict[str, str]] | None = None) -> None:
    """Install the secrets, prepare the models, and start ComfyUI; the Pod's own lease guard bounds it."""
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
        try:
            installed = subprocess.run([*_ssh_base(details), install_secret_script], input=secret_payload, text=True, capture_output=True, check=False, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"ssh secret preparation timed out after {exc.timeout} seconds") from exc
        if installed.returncode:
            detail = _safe_error(installed.stderr.strip() or installed.stdout.strip() or "ssh secret preparation failed")
            raise ValueError(f"ssh secret preparation failed with exit code {installed.returncode}: {detail}")
    staged_loras = lora_remote_files or (
        [{"name": lora_remote_name, "path": lora_remote_path}]
        if lora_remote_name and lora_remote_path else []
    )
    lora_line = "\n".join(
        f"printf '%s\\n' {shlex.quote('Kura LoRA staged: ' + item['name'] + ' -> ' + item['path'])} >> \"$KURA_LOG_PATH\""
        for item in staged_loras
    )
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
python /opt/kura_comfy_prepare.py {shlex.quote(workflow_remote)} --registry-json {shlex.quote(registry_remote)} --comfyui-root /opt/ComfyUI >> "$KURA_LOG_PATH" 2>&1
{lora_line}
cd /opt/ComfyUI
nohup python main.py --listen 127.0.0.1 --port 8188 >> "$KURA_LOG_PATH" 2>&1 &
""".strip()
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=600)
    if result.returncode:
        detail = _safe_error(result.stderr.strip() or result.stdout.strip() or "remote ComfyUI start failed")
        raise ValueError(f"remote ComfyUI start failed with exit code {result.returncode}: {detail}")


def launch_render_runpod(
    run_id: str,
    *,
    dry_run: bool,
    image: str | None = None,
    notify_channels: Any = None,
    yes: bool = False,
    max_lease_sec: int = DEFAULT_MAX_LEASE_SEC,
    controlled_by: dict[str, Any] | None = None,
    runpod_config_override: dict[str, Any] | None = None,
    check_only: bool = False,
    prepared: dict[str, Any] | None = None,
) -> int:
    """Render on a disposable RunPod Pod and delete it afterwards.

    `check_only` runs every check, shows the cost, and takes the billing
    confirmation without creating anything; the job runner launches later
    with the settings returned in `prepared`.
    """
    workspace = _workspace()
    run_dir = _run_path(run_id)
    launched = False
    runpod_config: dict[str, Any] = {}
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
        runpod_config = _render_runpod_config(config)
        remote_image = _effective_image("comfyui")["reference"]
        if image:
            remote_image = image
        compute = frozen.get("compute") if isinstance(frozen.get("compute"), dict) else {}
        requested = requested_gpu_types(compute)
        if requested is not None:
            runpod_config["gpu_type_ids"] = requested
            runpod_config["gpu_type_priority"] = "custom"
        model_specs = frozen.get("comfyui_models")
        model_registry = frozen.get("comfyui_model_registry")
        if not isinstance(model_specs, list) or not isinstance(model_registry, dict):
            raise ValueError("runpod render requires a manifest compiled for executor.name=runpod; set executor.name=runpod in run.yaml and recompile before launching on RunPod")
        loras = _render_runpod_loras(workspace, run_dir, frozen)
        images = _render_runpod_images(run_dir, frozen)
        multiple_loras = len(loras) > 1
        lora_name = loras[0]["name"] if len(loras) == 1 else None
        plan = {
            "executor": "runpod",
            "image": remote_image,
            "models": model_specs,
            "lora_name": lora_name,
            "ports": runpod_config.get("ports"),
            "gpu_type_ids": runpod_config.get("gpu_type_ids"),
        }
        if multiple_loras:
            plan["loras"] = [{"id": item["id"], "name": item["name"]} for item in loras]
        if images:
            plan["input_images"] = [
                {"frozen": item["frozen"], "name": item["name"], "bytes": item["bytes"]}
                for item in images
            ]
            plan["input_image_bytes"] = sum(int(item["bytes"]) for item in images)
        if runpod_config_override is not None:
            # A runner launch uses the settings the user confirmed, not workspace.yaml as it is now.
            runpod_config = dict(runpod_config_override)
        if dry_run:
            plan["billing"] = _render_runpod_billing_plan(runpod_config, max_lease_sec=max_lease_sec, image=remote_image)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if unresolved_create_intents(run_dir):
            raise ValueError(f"an earlier launch stopped before recording whether its Pod was created; run `kura run reconcile {run_id}` first")
        if (recovered := unstopped_recovered_pod(run_dir)) is not None:
            raise ValueError(f"Pod {recovered} from an earlier launch may still be billing; run `kura run stop {run_id}` before launching again")
        if check_only:
            confirm_runpod_billing(runpod_config, remote_image, yes=yes, max_lease_sec=max_lease_sec)
            if prepared is not None:
                prepared.update({"runpod_config": runpod_config, "remote_image": remote_image})
            return 0
        with file_lock(run_dir / ".locks" / "runpod-launch.lock", blocking=False):
            launch_runpod_session(
                run_dir=run_dir,
                image=remote_image,
                config=runpod_config,
                purpose="comfyui-render",
                dry_run=False,
                yes=yes,
                max_lease_sec=max_lease_sec,
                controlled_by=controlled_by,
            )
        launched = True
        details = _runpod_ssh_details(run_dir, timeout_sec=300, interval_sec=5)
        ssh_details = details
        remote_workspace = str(runpod_config.get("workspace_path") or "/workspace")
        remote_run_dir = f"{remote_workspace.rstrip('/')}/runs/{run_dir.name}"
        # The Pod armed its lease when it started; this second guard never moves that deadline
        # and stays for Pods whose start script predates it.
        _start_runpod_session_lease_guard(details, workspace=remote_workspace, run_id=run_dir.name, max_lease_sec=max_lease_sec)
        record_pod_lease_deadline(run_dir, details)
        check_stop()
        workspace_ready = subprocess.run([*_ssh_base(details), f"mkdir -p {shlex.quote(remote_run_dir + '/resolved')} /opt/ComfyUI/models/loras/Kura_tmp /opt/ComfyUI/input/Kura_tmp"], check=False, timeout=600)
        if workspace_ready.returncode:
            raise ValueError(f"ssh workspace preparation failed with exit code {workspace_ready.returncode}")
        workflow_path = run_dir / "resolved" / "workflow_used.json"
        remote_workflow = f"{remote_run_dir}/resolved/workflow_used.json"
        _scp_to_runpod(details, workflow_path, remote_workflow)
        registry_path = run_dir / "resolved" / "comfyui_model_registry.json"
        remote_registry = f"{remote_run_dir}/resolved/comfyui_model_registry.json"
        _scp_to_runpod(details, registry_path, remote_registry)
        remote_loras: list[dict[str, str]] = []
        for item in loras:
            remote_path = "/opt/ComfyUI/models/loras/" + item["name"]
            check_stop()
            _scp_to_runpod(details, item["source"], remote_path)
            remote_loras.append({"name": item["name"], "path": remote_path})
        for item in images:
            check_stop()
            _scp_to_runpod(details, item["source"], "/opt/ComfyUI/input/" + item["name"])
        check_stop()
        lora_remote_path = remote_loras[0]["path"] if len(remote_loras) == 1 else None
        _start_runpod_comfyui(details, workspace=remote_workspace, run_id=run_dir.name, workflow_remote=remote_workflow, registry_remote=remote_registry, lora_remote_name=lora_name, lora_remote_path=lora_remote_path, lora_remote_files=remote_loras)
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
            lora_name_overrides = {
                str(item["id"]): str(item["name"])
                for item in loras
                if isinstance(item.get("id"), str)
            }
            code = launch_render(
                workspace,
                run_dir,
                endpoint_override=endpoint,
                lora_name_override=lora_name,
                lora_name_overrides=lora_name_overrides,
                executor_name="runpod",
                manage_lora_stage=False,
                image_name_overrides={str(item["frozen"]): str(item["name"]) for item in images},
                controlled_by=controlled_by,
            )
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
        _settle_unfinished(run_dir, "interrupted")
        print("runpod render interrupted; stopping pod now", file=sys.stderr)
        return 130
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError, subprocess.TimeoutExpired) as exc:
        if ssh_details is not None:
            try:
                _sync_runpod_remote_stdout(run_dir, ssh_details, workspace=remote_workspace, run_id=run_dir.name, timeout_sec=15)
            except (OSError, ValueError, subprocess.TimeoutExpired) as sync_exc:
                print(f"warning: could not sync RunPod render logs: {_safe_error(sync_exc)}", file=sys.stderr)
        message = _safe_error(exc)
        if launched:
            _settle_unfinished(run_dir, "failed", error=message)
        print(f"cannot launch runpod render: {message}", file=sys.stderr)
        if not dry_run and not check_only:
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
                # Deleted here, not through `kura run stop`, which would hand the stop to this very follower.
                stop_runpod(run_dir, runpod_config)
            except Exception as exc:
                print(f"warning: could not stop RunPod render pod automatically: {_safe_error(exc)}", file=sys.stderr)


def _settle_unfinished(run_dir: Path, state: str, *, error: str | None = None) -> None:
    """A render that ended before its cases finished is never left looking like it still runs."""
    from kura.executors.common import end_run

    end_run(run_dir, state, reason="the RunPod render ended before its cases finished", error=error, unless_finished=True)

