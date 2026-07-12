"""RunPod SSH, SCP, upload, pull, and download helpers."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from kura.executors import _materialize_stdout_progress, _redact_secret_text, _redact_secrets
from kura.fsio import atomic_write_json
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import run_path as _run_path
from kura.workspace import workspace_config as _workspace_config
from kura.run_envelope import common_recipe
from kura.executors.common import append_run_event
from kura.run_commands.common import _safe_error
from kura.run_commands.plan import _configured_download_min_free_bytes, _ensure_free_bytes


RUNPOD_TRANSFER_TIMEOUT_SEC = 600


@contextlib.contextmanager
def _run_operation_lock(run_dir: Path, name: str, *, blocking: bool = True):
    """Serialize controller-side mutations without creating another truth store."""

    lock_dir = run_dir / ".locks"
    lock_dir.mkdir(exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"
    with lock_path.open("a+b") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            raise ValueError(f"another {name} operation is already active for run {run_dir.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_bounded(command: list[str], *, context: str, timeout: int = RUNPOD_TRANSFER_TIMEOUT_SEC, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(command, check=False, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{context} timed out after {exc.timeout} seconds") from exc


def _latest_runpod_transfer(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no RunPod realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    transfer = realization.get("transfer")
    if realization.get("executor") != "runpod" or not isinstance(transfer, dict):
        raise ValueError("latest realization has no RunPod upload transfer")
    return transfer


def _latest_runpod_stage(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stage_ref = status.get("last_stage")
    if not isinstance(stage_ref, str):
        raise ValueError("run has no RunPod upload stage")
    stage = json.loads((run_dir / stage_ref).read_text(encoding="utf-8"))
    if stage.get("storage_mode") != "upload":
        raise ValueError("latest RunPod stage is not an upload bundle")
    return stage


def cmd_run_upload(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        transfer = _latest_runpod_transfer(run_dir)
        archive = transfer.get("archive")
        upload_code = transfer.get("upload_code")
        if not isinstance(archive, str) or not isinstance(upload_code, str):
            raise ValueError("latest realization has no upload archive/code")
        archive_path = run_dir / archive
        if not archive_path.is_file():
            raise ValueError(f"upload archive is missing: {archive_path}")
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before uploading")
        return subprocess.run(["runpodctl", "send", str(archive_path), "--code", upload_code], check=False).returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot upload run bundle: {_safe_error(exc)}", file=sys.stderr)
        return 1


def download_run(run_id: str, *, force: bool = False) -> int:
    try:
        run_dir = _run_path(run_id)
        with _run_operation_lock(run_dir, "download", blocking=False):
            return _download_run_unlocked(run_id, force=force)
    except (OSError, ValueError) as exc:
        print(f"cannot download run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _download_run_unlocked(run_id: str, *, force: bool = False) -> int:
    try:
        run_dir = _run_path(run_id)
        destination = run_dir / "downloads"
        downloaded_run = destination / run_id

        def materialize_primary_outputs(output_dir: Path) -> list[str]:
            primary = run_dir / "outputs"
            outputs: list[str] = []
            if not output_dir.exists():
                return outputs
            temporary = run_dir / f".outputs.tmp-{secrets.token_hex(4)}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True, exist_ok=True)
            for source in sorted(path for path in output_dir.rglob("*") if path.is_file()):
                relative = source.relative_to(output_dir)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                outputs.append(str((primary / relative).relative_to(run_dir)))
            if primary.exists():
                shutil.rmtree(primary)
            temporary.rename(primary)
            return outputs

        def materialize_downloaded_status() -> bool:
            exits = sorted((downloaded_run / "realizations").glob("remote-exit-*.json"))
            if not exits:
                return False
            remote_exit = json.loads(exits[-1].read_text(encoding="utf-8"))
            exit_code = remote_exit.get("exit_code")
            if not isinstance(exit_code, int):
                return False
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            output_dir = downloaded_run / "outputs"
            outputs = materialize_primary_outputs(output_dir)
            status.update({"state": "completed" if exit_code == 0 else "failed", "exit_code": exit_code, "ended": remote_exit.get("timestamp"), "outputs": outputs, "downloaded_run": str(downloaded_run.relative_to(run_dir)), "remote_exit": str(exits[-1].relative_to(run_dir)), "remote_state": "completed" if exit_code == 0 else "failed", "remote_exit_code": exit_code, "remote_ended": remote_exit.get("timestamp"), "recovery_required": False})
            if exit_code == 0:
                try:
                    manifest = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
                    steps = common_recipe(manifest).get("steps")
                    if isinstance(steps, int) and steps > 0:
                        status["last_step"] = steps
                        status["total_steps"] = steps
                except (OSError, ValueError, yaml.YAMLError):
                    pass
            atomic_write_json(run_dir / "status.json", status)
            return True

        if downloaded_run.exists() and not force:
            if materialize_downloaded_status():
                print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
                return 0
            raise ValueError("downloaded run snapshot is missing remote-exit; use --force to retry or inspect the Pod before stopping it")
        if downloaded_run.exists() and force:
            shutil.rmtree(downloaded_run)
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before downloading")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        pod_id = status.get("pod_id")
        if not isinstance(pod_id, str):
            raise ValueError("run has no RunPod pod ID")
        config = _workspace_config()
        min_download_free = _configured_download_min_free_bytes(config)
        pod = _run_bounded(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, context="runpodctl pod get")
        if pod.returncode:
            raise ValueError(_redact_secret_text(pod.stderr.strip() or pod.stdout.strip() or "runpodctl pod get failed"))
        details = json.loads(pod.stdout)
        ssh = details.get("ssh", {})
        ip, port = ssh.get("ip"), ssh.get("port")
        key = ssh.get("ssh_key", {}).get("path")
        if not isinstance(ip, str) or not isinstance(port, int) or not isinstance(key, str):
            raise ValueError("pod SSH is not ready")
        destination.mkdir(exist_ok=True)
        workspace = _runpod_workspace_for_run(run_dir)
        remote_run_dir = f"{workspace.rstrip('/')}/runs/{run_id}"
        remote_size = _remote_path_size({"ip": ip, "port": port, "key": key}, remote_run_dir)
        if isinstance(remote_size, int) and remote_size > 0:
            _ensure_free_bytes(destination, max(min_download_free, remote_size * 2 + 5 * 1024**3), context="RunPod download")
        else:
            _ensure_free_bytes(destination, min_download_free, context="RunPod download")
        remote_archive = f"/tmp/kura-download-{run_id}.tar.gz"
        remote_script = (
            f"tar -C /workspace/runs "
            f"--exclude {shlex.quote(run_id + '/cache')} "
            f"--exclude {shlex.quote(run_id + '/transfer')} "
            f"-czf {shlex.quote(remote_archive)} {shlex.quote(run_id)}"
        )
        packed = _run_bounded([*_ssh_base({"ip": ip, "port": port, "key": key}), remote_script], context="remote archive packing")
        if packed.returncode:
            return packed.returncode
        local_archive = destination / f"kura-download-{run_id}.tar.gz"
        command = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(port),
            "-i", key,
            f"root@{ip}:{remote_archive}",
            str(local_archive),
        ]
        result = _run_bounded(command, context="scp download")
        _run_bounded([*_ssh_base({"ip": ip, "port": port, "key": key}), f"rm -f {shlex.quote(remote_archive)}"], context="remote archive cleanup")
        if result.returncode:
            return result.returncode
        extracted = _run_bounded(["tar", "--warning=no-timestamp", "-xzf", str(local_archive), "-C", str(destination)], context="download archive extraction")
        local_archive.unlink(missing_ok=True)
        if extracted.returncode:
            return extracted.returncode
        if not materialize_downloaded_status():
            raise ValueError("downloaded run snapshot is missing remote-exit; remote completion is not confirmed")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"cannot download run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def cmd_run_download(args: argparse.Namespace) -> int:
    return download_run(args.run_id, force=args.force)


def _runpod_workspace_for_run(run_dir: Path) -> str:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if isinstance(realization_ref, str):
        try:
            realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
            workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE")
            if isinstance(workspace, str) and workspace:
                return workspace
        except (OSError, json.JSONDecodeError):
            pass
    return "/workspace"


def _checkpoint_step(name: str) -> int | None:
    matches = re.findall(r"(?:step|_)(\d{4,})(?=\.safetensors$|[-_.])", name)
    if not matches:
        return None
    return int(matches[-1])


def _select_remote_outputs(items: list[dict[str, Any]], *, step: int | None = None, since_step: int | None = None, all_outputs: bool = False) -> list[dict[str, Any]]:
    candidates = [item for item in items if isinstance(item.get("name"), str)]
    for item in candidates:
        if not isinstance(item.get("step"), int):
            item["step"] = _checkpoint_step(str(item["name"]))
    if step is not None:
        return [item for item in candidates if item.get("step") == step]
    if since_step is not None:
        return [item for item in candidates if isinstance(item.get("step"), int) and item["step"] >= since_step]
    if all_outputs:
        return candidates
    stepped = [item for item in candidates if isinstance(item.get("step"), int)]
    if stepped:
        latest = max(int(item["step"]) for item in stepped)
        return [item for item in stepped if item.get("step") == latest]
    return candidates[-1:] if candidates else []


def _same_remote_output_version(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    """Return true only when a remote file did not change across a transfer."""

    if after is None:
        return False
    return all(before.get(key) == after.get(key) for key in ("path", "size", "mtime_ns"))


def _runpod_remote_outputs(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> list[dict[str, Any]]:
    remote_outputs = f"{workspace.rstrip('/')}/runs/{run_id}/outputs"
    script = f"""
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
python - <<'PY'
import glob
import json
import os
import re

directory = {remote_outputs!r}
items = []
for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
    name = os.path.basename(path)
    stat = os.stat(path)
    matches = re.findall(r"(?:step|_)(\\d{{4,}})(?=\\.safetensors$|[-_.])", name)
    step = int(matches[-1]) if matches else None
    items.append({{"path": path, "name": name, "step": step, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})
print(json.dumps(items))
PY
""".strip()
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "remote output listing failed"))
    data = json.loads(result.stdout or "[]")
    if not isinstance(data, list):
        raise ValueError("remote output listing did not return a list")
    return [item for item in data if isinstance(item, dict)]


def cmd_run_pull(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before pulling outputs")
        workspace = _runpod_workspace_for_run(run_dir)
        details = _runpod_ssh_details(run_dir, timeout_sec=args.ssh_timeout, interval_sec=2)
        items = _runpod_remote_outputs(details, workspace=workspace, run_id=args.run_id)
        selected = _select_remote_outputs(items, step=args.step, since_step=args.since_step, all_outputs=args.all)
        if not selected:
            raise ValueError("no matching remote .safetensors outputs found")
        config = _workspace_config()
        min_download_free = _configured_download_min_free_bytes(config)
        destination = run_dir / "pulled" / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        selected_size = sum(item.get("size") for item in selected if isinstance(item.get("size"), int))
        _ensure_free_bytes(destination, max(min(10 * 1024**3, min_download_free), selected_size + 5 * 1024**3), context="RunPod output pull")
        pulled: list[dict[str, Any]] = []
        for item in selected:
            name = item.get("name")
            remote_path = item.get("path")
            size = item.get("size")
            if not isinstance(name, str) or not isinstance(remote_path, str):
                continue
            local_path = destination / name
            if local_path.exists() and isinstance(size, int) and local_path.stat().st_size == size and not args.force:
                pulled.append({"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": size, "skipped": True})
                continue
            partial_path = local_path.with_name(f".{local_path.name}.partial")
            partial_path.unlink(missing_ok=True)
            try:
                result = _run_bounded([
                    "scp",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-P", str(details["port"]),
                    "-i", str(details["key"]),
                    f"root@{details['ip']}:{remote_path}",
                    str(partial_path),
                ], context="scp output pull")
                if result.returncode:
                    return result.returncode
                refreshed = _runpod_remote_outputs(details, workspace=workspace, run_id=args.run_id)
                after = next((candidate for candidate in refreshed if candidate.get("path") == remote_path), None)
                if not _same_remote_output_version(item, after) or not isinstance(size, int) or partial_path.stat().st_size != size:
                    raise ValueError(f"remote checkpoint changed while it was being copied: {name}; wait for the save to finish and retry")
                os.replace(partial_path, local_path)
            finally:
                partial_path.unlink(missing_ok=True)
            pulled.append({"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": local_path.stat().st_size, "skipped": False})
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["pulled_outputs"] = pulled
        status["pulled_outputs_synced_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(status_path, _redact_secrets(status))
        append_run_event(run_dir, {"event": "run_outputs_pulled", "timestamp": datetime.now().astimezone().isoformat(), "count": len(pulled), "outputs": pulled})
        print(json.dumps({"run_id": args.run_id, "destination": str(destination), "pulled": pulled}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"cannot pull run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _runpod_upload_process(run_dir: Path, code_seed: str, timeout_sec: int) -> tuple[subprocess.Popen[str], str]:
    stage = _latest_runpod_stage(run_dir)
    archive = stage.get("archive")
    if not isinstance(archive, str):
        raise ValueError("latest RunPod stage has no upload archive")
    archive_path = run_dir / archive
    if not archive_path.is_file():
        raise ValueError(f"upload archive is missing: {archive_path}")
    if not shutil.which("runpodctl"):
        raise ValueError("runpodctl is not installed locally")
    process = subprocess.Popen(["runpodctl", "send", str(archive_path), "--code", code_seed], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(min(max(timeout_sec, 0), 1))
    if process.poll() is not None:
        raise ValueError(f"runpodctl send exited early with exit code {process.returncode}")
    return process, code_seed


def _wait_process(process: subprocess.Popen[str], timeout_sec: int) -> None:
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        process.terminate()
        raise ValueError("runpodctl send did not complete before timeout") from exc
    if process.returncode:
        raise ValueError(f"runpodctl send failed with exit code {process.returncode}")


def _runpod_ssh_details(run_dir: Path, *, timeout_sec: int, interval_sec: int = 10) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    pod_id = status.get("pod_id")
    if not isinstance(pod_id, str):
        raise ValueError("run has no RunPod pod ID")
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, check=False, timeout=max(interval_sec * 3, 1))
        except subprocess.TimeoutExpired:
            last_error = "runpodctl pod get timed out"
            time.sleep(interval_sec)
            continue
        if result.returncode:
            last_error = _redact_secret_text(result.stderr.strip() or result.stdout.strip())
        else:
            try:
                pod = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                last_error = _safe_error(exc)
            else:
                ssh = pod.get("ssh") if isinstance(pod.get("ssh"), dict) else {}
                ip, port = ssh.get("ip"), ssh.get("port")
                key = ssh.get("ssh_key", {}).get("path") if isinstance(ssh.get("ssh_key"), dict) else None
                if isinstance(ip, str) and isinstance(port, int) and isinstance(key, str):
                    return {"pod_id": pod_id, "ip": ip, "port": port, "key": key}
                last_error = str(ssh.get("error") or "pod SSH is not ready")
        time.sleep(interval_sec)
    raise ValueError(f"pod SSH did not become ready before timeout: {last_error}")


def _ssh_base(details: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        "-i", str(details["key"]),
        "-p", str(details["port"]),
        f"root@{details['ip']}",
    ]


def _scp_to_runpod(details: dict[str, Any], source: Path, target: str) -> None:
    command = [
        "scp",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        "-P", str(details["port"]),
        "-i", str(details["key"]),
        str(source),
        f"root@{details['ip']}:{target}",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"scp upload timed out after {exc.timeout} seconds") from exc
    if result.returncode:
        detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip())
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"scp upload failed with exit code {result.returncode}{suffix}")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http_ready(endpoint: str, *, timeout_sec: int = 180) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + "/system_stats", timeout=5) as response:
                response.read()
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = _safe_error(exc)
        time.sleep(2)
    raise ValueError(f"ComfyUI endpoint did not become ready before timeout: {last_error}")


def _start_runpod_session_lease_guard(details: dict[str, Any], *, workspace: str, run_id: str, max_lease_sec: int = 12 * 3600) -> None:
    """Start the Pod-side lease fuse before any render setup or uploads."""

    if max_lease_sec <= 0:
        return
    pod_id = details.get("pod_id")
    pod_id_value = pod_id if isinstance(pod_id, str) else ""
    log_path = f"{workspace.rstrip('/')}/runs/{run_id}/logs/stdout.log"
    script = f"""
set -euo pipefail
mkdir -p {shlex.quote(str(PurePosixPath(log_path).parent))}
touch {shlex.quote(log_path)}
(
  sleep {int(max_lease_sec)}
  echo "Kura render max lease expired after {int(max_lease_sec)} seconds; attempting to delete RunPod pod" >> {shlex.quote(log_path)} 2>&1 || true
  if command -v runpodctl >/dev/null 2>&1 && [ -n {shlex.quote(pod_id_value)} ]; then
    runpodctl pod delete {shlex.quote(pod_id_value)} >> {shlex.quote(log_path)} 2>&1 || true
  else
    echo "Kura render max lease could not delete pod: runpodctl or pod id is unavailable" >> {shlex.quote(log_path)} 2>&1 || true
  fi
) </dev/null >/dev/null 2>&1 &
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"remote lease guard setup timed out after {exc.timeout} seconds") from exc
    if result.returncode:
        detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "lease guard setup failed")
        raise ValueError(f"remote lease guard setup failed with exit code {result.returncode}: {detail}")


def _sync_runpod_remote_stdout(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> bool:
    """Mirror remote stdout progress into local run artifacts."""

    try:
        with _run_operation_lock(run_dir, "remote-log"):
            return _sync_runpod_remote_stdout_unlocked(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=timeout_sec)
    except (OSError, ValueError):
        return False


def _sync_runpod_remote_stdout_unlocked(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> bool:
    """Perform one remote-log sync while the per-run log lock is held."""

    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    offset = status.get("remote_log_bytes")
    if not isinstance(offset, int) or offset < 0:
        offset = 0
    remote_log = f"{workspace.rstrip('/')}/runs/{run_id}/logs/stdout.log"
    marker = "__KURA_LOG_SIZE__:"
    script = f"""
set -u
log={shlex.quote(remote_log)}
offset={offset}
if [ -f "$log" ]; then
  size=$(wc -c < "$log" | tr -d ' ')
  if [ "$size" -lt "$offset" ]; then
    offset=0
  fi
  if [ "$size" -gt "$offset" ]; then
    tail -c +$((offset + 1)) "$log"
  fi
  printf '\\n{marker}%s\\n' "$size"
else
  printf '\\n{marker}0\\n'
fi
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], capture_output=True, check=False, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode:
        return False
    sentinel = ("\n" + marker).encode("utf-8")
    if sentinel not in result.stdout:
        return False
    payload, suffix = result.stdout.rsplit(sentinel, 1)
    first = suffix.splitlines()[0] if suffix.splitlines() else b""
    try:
        remote_size = int(first.decode("ascii", errors="replace").strip())
    except ValueError:
        return False
    if remote_size < offset:
        offset = 0
    if payload.startswith(b"\n") and offset == remote_size:
        payload = payload[1:]
    if payload:
        log_path = run_dir / "logs" / "stdout.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("ab") as handle:
            handle.write(payload)
    status["remote_log_bytes"] = remote_size
    status["remote_log_synced_at"] = datetime.now().astimezone().isoformat()
    _materialize_stdout_progress(run_dir, status, state=str(status.get("state") or "running"))
    atomic_write_json(status_path, _redact_secrets(status))
    return True


def _try_sync_runpod_remote_stdout(run_dir: Path, *, ssh_timeout_sec: int = 10) -> bool:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            return False
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
        if not isinstance(workspace, str):
            workspace = "/workspace"
        details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=2)
        return _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_dir.name)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _read_runpod_remote_exit(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> dict[str, Any] | None:
    remote_dir = f"{workspace.rstrip('/')}/runs/{run_id}/realizations"
    script = f"""
set -u
dir={shlex.quote(remote_dir)}
latest=$(ls -1 "$dir"/remote-exit-*.json 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$latest" ]; then
  cat "$latest"
fi
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _record_remote_exit_observation(run_dir: Path, exit_record: dict[str, Any]) -> None:
    """Append a remote-completion fact while local output recovery is pending."""

    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    exit_code = exit_record.get("exit_code")
    if not isinstance(exit_code, int):
        return
    if status.get("remote_exit_code") == exit_code and status.get("last_remote_exit_observation"):
        return
    realization_ref = status.get("last_realization")
    realization_id = Path(realization_ref).stem if isinstance(realization_ref, str) else "runpod"
    observed_at = datetime.now().astimezone().isoformat()
    compact = re.sub(r"[^0-9]", "", observed_at)[:20]
    observation_path = run_dir / "realizations" / f"{realization_id}.remote-exit-observed-{compact}.json"
    observation = {
        "event": "remote_exit_observed",
        "realization_id": realization_id,
        "observed_at": observed_at,
        "remote_state": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "remote_timestamp": exit_record.get("timestamp"),
        "recovery_required": True,
    }
    atomic_write_json(observation_path, _redact_secrets(observation))
    status.update({
        "remote_state": observation["remote_state"],
        "remote_exit_code": exit_code,
        "remote_ended": exit_record.get("timestamp"),
        "recovery_required": True,
        "last_remote_exit_observation": str(observation_path.relative_to(run_dir)),
    })
    atomic_write_json(status_path, _redact_secrets(status))
    append_run_event(run_dir, observation, best_effort=True)


def _try_observe_runpod_remote_exit(run_dir: Path, *, ssh_timeout_sec: int = 10) -> bool:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            return False
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
        if not isinstance(workspace, str):
            workspace = "/workspace"
        details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=2)
        exit_record = _read_runpod_remote_exit(details, workspace=workspace, run_id=run_dir.name, timeout_sec=30)
        if exit_record is None:
            return False
        _record_remote_exit_observation(run_dir, exit_record)
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _runpod_secret_env_payload(*, remote_notify: bool = False) -> str | None:
    lines: list[str] = []
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        quoted = shlex.quote(hf_token)
        lines.extend([
            f"export HF_TOKEN={quoted}",
            f"export HUGGINGFACE_HUB_TOKEN={quoted}",
        ])
    if remote_notify and os.environ.get("KURA_NTFY_TOPIC"):
        lines.append("export KURA_REMOTE_NOTIFY_NTFY=1")
        for key in ("KURA_NTFY_TOPIC", "KURA_NTFY_SERVER", "KURA_NTFY_TOKEN", "KURA_NTFY_PRIORITY"):
            value = os.environ.get(key)
            if value:
                lines.append(f"export {key}={shlex.quote(value)}")
    if not lines:
        return None
    return "\n".join([*lines, ""])


def _runpod_remote_job_script(
    *,
    workspace: str,
    run_id: str,
    remote_secret_path: str,
    archive_name: str,
    remote_archive: str,
    cwd: str,
    command: str,
) -> str:
    return f"""
set -u
secret_file={shlex.quote(remote_secret_path)}
cleanup() {{
  rm -f "$secret_file"
}}
trap cleanup EXIT
if [ -f "$secret_file" ]; then
  . "$secret_file"
fi
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
export KURA_WORKSPACE={shlex.quote(workspace)}
export KURA_RUN_ID={shlex.quote(run_id)}
export KURA_LOG_PATH={shlex.quote(workspace + '/runs/' + run_id + '/logs/stdout.log')}
export HF_HOME="$KURA_WORKSPACE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/samples" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/metrics"
mkdir -p "$HF_HUB_CACHE" "$KURA_WORKSPACE/cache/models"
case "$HF_HOME" in "$KURA_WORKSPACE"/*) ;; *) echo "[kura] HF_HOME must be under KURA_WORKSPACE before remote job start: $HF_HOME" >&2; exit 1 ;; esac
case "$HF_HUB_CACHE" in "$HF_HOME"/*) ;; *) echo "[kura] HF_HUB_CACHE must be under HF_HOME before remote job start: $HF_HUB_CACHE" >&2; exit 1 ;; esac
touch "$KURA_LOG_PATH"
echo "Kura controller uploaded {shlex.quote(archive_name)}" >> "$KURA_LOG_PATH"
read_cgroup_value() {{
  file="$1"
  key="${{2:-}}"
  if [ ! -r "$file" ]; then
    printf '%s' "unknown"
    return
  fi
  if [ -n "$key" ]; then
    value=$(awk -v key="$key" '$1 == key {{ print $2; exit }}' "$file" 2>/dev/null || true)
  else
    value=$(cat "$file" 2>/dev/null || true)
  fi
  printf '%s' "${{value:-unknown}}"
}}
read_memory_current() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.current)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.usage_in_bytes); fi
  printf '%s' "$value"
}}
read_memory_peak() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.peak)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.max_usage_in_bytes); fi
  printf '%s' "$value"
}}
read_memory_max() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.max)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.limit_in_bytes); fi
  printf '%s' "$value"
}}
read_oom_kill() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.events oom_kill)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.oom_control oom_kill); fi
  printf '%s' "$value"
}}
collect_runtime_diagnostics() {{
  phase="$1"
  {{
    echo "[kura] runtime diagnostics $phase"
    echo "[kura] cgroup memory.current=$(read_memory_current)"
    echo "[kura] cgroup memory.peak=$(read_memory_peak)"
    echo "[kura] cgroup memory.max=$(read_memory_max)"
    echo "[kura] cgroup memory.events"
    if [ -r /sys/fs/cgroup/memory.events ]; then
      sed 's/^/[kura]   /' /sys/fs/cgroup/memory.events
    elif [ -r /sys/fs/cgroup/memory/memory.oom_control ]; then
      sed 's/^/[kura]   /' /sys/fs/cgroup/memory/memory.oom_control
      echo "[kura]   failcnt $(read_cgroup_value /sys/fs/cgroup/memory/memory.failcnt)"
    else
      echo "[kura]   unavailable"
    fi
    echo "[kura] proc meminfo"
    if [ -r /proc/meminfo ]; then grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo | sed 's/^/[kura]   /'; else echo "[kura]   unavailable"; fi
    echo "[kura] cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits 2>&1 | sed 's/^/[kura] gpu /'
    else
      echo "[kura] gpu nvidia-smi unavailable"
    fi
    df -Pk "$KURA_WORKSPACE" 2>&1 | sed 's/^/[kura] disk /'
  }} >> "$KURA_LOG_PATH" 2>&1
}}
export KURA_CGROUP_OOM_KILL_BEFORE=$(read_oom_kill)
collect_runtime_diagnostics before_backend
exit_code=0
tar -xzf {shlex.quote(remote_archive)} -C "$KURA_WORKSPACE" >> "$KURA_LOG_PATH" 2>&1 || exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  cd {shlex.quote(cwd)} || exit_code=$?
fi
if [ "$exit_code" -eq 0 ]; then
  {command} >> "$KURA_LOG_PATH" 2>&1
  exit_code=$?
fi
collect_runtime_diagnostics after_backend
export KURA_CGROUP_OOM_KILL_AFTER=$(read_oom_kill)
export KURA_CGROUP_MEMORY_PEAK=$(read_memory_peak)
export KURA_EXIT_CODE="$exit_code"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/realizations"
python - <<'PY'
import json, os, urllib.request
from datetime import datetime
run_id = os.environ["KURA_RUN_ID"]
workspace = os.environ.get("KURA_WORKSPACE", "/workspace")
now = datetime.now().astimezone().isoformat()
exit_code = int(os.environ.get("KURA_EXIT_CODE", "0"))
def optional_int(name):
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return None
oom_before = optional_int("KURA_CGROUP_OOM_KILL_BEFORE")
oom_after = optional_int("KURA_CGROUP_OOM_KILL_AFTER")
memory_peak = optional_int("KURA_CGROUP_MEMORY_PEAK")
path = f"{{workspace}}/runs/{{run_id}}/realizations/remote-exit-{{now.replace(':', '').replace('.', '-')}}.json"
with open(path, "w", encoding="utf-8") as handle:
    json.dump({{
        "event": "remote_exit",
        "timestamp": now,
        "exit_code": exit_code,
        "diagnostics": {{
            "cgroup_oom_kill_before": oom_before,
            "cgroup_oom_kill_after": oom_after,
            "cgroup_oom_kill_delta": oom_after - oom_before if oom_before is not None and oom_after is not None else None,
            "cgroup_memory_peak_bytes": memory_peak,
        }},
    }}, handle, ensure_ascii=False, indent=2)
    handle.write("\\n")
if os.environ.get("KURA_REMOTE_NOTIFY_NTFY") == "1" and os.environ.get("KURA_NTFY_TOPIC"):
    try:
        server = os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        topic = os.environ["KURA_NTFY_TOPIC"].lstrip("/")
        title = f"Kura remote finished: {{run_id}}"
        body = f"Remote training finished with exit code {{exit_code}}. Pod may still be billing until the controller downloads outputs and stops it."
        headers = {{"Title": title, "Tags": "warning" if exit_code else "white_check_mark", "Priority": os.environ.get("KURA_NTFY_PRIORITY", "4")}}
        token = os.environ.get("KURA_NTFY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {{token}}"
        request = urllib.request.Request(f"{{server}}/{{topic}}", data=body.encode("utf-8"), method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except Exception:
        pass
PY
exit "$exit_code"
""".strip()


def _runpod_run_over_ssh(run_dir: Path, *, ssh_timeout_sec: int, job_timeout_sec: int | None, remote_notify: bool = False, max_lease_sec: int = 12 * 3600) -> int:
    stage = _latest_runpod_stage(run_dir)
    archive = stage.get("archive")
    archive_name = stage.get("archive_name")
    if not isinstance(archive, str) or not isinstance(archive_name, str):
        raise ValueError("latest RunPod stage has no upload archive")
    archive_path = run_dir / archive
    if not archive_path.is_file():
        raise ValueError(f"upload archive is missing: {archive_path}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no RunPod realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
    run_id = run_dir.name
    cwd = realization.get("container_cwd")
    argv = realization.get("backend_command")
    if not isinstance(workspace, str) or not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("latest realization has no runnable RunPod command")
    details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec)
    remote_archive = f"{workspace}/{archive_name}"
    prepared = _run_bounded([*_ssh_base(details), f"mkdir -p {shlex.quote(workspace)}"], context="ssh workspace preparation")
    if prepared.returncode:
        raise ValueError(f"ssh workspace preparation failed with exit code {prepared.returncode}")
    scp = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(details["port"]),
        "-i", str(details["key"]),
        str(archive_path),
        f"root@{details['ip']}:{remote_archive}",
    ]
    uploaded = _run_bounded(scp, context="scp upload")
    if uploaded.returncode:
        raise ValueError(f"scp upload failed with exit code {uploaded.returncode}")
    command = " ".join(shlex.quote(arg) for arg in argv)
    remote_secret_path = f"/tmp/kura-secrets/{run_id}.env"
    secret_payload = _runpod_secret_env_payload(remote_notify=remote_notify)
    if secret_payload is not None:
        install_secret_script = f"""
set -euo pipefail
umask 077
mkdir -p /tmp/kura-secrets
cat > {shlex.quote(remote_secret_path)}
chmod 600 {shlex.quote(remote_secret_path)}
""".strip()
        installed = _run_bounded([*_ssh_base(details), install_secret_script], input=secret_payload, text=True, context="ssh secret preparation")
        if installed.returncode:
            raise ValueError(f"ssh secret preparation failed with exit code {installed.returncode}")
    lease_log_path = f"{workspace}/runs/{run_id}/logs/stdout.log"
    pod_id = status.get("pod_id")
    pod_id_value = pod_id if isinstance(pod_id, str) else ""
    remote_job_script = _runpod_remote_job_script(
        workspace=workspace,
        run_id=run_id,
        remote_secret_path=remote_secret_path,
        archive_name=archive_name,
        remote_archive=remote_archive,
        cwd=cwd,
        command=command,
    )
    remote_job_path = f"/tmp/kura-jobs/{run_id}.sh"
    remote_controller_log = f"/tmp/kura-jobs/{run_id}.controller.log"
    lease_guard = ""
    if max_lease_sec > 0:
        lease_guard = f"""
(
  KURA_LEASE_LOG_PATH={shlex.quote(lease_log_path)}
  RUNPOD_POD_ID={shlex.quote(pod_id_value)}
  sleep {int(max_lease_sec)}
  mkdir -p "$(dirname "$KURA_LEASE_LOG_PATH")" || true
  echo "Kura max lease expired after {int(max_lease_sec)} seconds; attempting to delete RunPod pod" >> "$KURA_LEASE_LOG_PATH" 2>&1 || true
  if command -v runpodctl >/dev/null 2>&1 && [ -n "$RUNPOD_POD_ID" ]; then
    runpodctl pod delete "$RUNPOD_POD_ID" >> "$KURA_LEASE_LOG_PATH" 2>&1 || true
  else
    echo "Kura max lease could not delete pod: runpodctl or RUNPOD_POD_ID is unavailable" >> "$KURA_LEASE_LOG_PATH" 2>&1 || true
  fi
) </dev/null >/dev/null 2>&1 &
""".strip()
    start_script = f"""
set -euo pipefail
mkdir -p /tmp/kura-jobs
cat > {shlex.quote(remote_job_path)}
chmod 700 {shlex.quote(remote_job_path)}
{lease_guard}
nohup sh {shlex.quote(remote_job_path)} </dev/null >{shlex.quote(remote_controller_log)} 2>&1 &
echo $!
""".strip()
    started = _run_bounded([*_ssh_base(details), start_script], input=remote_job_script, text=True, capture_output=True, context="remote job start")
    if started.returncode:
        detail = _redact_secret_text(started.stderr.strip() or started.stdout.strip() or "remote job start failed")
        raise ValueError(f"remote job start failed with exit code {started.returncode}: {detail}")
    remote_pid = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else None
    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["remote_pid"] = remote_pid
        status["remote_job_started_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(status_path, _redact_secrets(status))
    except (OSError, json.JSONDecodeError):
        pass
    deadline = time.monotonic() + job_timeout_sec if job_timeout_sec and job_timeout_sec > 0 else None
    next_sync = 0.0
    sync_interval_sec = 20.0
    while True:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise subprocess.TimeoutExpired(["runpod-remote-job", run_id], job_timeout_sec)
        if now >= next_sync:
            _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=30)
            exit_record = _read_runpod_remote_exit(details, workspace=workspace, run_id=run_id, timeout_sec=30)
            if exit_record is not None:
                _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=30)
                _record_remote_exit_observation(run_dir, exit_record)
                exit_code = exit_record.get("exit_code")
                return int(exit_code) if isinstance(exit_code, int) else 1
            next_sync = now + sync_interval_sec
        time.sleep(2)


def download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    for _ in range(attempts):
        code = download_run(run_id, force=True)
        if code == 0:
            return 0
        time.sleep(interval_sec)
    return 1


def _download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    return download_with_retries(run_id, attempts, interval_sec)


def _remote_path_size(details: dict[str, Any], path: str, *, timeout_sec: int = 60) -> int | None:
    script = f"du -sb {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
