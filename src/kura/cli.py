"""Command-line interface for Kura's initial file-based workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from copy import deepcopy

import yaml

from kura import __version__
from kura.backends import command_ai_toolkit, command_musubi_tuner, compile_ai_toolkit, compile_musubi_tuner
from kura.executors import _materialize_stdout_progress, _redact_secret_text, _redact_secrets, launch_docker, launch_runpod, reconcile_docker, reconcile_runpod, stage_runpod, stop_docker, stop_runpod
from kura.render import compile_render, launch_render
from kura.tui import run_textual_monitor


def _dump_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _workspace() -> Path:
    return Path.cwd()


def _workspace_config() -> dict[str, Any]:
    return _load_yaml(_workspace() / "workspace.yaml")


def _parse_env_file_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export "):].lstrip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def _load_env_local(path: Path | None = None) -> None:
    env_path = path or (_workspace() / ".env.local")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_file_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _image_config(name: str) -> dict[str, Any]:
    try:
        image = _workspace_config()["docker"]["images"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"workspace.yaml has no docker.images.{name} configuration") from exc
    if not isinstance(image, dict) or not all(isinstance(image.get(key), str) for key in ("local", "remote", "dockerfile", "context")):
        raise ValueError(f"docker.images.{name} requires local, remote, dockerfile, and context strings")
    return image


def _backend_image_name(backend_name: Any) -> str:
    if backend_name == "musubi-tuner":
        return "musubi-tuner"
    return "ai-toolkit"


def _command_for_backend(run: dict[str, Any]) -> dict[str, Any]:
    backend_name = run.get("backend", {}).get("name") if isinstance(run.get("backend"), dict) else None
    if backend_name == "ai-toolkit":
        return command_ai_toolkit(run)
    if backend_name == "musubi-tuner":
        return command_musubi_tuner(run)
    raise ValueError(f"unsupported backend: {backend_name}")


def _docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=capture, check=False)


def _secret_state() -> dict[str, str]:
    import os
    return {name: "present" if os.environ.get(name) else "absent" for name in ("HF_TOKEN", "RUNPOD_API_KEY", "GHCR_TOKEN", "DOCKERHUB_TOKEN")}


def _safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def _dataset_digest(dataset_id: str) -> str:
    if not dataset_id or Path(dataset_id).name != dataset_id:
        raise ValueError("training run dataset.id must name a dataset directory")
    directory = _workspace() / "datasets" / dataset_id
    files = (directory / "dataset.yaml", directory / "items.jsonl")
    if not all(path.is_file() for path in files):
        raise ValueError(f"dataset {dataset_id!r} must contain dataset.yaml and items.jsonl")
    hasher = hashlib.sha256()
    for path in files:
        hasher.update(path.name.encode("utf-8") + b"\0")
        hasher.update(path.read_bytes() + b"\0")
    return "sha256:" + hasher.hexdigest()


def _run_datasets(run: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = run.get("datasets")
    if isinstance(datasets, list):
        return [item for item in datasets if isinstance(item, dict)]
    dataset = run.get("dataset")
    if isinstance(dataset, dict):
        return [dataset]
    return []


def _run_path(run_id: str) -> Path:
    return _workspace() / "runs" / run_id


def _now() -> datetime:
    return datetime.now().astimezone()


def _parse_duration_seconds(value: Any) -> int:
    if value in (None, "", False):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    text = str(value).strip().lower()
    if not text:
        return 0
    match = re.fullmatch(r"(\d+)([smhd]?)", text)
    if not match:
        raise ValueError("duration must be an integer seconds value or use s/m/h/d suffix")
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * scale


def _format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 86400 and seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60
        return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"
    if seconds >= 60:
        minutes, remainder = divmod(seconds, 60)
        return f"{minutes}m{remainder:02d}s" if remainder else f"{minutes}m"
    return f"{seconds}s"


def _event(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _notification_channels(raw: Any) -> list[str]:
    explicit = raw
    if explicit in (None, "", False):
        explicit = os.environ.get("KURA_NOTIFY")
    if explicit not in (None, "", False):
        if isinstance(explicit, str):
            values = [part.strip().lower() for part in explicit.split(",") if part.strip()]
            if any(value in ("none", "off", "false", "0") for value in values):
                return []
            return values
        if isinstance(explicit, (list, tuple)):
            return [str(part).strip().lower() for part in explicit if str(part).strip()]
    channels: list[str] = []
    if shutil.which("notify-send"):
        channels.append("desktop")
    if os.environ.get("KURA_NTFY_TOPIC"):
        channels.append("ntfy")
    return channels


def _notify(channels: Any, *, subject: str, body: str) -> None:
    selected = _notification_channels(channels)
    if not selected:
        return
    for channel in selected:
        try:
            if channel == "desktop":
                if shutil.which("notify-send"):
                    subprocess.run(["notify-send", subject, body], check=False)
                continue
            if channel == "ntfy":
                _send_ntfy_notification(subject, body)
                continue
            print(f"warning: unknown notification channel: {channel}", file=sys.stderr)
        except Exception as exc:  # notification must never break run lifecycle
            print(f"warning: notification failed ({channel}): {_safe_error(exc)}", file=sys.stderr)


def _send_ntfy_notification(subject: str, body: str) -> None:
    topic = os.environ.get("KURA_NTFY_TOPIC")
    if not topic:
        raise ValueError("ntfy notification requires KURA_NTFY_TOPIC")
    server = os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    token = os.environ.get("KURA_NTFY_TOKEN")
    priority = os.environ.get("KURA_NTFY_PRIORITY", "4")
    url = f"{server}/{topic.lstrip('/')}"
    headers = {"Title": subject, "Tags": "rocket", "Priority": priority}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body.encode("utf-8"), method="POST", headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def _sleep_with_completion_reminders(*, delay_sec: int, interval_sec: int, channels: Any, subject: str, body: str) -> None:
    remaining = max(0, int(delay_sec))
    interval = max(0, int(interval_sec))
    elapsed = 0
    while remaining > 0:
        chunk = remaining if interval <= 0 else min(interval, remaining)
        time.sleep(chunk)
        elapsed += chunk
        remaining -= chunk
        if interval > 0 and remaining > 0:
            _notify(
                channels,
                subject=f"{subject} (reminder)",
                body=f"{body}\nReminder: {_format_duration(elapsed)} since completion. Pod stops in about {_format_duration(remaining)}.",
            )


def cmd_init(_: argparse.Namespace) -> int:
    root = _workspace()
    for relative in ("datasets", "experiments", "runs", "workflows", "promptsets", "backends", "executors", "docker/ai-toolkit", "docker/musubi-tuner"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace.yaml"
    if not workspace.exists():
        _dump_yaml(workspace, {"schema_version": 1, "name": root.name, "docker": {"images": {"ai-toolkit": {"local": "kura/ai-toolkit:dev", "remote": "nomadoor/kura-ai-toolkit:dev", "dockerfile": "docker/ai-toolkit/Dockerfile", "context": "."}, "musubi-tuner": {"local": "kura/musubi-tuner:dev", "remote": "nomadoor/kura-musubi-tuner:dev", "dockerfile": "docker/musubi-tuner/Dockerfile", "context": "."}}, "workspace_target": "/workspace", "gpu": True, "mounts": [{"source": "/mnt/d/kura-cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]}, "runpod": {"default_image": {"ai-toolkit": "ostris/aitoolkit:latest", "musubi-tuner": "nomadoor/kura-musubi-tuner:dev"}, "template_id": "0fqzfjy6f3", "api_key_env": "RUNPOD_API_KEY", "storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"], "gpu_count": 1, "container_disk_gb": 150, "volume_in_gb": 0, "workspace_path": "/workspace", "container_cwd": "/app/ai-toolkit", "ports": ["8675/http", "22/tcp"], "cloud_type": "ANY", "gpu_type_priority": "availability", "interruptible": False}})
    agents = root / "AGENTS.md"
    if not agents.exists():
        agents.write_text("# Repository Guidelines\n\nKura is file-first: use the CLI for mutations and keep secrets out of run artifacts.\n", encoding="utf-8")
    (root / "index.jsonl").touch(exist_ok=True)
    dockerfile = root / "docker/ai-toolkit/Dockerfile"
    if not dockerfile.exists():
        dockerfile.write_text(
            "FROM ostris/aitoolkit:latest\n\nCOPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py\nRUN ln -sfn /app/ai-toolkit /opt/ai-toolkit\nWORKDIR /workspace\nCMD [\"/start.sh\"]\n",
            encoding="utf-8",
        )
    musubi_dockerfile = root / "docker/musubi-tuner/Dockerfile"
    if not musubi_dockerfile.exists():
        musubi_dockerfile.write_text(
            "ARG PYTORCH_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime\nFROM ${PYTORCH_BASE}\n\nARG MUSUBI_TUNER_REF=main\nENV DEBIAN_FRONTEND=noninteractive\nENV PIP_DEFAULT_TIMEOUT=180\nENV PIP_RETRIES=5\n\nRUN apt-get update \\\n    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \\\n    && rm -rf /var/lib/apt/lists/*\n\nRUN git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner \\\n    && cd /opt/musubi-tuner \\\n    && git checkout \"$MUSUBI_TUNER_REF\" \\\n    && pip install --no-cache-dir -e .\n\nCOPY docker/musubi-tuner/patch_flux2_diffusers_vae.py /tmp/patch_flux2_diffusers_vae.py\nRUN python /tmp/patch_flux2_diffusers_vae.py \\\n    && rm /tmp/patch_flux2_diffusers_vae.py\n\nCOPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py\nWORKDIR /workspace\nCMD [\"sleep\", \"infinity\"]\n",
            encoding="utf-8",
        )
    musubi_patch = root / "docker/musubi-tuner/patch_flux2_diffusers_vae.py"
    if not musubi_patch.exists():
        musubi_patch.write_text(
            "from pathlib import Path\n\n\nTARGET = Path(\"/opt/musubi-tuner/src/musubi_tuner/flux_2/flux2_utils.py\")\n\n\nOLD = \"\"\"    logger.info(f\\\"Loading state dict from {ckpt_path}\\\")\n    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)\n    info = ae.load_state_dict(sd, strict=True, assign=True)\n\"\"\"\n\n\nNEW = \"\"\"    logger.info(f\\\"Loading state dict from {ckpt_path}\\\")\n    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)\n    if any(\\\".down_blocks.\\\" in key or \\\".up_blocks.\\\" in key or key.endswith(\\\"conv_norm_out.weight\\\") for key in sd):\n        logger.info(\\\"Converting Diffusers-layout Flux2 VAE state dict\\\")\n        from musubi_tuner.ideogram4.ideogram4_autoencoder import convert_diffusers_state_dict\n\n        sd = convert_diffusers_state_dict(sd)\n    info = ae.load_state_dict(sd, strict=True, assign=True)\n\"\"\"\n\n\ndef main() -> None:\n    text = TARGET.read_text()\n    if NEW in text:\n        return\n    if OLD not in text:\n        raise SystemExit(f\"expected load_ae block not found in {TARGET}\")\n    TARGET.write_text(text.replace(OLD, NEW))\n\n\nif __name__ == \"__main__\":\n    main()\n",
            encoding="utf-8",
        )
    print(f"initialized workspace: {root}")
    return 0


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    directory = Path(args.dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest = directory / "dataset.yaml"
    items = directory / "items.jsonl"
    if not manifest.exists():
        errors.append("missing dataset.yaml")
    if not items.exists():
        errors.append("missing items.jsonl")
    if errors:
        print("dataset validation failed: " + "; ".join(errors), file=sys.stderr)
        return 1
    try:
        metadata = _load_yaml(manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"dataset validation failed: invalid dataset.yaml: {_safe_error(exc)}", file=sys.stderr)
        return 1
    count = 0
    for number, line in enumerate(items.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"items.jsonl:{number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(item, dict) or not item.get("id") or not item.get("path"):
            errors.append(f"items.jsonl:{number}: item requires id and path")
        if not item.get("caption"):
            warnings.append(f"items.jsonl:{number}: missing caption")
        if not item.get("hash"):
            warnings.append(f"items.jsonl:{number}: missing hash")
        count += 1
    declared = metadata.get("stats", {}).get("count")
    if declared != count:
        warnings.append(f"stats.count is {declared!r}, but items.jsonl contains {count} items")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"dataset valid: {count} items")
    return 0


def cmd_run_new(args: argparse.Namespace) -> int:
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    if not safe_slug:
        print("slug must contain letters or numbers", file=sys.stderr)
        return 1
    timestamp = _now()
    run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"
    run_dir = _run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("resolved", "logs", "metrics", "samples", "checkpoints", "outputs"):
        (run_dir / relative).mkdir()
    run = {
        "schema_version": 1, "id": run_id, "type": "train", "experiment": args.experiment,
        "created": timestamp.isoformat(), "created_by": "human", "parent_run": None, "intent": "",
        "backend": {"name": "ai-toolkit", "version": None, "adapter_version": 1},
        "model": {"base": "", "revision": None},
        "datasets": [{"id": "", "digest": None, "role": None}],
        "params": {key: None for key in ("rank", "alpha", "lr", "scheduler", "steps", "batch_size", "resolution", "seed")},
        "backend_overrides": {}, "compute": {"executor": "docker", "gpu": None},
        "sampling": {"prompts": [], "cadence_steps": None},
    }
    _dump_yaml(run_dir / "run.yaml", run)
    (run_dir / "status.json").write_text(json.dumps({"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "plan.md").write_text("# Training plan\n\n", encoding="utf-8")
    (run_dir / "notes.md").write_text("# Notes\n\n", encoding="utf-8")
    for relative in ("logs/events.jsonl", "metrics/metrics.jsonl", "samples/samples.jsonl"):
        (run_dir / relative).touch()
    print(run_id)
    return 0


def cmd_render_new(args: argparse.Namespace) -> int:
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    if not safe_slug:
        print("slug must contain letters or numbers", file=sys.stderr); return 1
    timestamp = _now(); run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"; run_dir = _run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("resolved", "logs", "samples/images", "outputs"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    run = {"schema_version": 1, "id": run_id, "type": "render", "created": timestamp.isoformat(), "created_by": "human", "intent": "", "inputs": {"train_run": None, "checkpoint": {"path": "", "hash": None}, "workflow": {"path": "", "digest": None}, "promptset": {"path": "", "digest": None}}, "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"}, "executor": {"name": "local"}, "workflow_patches": {}, "render": {"output_dir": "samples/images", "timeout_sec": 600, "default_seed": None}}
    _dump_yaml(run_dir / "run.yaml", run)
    (run_dir / "status.json").write_text(json.dumps({"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "plan.md").write_text("# Render plan\n\n", encoding="utf-8"); (run_dir / "notes.md").write_text("# Notes\n\n", encoding="utf-8")
    (run_dir / "logs/events.jsonl").touch(); (run_dir / "samples/images.jsonl").touch()
    print(run_id); return 0


def cmd_run_compile(args: argparse.Namespace) -> int:
    run_dir = _run_path(args.run_id)
    try:
        run = _load_yaml(run_dir / "run.yaml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    try:
        current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if current_status.get("state") != "draft":
        print("cannot compile run: resolved artifacts are immutable; create a new run instead", file=sys.stderr)
        return 1
    if run.get("type", "train") == "render":
        try:
            compile_render(_workspace(), run_dir)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            print(f"cannot compile render: {_safe_error(exc)}", file=sys.stderr); return 1
        print(f"compiled render: {args.run_id}"); return 0
    backend = run.get("backend", {})
    if backend.get("name") not in ("ai-toolkit", "musubi-tuner"):
        print(f"unsupported backend: {backend.get('name')}", file=sys.stderr)
        return 1
    try:
        datasets = _run_datasets(run)
        if not datasets:
            raise ValueError("training run requires datasets[]")
        locked_datasets = []
        for dataset in datasets:
            dataset_id = dataset.get("id")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError("training run datasets[].id must name a dataset directory")
            actual_digest = _dataset_digest(dataset_id)
            if dataset.get("digest") not in (None, actual_digest):
                raise ValueError(f"dataset {dataset_id!r} digest does not match the current dataset files")
            locked_item = deepcopy(dataset)
            locked_item["digest"] = actual_digest
            locked_datasets.append(locked_item)
        image = _image_config(_backend_image_name(backend.get("name")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    resolved = run_dir / "resolved"
    resolved.mkdir(exist_ok=True)
    locked = deepcopy(run)
    locked["datasets"] = locked_datasets
    locked.pop("dataset", None)
    locked["_kura"] = {"frozen_at": _now().isoformat(), "artifact": "manifest.lock"}
    _dump_yaml(resolved / "manifest.lock.yaml", locked)
    if backend.get("name") == "ai-toolkit":
        compile_ai_toolkit(locked, resolved / "ai-toolkit.toml")
    else:
        compile_musubi_tuner(locked, resolved / "musubi")
    env = {
        "kura_version": __version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "backend_name": backend.get("name"),
        "backend_adapter_version": backend.get("adapter_version"), "generated_at": _now().isoformat(),
        "declared_executor": "docker", "local_image": image["local"], "dockerfile": image["dockerfile"],
    }
    _dump_yaml(resolved / "env.lock", env)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["state"] = "compiled"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"compiled run: {args.run_id}")
    return 0


def cmd_run_status(args: argparse.Namespace) -> int:
    path = _run_path(args.run_id) / "status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        run_dir = path.parent
        realization_ref = status.get("last_realization")
        if isinstance(realization_ref, str) and (run_dir / realization_ref).is_file():
            status["latest_realization"] = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        observation_ref = status.get("last_observation")
        if isinstance(observation_ref, str) and (run_dir / observation_ref).is_file():
            status["latest_observation"] = json.loads((run_dir / observation_ref).read_text(encoding="utf-8"))
        status["summary"] = {
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "pod_id": status.get("pod_id"),
            "downloaded_run": status.get("downloaded_run"),
            "outputs": status.get("outputs", []),
        }
        print(json.dumps(status, indent=2))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read status: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_reconcile(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
        if realization.get("executor") == "runpod":
            try:
                reconcile_runpod(run_dir, _workspace_config().get("runpod", {}))
            except ValueError as exc:
                if "must be exported to reconcile a RunPod run" not in str(exc) or not _try_sync_runpod_remote_stdout(run_dir):
                    raise
                print(f"warning: skipped RunPod API reconcile because {_safe_error(exc)}; synced remote log over SSH only", file=sys.stderr)
            else:
                _try_sync_runpod_remote_stdout(run_dir)
            print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
        else:
            print(json.dumps(reconcile_docker(run_dir), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot reconcile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_prune(args: argparse.Namespace) -> int:
    workspace = _workspace()
    states = {state.strip() for state in args.states.split(",") if state.strip()}
    runs: list[dict[str, Any]] = []
    for run_file in sorted((workspace / "runs").glob("*/run.yaml")):
        run_dir = run_file.parent
        try:
            run = _load_yaml(run_file)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
            print(f"warning: skipped {run_dir.name}: {_safe_error(exc)}", file=sys.stderr)
            continue
        state = str(status.get("state") or "unknown")
        recency = status.get("ended") or status.get("started") or run.get("created") or ""
        runs.append({"id": run_dir.name, "state": state, "recency": recency, "path": run_dir})

    runs.sort(key=lambda item: str(item["recency"]), reverse=True)
    keep_ids = {item["id"] for item in runs[: max(args.keep, 0)]}
    candidates = [item for item in runs if item["id"] not in keep_ids and item["state"] in states]
    actions: list[dict[str, Any]] = []
    for item in candidates:
        run_dir = item["path"]
        if args.outputs_only:
            targets = [path for path in (run_dir / "outputs", run_dir / "downloads") if path.exists()]
        else:
            targets = [run_dir]
        actions.append({"id": item["id"], "state": item["state"], "targets": [str(path.relative_to(workspace)) for path in targets]})
        if args.yes:
            for target in targets:
                if target.exists():
                    shutil.rmtree(target)

    print(json.dumps({"dry_run": not args.yes, "outputs_only": args.outputs_only, "keep": args.keep, "states": sorted(states), "actions": actions}, ensure_ascii=False, indent=2))
    return 0


def cmd_run_stop(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
        if realization.get("executor") == "runpod":
            print(json.dumps(stop_runpod(run_dir, _workspace_config().get("runpod", {})), indent=2))
        else:
            print(json.dumps(stop_docker(run_dir), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot stop run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_logs(args: argparse.Namespace) -> int:
    try:
        status = json.loads((_run_path(args.run_id) / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if isinstance(realization_ref, str):
            realization = json.loads((_run_path(args.run_id) / realization_ref).read_text(encoding="utf-8"))
            if realization.get("executor") == "runpod":
                print(f"RunPod logs live in the remote workspace at {realization.get('logs_path')}; use the configured transfer method to fetch them. The RunPod console is diagnostic only.", file=sys.stderr)
                return 1
    except (OSError, json.JSONDecodeError):
        pass
    path = _run_path(args.run_id) / "logs" / "stdout.log"
    if not path.exists():
        print(f"no run log exists yet: {path}", file=sys.stderr)
        return 1
    command = ["tail", "-n", "200"]
    if args.follow:
        command.append("-f")
    command.append(str(path))
    try:
        return subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        print(path.read_text(encoding="utf-8"), end="")
        return 0


def cmd_run_stage(args: argparse.Namespace) -> int:
    if args.executor != "runpod":
        print(f"staging is not implemented for executor: {args.executor}", file=sys.stderr)
        return 2
    run_dir = _run_path(args.run_id)
    try:
        locked = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") == "running":
            raise ValueError("run is running; stop or reconcile it before staging")
        if status.get("state") not in ("compiled", "failed", "interrupted", "unknown", "launch_failed"):
            raise ValueError("run must be compiled before staging")
        dataset_ids = [item.get("id") for item in _run_datasets(locked)]
        dataset_ids = [item for item in dataset_ids if isinstance(item, str) and item]
        if not dataset_ids:
            raise ValueError("compiled run has no dataset IDs")
        print(json.dumps(stage_runpod(workspace=_workspace(), run_dir=run_dir, dataset_ids=dataset_ids, config=_workspace_config().get("runpod", {})), indent=2))
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"cannot stage run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


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


def cmd_run_download(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        destination = run_dir / "downloads"
        downloaded_run = destination / args.run_id

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
            status.update({"state": "completed" if exit_code == 0 else "failed", "exit_code": exit_code, "ended": remote_exit.get("timestamp"), "outputs": outputs, "downloaded_run": str(downloaded_run.relative_to(run_dir)), "remote_exit": str(exits[-1].relative_to(run_dir))})
            if exit_code == 0:
                try:
                    manifest = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
                    params = manifest.get("params") if isinstance(manifest.get("params"), dict) else {}
                    steps = params.get("steps")
                    if isinstance(steps, int) and steps > 0:
                        status["last_step"] = steps
                        status["total_steps"] = steps
                except (OSError, ValueError, yaml.YAMLError):
                    pass
            (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return True

        if downloaded_run.exists() and not getattr(args, "force", False):
            if materialize_downloaded_status():
                print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
                return 0
            raise ValueError("downloaded run snapshot is missing remote-exit; use --force to retry or inspect the Pod before stopping it")
        if downloaded_run.exists() and getattr(args, "force", False):
            shutil.rmtree(downloaded_run)
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before downloading")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        pod_id = status.get("pod_id")
        if not isinstance(pod_id, str):
            raise ValueError("run has no RunPod pod ID")
        pod = subprocess.run(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, check=False)
        if pod.returncode:
            raise ValueError(_redact_secret_text(pod.stderr.strip() or pod.stdout.strip() or "runpodctl pod get failed"))
        details = json.loads(pod.stdout)
        ssh = details.get("ssh", {})
        ip, port = ssh.get("ip"), ssh.get("port")
        key = ssh.get("ssh_key", {}).get("path")
        if not isinstance(ip, str) or not isinstance(port, int) or not isinstance(key, str):
            raise ValueError("pod SSH is not ready")
        destination.mkdir(exist_ok=True)
        remote_archive = f"/tmp/kura-download-{args.run_id}.tar.gz"
        remote_script = (
            f"tar -C /workspace/runs "
            f"--exclude {shlex.quote(args.run_id + '/cache')} "
            f"--exclude {shlex.quote(args.run_id + '/transfer')} "
            f"-czf {shlex.quote(remote_archive)} {shlex.quote(args.run_id)}"
        )
        packed = subprocess.run([*_ssh_base({"ip": ip, "port": port, "key": key}), remote_script], check=False)
        if packed.returncode:
            return packed.returncode
        local_archive = destination / f"kura-download-{args.run_id}.tar.gz"
        command = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(port),
            "-i", key,
            f"root@{ip}:{remote_archive}",
            str(local_archive),
        ]
        result = subprocess.run(command, check=False)
        subprocess.run([*_ssh_base({"ip": ip, "port": port, "key": key}), f"rm -f {shlex.quote(remote_archive)}"], check=False)
        if result.returncode:
            return result.returncode
        extracted = subprocess.run(["tar", "--warning=no-timestamp", "-xzf", str(local_archive), "-C", str(destination)], check=False)
        local_archive.unlink(missing_ok=True)
        if extracted.returncode:
            return extracted.returncode
        if not materialize_downloaded_status():
            raise ValueError("downloaded run snapshot is missing remote-exit; remote completion is not confirmed")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot download run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


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
    matches = re.findall(r"(?:step|_)(\\d{{4,}})(?=\\.safetensors$|[-_.])", name)
    step = int(matches[-1]) if matches else None
    items.append({{"path": path, "name": name, "step": step, "size": os.path.getsize(path)}})
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
        destination = run_dir / "pulled" / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
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
            result = subprocess.run([
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(details["port"]),
                "-i", str(details["key"]),
                f"root@{details['ip']}:{remote_path}",
                str(local_path),
            ], check=False)
            if result.returncode:
                return result.returncode
            pulled.append({"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": local_path.stat().st_size, "skipped": False})
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["pulled_outputs"] = pulled
        status["pulled_outputs_synced_at"] = datetime.now().astimezone().isoformat()
        status_path.write_text(json.dumps(_redact_secrets(status), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _event(run_dir, {"event": "run_outputs_pulled", "timestamp": datetime.now().astimezone().isoformat(), "count": len(pulled), "outputs": pulled})
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
        result = subprocess.run(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, check=False)
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


def _sync_runpod_remote_stdout(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> bool:
    """Mirror remote stdout progress into local run artifacts.

    The Pod's container disk is the live execution workspace during a remote
    run, while monitor intentionally reads only local Kura files.  This copies
    only the append-only stdout delta back to the local run log, then
    materializes step/total from that log into status.json.  It does not launch,
    stop, or otherwise mutate remote compute.
    """

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
    status_path.write_text(json.dumps(_redact_secrets(status), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    prepared = subprocess.run([*_ssh_base(details), f"mkdir -p {shlex.quote(workspace)}"], check=False)
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
    uploaded = subprocess.run(scp, check=False)
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
        installed = subprocess.run([*_ssh_base(details), install_secret_script], input=secret_payload, text=True, check=False)
        if installed.returncode:
            raise ValueError(f"ssh secret preparation failed with exit code {installed.returncode}")
    remote_job_script = f"""
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
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
touch "$KURA_LOG_PATH"
echo "Kura controller uploaded {shlex.quote(archive_name)}" >> "$KURA_LOG_PATH"
exit_code=0
tar -xzf {shlex.quote(remote_archive)} -C "$KURA_WORKSPACE" >> "$KURA_LOG_PATH" 2>&1 || exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  cd {shlex.quote(cwd)} || exit_code=$?
fi
if [ "$exit_code" -eq 0 ]; then
  {command} >> "$KURA_LOG_PATH" 2>&1
  exit_code=$?
fi
export KURA_EXIT_CODE="$exit_code"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/realizations"
python - <<'PY'
import json, os, urllib.request
from datetime import datetime
run_id = os.environ["KURA_RUN_ID"]
workspace = os.environ.get("KURA_WORKSPACE", "/workspace")
now = datetime.now().astimezone().isoformat()
exit_code = int(os.environ.get("KURA_EXIT_CODE", "0"))
path = f"{{workspace}}/runs/{{run_id}}/realizations/remote-exit-{{now.replace(':', '').replace('.', '-')}}.json"
with open(path, "w", encoding="utf-8") as handle:
    json.dump({{"event": "remote_exit", "timestamp": now, "exit_code": exit_code}}, handle, ensure_ascii=False, indent=2)
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
    remote_job_path = f"/tmp/kura-jobs/{run_id}.sh"
    remote_controller_log = f"/tmp/kura-jobs/{run_id}.controller.log"
    lease_guard = ""
    if max_lease_sec > 0:
        lease_guard = f"""
(
  sleep {int(max_lease_sec)}
  echo "Kura max lease expired after {int(max_lease_sec)} seconds; attempting to stop RunPod pod" >> "$KURA_LOG_PATH" 2>&1 || true
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${{RUNPOD_POD_ID:-}}" ]; then
    runpodctl pod stop "$RUNPOD_POD_ID" >> "$KURA_LOG_PATH" 2>&1 || true
  else
    echo "Kura max lease could not stop pod: runpodctl or RUNPOD_POD_ID is unavailable" >> "$KURA_LOG_PATH" 2>&1 || true
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
    started = subprocess.run([*_ssh_base(details), start_script], input=remote_job_script, text=True, capture_output=True, check=False)
    if started.returncode:
        detail = _redact_secret_text(started.stderr.strip() or started.stdout.strip() or "remote job start failed")
        raise ValueError(f"remote job start failed with exit code {started.returncode}: {detail}")
    remote_pid = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else None
    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["remote_pid"] = remote_pid
        status["remote_job_started_at"] = datetime.now().astimezone().isoformat()
        status_path.write_text(json.dumps(_redact_secrets(status), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
                exit_code = exit_record.get("exit_code")
                return int(exit_code) if isinstance(exit_code, int) else 1
            next_sync = now + sync_interval_sec
        time.sleep(2)


def _download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    for _ in range(attempts):
        code = cmd_run_download(argparse.Namespace(run_id=run_id, force=True))
        if code == 0:
            return 0
        time.sleep(interval_sec)
    return 1


def cmd_run_remote(args: argparse.Namespace) -> int:
    run_dir = _run_path(args.run_id)
    launched = False
    safe_to_stop = False
    exit_code = 1
    hold_for = 0
    notify_channels = getattr(args, "notify", None)
    notify_subject: str | None = None
    notify_body: str | None = None
    try:
        hold_for = _parse_duration_seconds(getattr(args, "hold_for", "30m"))
        max_lease = _parse_duration_seconds(getattr(args, "max_lease", "12h"))
        repeat_interval = _parse_duration_seconds(getattr(args, "notify_repeat_interval", "10m"))
        stage_code = cmd_run_stage(argparse.Namespace(run_id=args.run_id, executor="runpod"))
        if stage_code:
            return stage_code
        launch_code = cmd_run_launch(argparse.Namespace(run_id=args.run_id, executor="runpod", dry_run=False, image=getattr(args, "image", None)))
        if launch_code:
            return launch_code
        launched = True
        exit_code = _runpod_run_over_ssh(
            run_dir,
            ssh_timeout_sec=args.upload_timeout,
            job_timeout_sec=args.job_timeout,
            remote_notify="ntfy" in _notification_channels(notify_channels),
            max_lease_sec=max_lease,
        )
        download_code = _download_with_retries(args.run_id, args.download_attempts, args.download_interval)
        if download_code:
            raise ValueError("download did not complete before timeout")
        safe_to_stop = True
        state_word = "completed" if exit_code == 0 else "failed"
        stop_note = f" Pod is held for review and will be stopped after {hold_for} seconds." if hold_for else " Pod will be stopped now."
        notify_subject = f"Kura run {state_word}: {args.run_id}"
        notify_body = f"Run {args.run_id} {state_word} with exit code {exit_code}.{stop_note}"
        _notify(notify_channels, subject=notify_subject, body=notify_body)
        return exit_code
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        message = _safe_error(exc)
        print(f"cannot run remote job: {message}", file=sys.stderr)
        _notify(
            notify_channels,
            subject=f"Kura run controller failed: {args.run_id}",
            body=(
                f"Run {args.run_id} controller stopped before confirmed download/stop:\n"
                f"{message}\n\n"
                "The remote Pod may still be running and billing. Recover with:\n"
                f"uv run kura run reconcile {args.run_id}\n"
                f"uv run kura run download {args.run_id} --force\n"
                f"uv run kura run stop {args.run_id}"
            ),
        )
        return 1
    finally:
        if launched and safe_to_stop:
            if hold_for > 0:
                print(f"RunPod pod is held for review and will stop after {hold_for} seconds.", file=sys.stderr)
                try:
                    if notify_subject and notify_body:
                        _sleep_with_completion_reminders(delay_sec=hold_for, interval_sec=repeat_interval, channels=notify_channels, subject=notify_subject, body=notify_body)
                    else:
                        time.sleep(hold_for)
                except KeyboardInterrupt:
                    print("review hold interrupted; stopping RunPod pod now.", file=sys.stderr)
            cmd_run_stop(argparse.Namespace(run_id=args.run_id))
        elif launched:
            print(f"warning: leaving RunPod pod running because remote completion/download was not confirmed; inspect and stop explicitly with `uv run kura run stop {args.run_id}` after recovery", file=sys.stderr)


def cmd_run_launch(args: argparse.Namespace) -> int:
    run_dir = _run_path(args.run_id)
    try:
        locked = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        run_type = locked.get("type", "train")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot launch run: compile the run first ({_safe_error(exc)})", file=sys.stderr); return 1
    if run_type == "render":
        try:
            code = launch_render(_workspace(), run_dir, dry_run=args.dry_run)
            if not args.dry_run:
                state_word = "completed" if code == 0 else "failed"
                _notify(getattr(args, "notify", None), subject=f"Kura render {state_word}: {args.run_id}", body=f"Render {args.run_id} {state_word} with exit code {code}.")
            return code
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            message = _safe_error(exc)
            print(f"cannot launch render: {message}", file=sys.stderr)
            _notify(getattr(args, "notify", None), subject=f"Kura render failed: {args.run_id}", body=f"Render {args.run_id} failed before completion:\n{message}")
            return 1
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") == "running":
            raise ValueError("run already has a running realization; reconcile or stop it first")
        if status.get("state") not in ("compiled", "failed", "interrupted", "unknown", "launch_failed"):
            raise ValueError("run must be compiled before launch")
        spec = _command_for_backend(locked)
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    try:
        config = _workspace_config()
        backend_name = locked.get("backend", {}).get("name") if isinstance(locked.get("backend"), dict) else None
        image_name = _backend_image_name(backend_name)
        image = _image_config(image_name)
        if args.executor == "docker":
            docker = config.get("docker", {})
            mounts = docker.get("mounts", [])
            if not isinstance(mounts, list):
                raise ValueError("docker.mounts must be a list")
            launch_docker(workspace=_workspace(), run_dir=run_dir, spec=spec, image=image["local"], dockerfile=image["dockerfile"], mounts=mounts, gpu=bool(docker.get("gpu", False)), workspace_target=str(docker.get("workspace_target", "/workspace")), dry_run=args.dry_run)
        else:
            source_runpod_config = config.get("runpod", {})
            runpod_config = dict(source_runpod_config) if isinstance(source_runpod_config, dict) else {}
            remote_spec = dict(spec)
            remote_image = image["remote"]
            if image_name == "ai-toolkit" and isinstance(runpod_config.get("container_cwd"), str):
                remote_spec["cwd"] = runpod_config["container_cwd"]
            if image_name != "ai-toolkit":
                runpod_config.pop("template_id", None)
                backend_ports = runpod_config.get("backend_ports")
                if isinstance(backend_ports, dict) and isinstance(backend_ports.get(image_name), list):
                    runpod_config["ports"] = backend_ports[image_name]
                else:
                    runpod_config["ports"] = ["22/tcp"]
            default_image = runpod_config.get("default_image")
            if isinstance(default_image, dict) and isinstance(default_image.get(image_name), str):
                remote_image = default_image[image_name]
            if getattr(args, "image", None):
                remote_image = args.image
            launch_runpod(run_dir=run_dir, spec=remote_spec, image=remote_image, config=runpod_config, dry_run=args.dry_run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    return 0


def cmd_image_build(args: argparse.Namespace) -> int:
    try:
        image = _image_config(args.name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot build image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    ref_arg = "MUSUBI_TUNER_REF" if args.name == "musubi-tuner" else "AI_TOOLKIT_REF"
    default_ref = "main" if args.name == "musubi-tuner" else "548a286992261fbef40c380e82495d21fd3bca86"
    command = ["docker", "build", "--tag", image["local"], "--file", image["dockerfile"], "--build-arg", f"{ref_arg}={args.ref or default_ref}", image["context"]]
    try:
        result = _docker_run(command)
    except FileNotFoundError:
        print("docker command was not found", file=sys.stderr)
        return 1
    if result.returncode:
        return result.returncode
    inspect = _docker_run(["docker", "image", "inspect", "--format", "{{.Id}}", image["local"]], capture=True)
    print(inspect.stdout.strip() or image["local"])
    return 0


def cmd_image_inspect(args: argparse.Namespace) -> int:
    try:
        image = _image_config(args.name)
        result = _docker_run(["docker", "image", "inspect", image["local"]], capture=True)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot inspect image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if result.returncode:
        print(f"local image does not exist: {image['local']}")
        return 1
    metadata = json.loads(result.stdout)[0]
    print(json.dumps({"local_image": image["local"], "remote_image": image["remote"], "dockerfile": image["dockerfile"], "image_id": metadata.get("Id"), "created": metadata.get("Created"), "labels": metadata.get("Config", {}).get("Labels") or {}}, indent=2))
    return 0


def cmd_image_publish(args: argparse.Namespace) -> int:
    try:
        image = _image_config(args.name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot publish image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    commands = [["docker", "tag", image["local"], image["remote"]], ["docker", "push", image["remote"]]]
    if args.dry_run:
        print(json.dumps({"tag": commands[0], "push": commands[1]}, indent=2))
        return 0
    try:
        for command in commands:
            result = _docker_run(command)
            if result.returncode:
                return result.returncode
    except FileNotFoundError:
        print("docker command was not found", file=sys.stderr)
        return 1
    print(f"published {image['remote']}")
    return 0


def cmd_doctor_docker(_: argparse.Namespace) -> int:
    try:
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
            checks["local_image"] = _docker_run(["docker", "image", "inspect", image["local"]], capture=True).returncode == 0
            if not checks["local_image"]:
                diagnosis = f"Docker daemon is reachable but local image is missing. Run: kura image build ai-toolkit"
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
    cache_path = next((Path(item["source"]).expanduser() for item in mounts if isinstance(item, dict) and item.get("target") == "/root/.cache/huggingface"), None)
    cache: dict[str, Any] = {"path": str(cache_path) if cache_path else None, "exists": bool(cache_path and cache_path.exists())}
    if cache_path:
        try:
            filesystem_path = cache_path
            while not filesystem_path.exists() and filesystem_path != filesystem_path.parent:
                filesystem_path = filesystem_path.parent
            usage = shutil.disk_usage(filesystem_path)
            cache["free_bytes"] = usage.free
        except OSError:
            cache["free_bytes"] = None
    if checks["daemon_reachable"] and not diagnosis:
        diagnosis = "Docker is ready. Keep Docker Desktop and WSL updated; configure global memory/swap limits outside Kura only when the host requires them."
    print(json.dumps({**checks, "runtime": runtime, "huggingface_cache": cache, "docker_storage": docker_storage, "diagnostics": diagnostics, "diagnosis": diagnosis}, indent=2))
    return 0 if all(checks.values()) else 1


def cmd_doctor_runpod(_: argparse.Namespace) -> int:
    config = _workspace_config().get("runpod", {})
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
            request = __import__("urllib.request").request.Request("https://rest.runpod.io/v1/pods", headers={"Authorization": "Bearer " + os.environ[str(api_key_env)], "Content-Type": "application/json"})
            with __import__("urllib.request").request.urlopen(request, timeout=20) as response:
                all_pods = json.loads(response.read().decode("utf-8"))
            checks["rest_pods"] = True
            diagnostics["pods"] = _redact_secrets(all_pods)
            checks["pods_empty"] = all_pods == []
        except Exception as exc:
            diagnostics["pods_error"] = _redact_secret_text(str(exc))
        try:
            request = __import__("urllib.request").request.Request("https://rest.runpod.io/v1/networkvolumes", headers={"Authorization": "Bearer " + os.environ[str(api_key_env)], "Content-Type": "application/json"})
            with __import__("urllib.request").request.urlopen(request, timeout=20) as response:
                volumes = json.loads(response.read().decode("utf-8"))
            diagnostics["network_volumes"] = _redact_secrets(volumes)
            checks["network_volumes_empty"] = volumes == []
        except Exception as exc:  # read-only doctor; keep diagnosis broad.
            diagnostics["network_volumes_error"] = _redact_secret_text(str(exc))
    ok = bool(checks["runpodctl_command"] and checks["api_key"] and checks["pod_list"] and checks["rest_pods"] and checks["pods_empty"] is not False and checks["network_volumes_empty"] is not False)
    if ok:
        diagnosis = "RunPod CLI/API are ready."
    elif checks["pods_empty"] is False:
        diagnosis = "RunPod has Pods remaining; delete stopped/exited Pods if they should not persist."
    elif checks["network_volumes_empty"] is False:
        diagnosis = "RunPod has Network Volumes remaining; delete volumes that should not persist."
    else:
        diagnosis = "RunPod is not fully ready; inspect checks and diagnostics."
    print(json.dumps(_redact_secrets({"checks": checks, "diagnostics": diagnostics, "diagnosis": diagnosis}), indent=2))
    return 0 if ok else 1


def cmd_doctor_secrets(_: argparse.Namespace) -> int:
    config = Path.home() / ".docker" / "config.json"
    registries: list[str] = []
    try:
        registries = sorted(json.loads(config.read_text(encoding="utf-8")).get("auths", {}).keys())
    except (OSError, json.JSONDecodeError):
        pass
    print(json.dumps({"secrets": _secret_state(), "docker_login_registries": registries}, indent=2))
    return 0


def cmd_index_rebuild(_: argparse.Namespace) -> int:
    entries = []
    for run_file in sorted((_workspace() / "runs").glob("*/run.yaml")):
        try:
            run = _load_yaml(run_file)
            status = json.loads((run_file.parent / "status.json").read_text(encoding="utf-8"))
            entry = {"id": run.get("id"), "type": run.get("type", "train"), "experiment": run.get("experiment"), "created": run.get("created"), "state": status.get("state")}
            if run.get("type") == "render": entry["inputs"] = {"train_run": run.get("inputs", {}).get("train_run")}
            entries.append(entry)
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
            print(f"warning: skipped {run_file.parent.name}: {_safe_error(exc)}", file=sys.stderr)
    with (_workspace() / "index.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"rebuilt index: {len(entries)} runs")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    return run_textual_monitor(_workspace(), interval=args.interval, stale_after=args.stale_after, limit=args.limit)


def cmd_run_watch(args: argparse.Namespace) -> int:
    return run_textual_monitor(_workspace(), interval=args.interval, initial_run_id=args.run_id)


def main() -> None:
    _load_env_local()
    parser = argparse.ArgumentParser(prog="kura")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.set_defaults(func=cmd_init)
    monitor = sub.add_parser("monitor"); monitor.add_argument("--interval", type=float, default=2.0); monitor.add_argument("--stale-after", type=float, default=90.0); monitor.add_argument("--limit", type=int, default=30); monitor.set_defaults(func=cmd_monitor)
    dataset = sub.add_parser("dataset"); dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_sub.add_parser("validate"); validate.add_argument("dataset_dir"); validate.set_defaults(func=cmd_dataset_validate)
    run = sub.add_parser("run"); run_sub = run.add_subparsers(dest="run_command", required=True)
    new = run_sub.add_parser("new"); new.add_argument("--experiment", required=True); new.add_argument("--slug", required=True); new.set_defaults(func=cmd_run_new)
    compile_parser = run_sub.add_parser("compile"); compile_parser.add_argument("run_id"); compile_parser.set_defaults(func=cmd_run_compile)
    status = run_sub.add_parser("status"); status.add_argument("run_id"); status.set_defaults(func=cmd_run_status)
    stage = run_sub.add_parser("stage"); stage.add_argument("run_id"); stage.add_argument("--executor", default="runpod", choices=("runpod",)); stage.set_defaults(func=cmd_run_stage)
    logs = run_sub.add_parser("logs"); logs.add_argument("run_id"); logs.add_argument("--follow", action="store_true"); logs.set_defaults(func=cmd_run_logs)
    watch = run_sub.add_parser("watch"); watch.add_argument("run_id"); watch.add_argument("--interval", type=float, default=2.0); watch.set_defaults(func=cmd_run_watch)
    upload = run_sub.add_parser("upload"); upload.add_argument("run_id"); upload.set_defaults(func=cmd_run_upload)
    download = run_sub.add_parser("download"); download.add_argument("run_id"); download.add_argument("--force", action="store_true"); download.set_defaults(func=cmd_run_download)
    pull = run_sub.add_parser("pull")
    pull.add_argument("run_id")
    pull.add_argument("--step", type=int, help="Pull the checkpoint for one exact step")
    pull.add_argument("--since-step", type=int, help="Pull checkpoints at or after this step")
    pull.add_argument("--all", action="store_true", help="Pull every remote checkpoint output")
    pull.add_argument("--force", action="store_true", help="Copy even when a same-size local file already exists")
    pull.add_argument("--ssh-timeout", type=int, default=60)
    pull.set_defaults(func=cmd_run_pull)
    remote = run_sub.add_parser("remote")
    remote.add_argument("run_id")
    remote.add_argument("--upload-timeout", type=int, default=600)
    remote.add_argument("--job-timeout", type=int, default=0, help="Optional controller wait limit in seconds; 0 means wait until the remote job exits")
    remote.add_argument("--download-attempts", type=int, default=60)
    remote.add_argument("--download-interval", type=int, default=20)
    remote.add_argument("--image", help="Override the RunPod image for this run only")
    remote.add_argument("--hold-for", default="30m", help="Keep the Pod running for review after confirmed download, e.g. 30m. Defaults to 30m; use 0 to stop immediately.")
    remote.add_argument("--max-lease", default="12h", help="Best-effort Pod-side billing safety lease, e.g. 12h. Use 0 to disable.")
    remote.add_argument("--notify", help="Override notification channels: desktop,ntfy, or none. Defaults to auto-detection")
    remote.add_argument("--notify-repeat-interval", default="10m", help="Repeat completion notifications while the Pod is held for review; use 0 to disable")
    remote.set_defaults(func=cmd_run_remote)
    stop = run_sub.add_parser("stop"); stop.add_argument("run_id"); stop.set_defaults(func=cmd_run_stop)
    reconcile = run_sub.add_parser("reconcile"); reconcile.add_argument("run_id"); reconcile.set_defaults(func=cmd_run_reconcile)
    prune = run_sub.add_parser("prune")
    prune.add_argument("--keep", type=int, default=30)
    prune.add_argument("--states", default="completed,failed,interrupted,launch_failed")
    prune.add_argument("--outputs-only", action="store_true")
    prune.add_argument("--yes", action="store_true")
    prune.set_defaults(func=cmd_run_prune)
    launch = run_sub.add_parser("launch"); launch.add_argument("run_id"); launch.add_argument("--executor", default="docker", choices=("docker", "runpod")); launch.add_argument("--dry-run", action="store_true"); launch.add_argument("--image", help="Override the RunPod image for this run only"); launch.set_defaults(func=cmd_run_launch)
    render = sub.add_parser("render"); render_sub = render.add_subparsers(dest="render_command", required=True)
    render_new = render_sub.add_parser("new"); render_new.add_argument("--slug", required=True); render_new.set_defaults(func=cmd_render_new)
    render_compile = render_sub.add_parser("compile"); render_compile.add_argument("run_id"); render_compile.set_defaults(func=cmd_run_compile)
    render_launch = render_sub.add_parser("launch"); render_launch.add_argument("run_id"); render_launch.add_argument("--dry-run", action="store_true"); render_launch.add_argument("--notify", help="Override notification channels: desktop,ntfy, or none. Defaults to auto-detection"); render_launch.set_defaults(func=cmd_run_launch, executor="local")
    render_status = render_sub.add_parser("status"); render_status.add_argument("run_id"); render_status.set_defaults(func=cmd_run_status)
    image = sub.add_parser("image"); image_sub = image.add_subparsers(dest="image_command", required=True)
    build = image_sub.add_parser("build"); build.add_argument("name", choices=("ai-toolkit", "musubi-tuner")); build.add_argument("--ref"); build.set_defaults(func=cmd_image_build)
    inspect = image_sub.add_parser("inspect"); inspect.add_argument("name", choices=("ai-toolkit", "musubi-tuner")); inspect.set_defaults(func=cmd_image_inspect)
    publish = image_sub.add_parser("publish"); publish.add_argument("name", choices=("ai-toolkit", "musubi-tuner")); publish.add_argument("--dry-run", action="store_true"); publish.set_defaults(func=cmd_image_publish)
    doctor = sub.add_parser("doctor"); doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    doctor_docker = doctor_sub.add_parser("docker"); doctor_docker.set_defaults(func=cmd_doctor_docker)
    doctor_runpod = doctor_sub.add_parser("runpod"); doctor_runpod.set_defaults(func=cmd_doctor_runpod)
    doctor_secrets = doctor_sub.add_parser("secrets"); doctor_secrets.set_defaults(func=cmd_doctor_secrets)
    index = sub.add_parser("index"); index_sub = index.add_subparsers(dest="index_command", required=True)
    rebuild = index_sub.add_parser("rebuild"); rebuild.set_defaults(func=cmd_index_rebuild)
    args = parser.parse_args()
    raise SystemExit(args.func(args))
