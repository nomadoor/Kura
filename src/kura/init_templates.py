"""Workspace initialization templates and command."""

from __future__ import annotations

import argparse
from pathlib import Path

from kura.workspace import dump_yaml


RUNPOD_OBJECT_JOB_TEMPLATE = '''"""Run a Kura job on ephemeral RunPod storage with S3-compatible staging.

The container disk is disposable. This wrapper downloads the staged Kura
workspace prefix into /workspace, runs the backend command, and uploads the run
directory back to the same object store before exiting with the backend code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=env("KURA_OBJECT_ENDPOINT_URL"),
        region_name=os.environ.get("KURA_OBJECT_REGION", "auto"),
        aws_access_key_id=env("KURA_OBJECT_ACCESS_KEY_ID"),
        aws_secret_access_key=env("KURA_OBJECT_SECRET_ACCESS_KEY"),
        config=Config(retries={"max_attempts": 10, "mode": "standard"}, read_timeout=7200),
    )


def download_prefix(client, bucket: str, prefix: str, workspace: Path) -> int:
    count = 0
    workspace = workspace.resolve()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative = key[len(prefix) + 1:]
            if not relative or relative.endswith("/"):
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"refusing unsafe object key: {key}")
            target = (workspace / relative_path).resolve()
            if not target.is_relative_to(workspace):
                raise RuntimeError(f"refusing object key outside workspace: {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    return count


def upload_tree(client, bucket: str, prefix: str, workspace: Path, relative_root: str) -> int:
    root = workspace / relative_root
    if not root.exists():
        return 0
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            key = f"{prefix}/{path.relative_to(workspace).as_posix()}"
            client.upload_file(str(path), bucket, key)
            count += 1
    return count


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("backend command is required")

    bucket = env("KURA_OBJECT_BUCKET")
    prefix = env("KURA_OBJECT_PREFIX").strip("/")
    run_id = env("KURA_RUN_ID")
    workspace = Path(os.environ.get("KURA_WORKSPACE", "/workspace"))
    log_path = Path(os.environ.get("KURA_LOG_PATH", str(workspace / "runs" / run_id / "logs" / "stdout.log")))
    events_path = workspace / "runs" / run_id / "logs" / "events.jsonl"
    client = s3_client()

    exit_code = 1
    started = datetime.now().astimezone().isoformat()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        downloaded = download_prefix(client, bucket, prefix, workspace)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(events_path, {"event": "remote_workspace_downloaded", "timestamp": started, "object_prefix": prefix, "files": downloaded})
        with log_path.open("ab") as log:
            result = subprocess.run(command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
        exit_code = int(result.returncode)
    except Exception:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        exit_code = 1
    finally:
        ended = datetime.now().astimezone().isoformat()
        exit_record = workspace / "runs" / run_id / "realizations" / f"remote-exit-{ended.replace(':', '').replace('.', '-')}.json"
        exit_record.parent.mkdir(parents=True, exist_ok=True)
        exit_record.write_text(json.dumps({"event": "remote_exit", "timestamp": ended, "exit_code": exit_code}, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        append_jsonl(events_path, {"event": "remote_workspace_uploading", "timestamp": ended, "object_prefix": prefix, "exit_code": exit_code})
        try:
            upload_tree(client, bucket, prefix, workspace, f"runs/{run_id}")
        except Exception:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
'''


COMFYUI_DOCKERFILE_TEMPLATE = """ARG PYTORCH_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_BASE}

ARG COMFYUI_REF=50e5270b86765bac2da70248d61050abba72b19f
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV COMFYUI_ROOT=/opt/ComfyUI

RUN apt-get update \\
    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFYUI_ROOT" \\
    && cd "$COMFYUI_ROOT" \\
    && git checkout "$COMFYUI_REF" \\
    && pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY docker/comfyui/kura_comfy_prepare.py /opt/kura_comfy_prepare.py

WORKDIR /workspace
EXPOSE 8188
CMD ["python", "/opt/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"]
"""


COMFYUI_PREPARE_TEMPLATE = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MODEL_INPUTS = {
    "CheckpointLoaderSimple": (("checkpoints", "ckpt_name"),),
    "VAELoader": (("vae", "vae_name"),),
    "CLIPLoader": (("clip", "clip_name"),),
    "DualCLIPLoader": (("clip", "clip_name1"), ("clip", "clip_name2")),
    "TripleCLIPLoader": (("clip", "clip_name1"), ("clip", "clip_name2"), ("clip", "clip_name3")),
    "UNETLoader": (("diffusion_models", "unet_name"),),
    "ControlNetLoader": (("controlnet", "control_net_name"),),
}
MODEL_DIRS = {"checkpoints": "checkpoints", "vae": "vae", "clip": "clip", "diffusion_models": "diffusion_models", "controlnet": "controlnet"}
MODEL_REGISTRY = {"checkpoints": {"v1-5-pruned-emaonly-fp16.safetensors": {"repo": "Comfy-Org/stable-diffusion-v1-5-archive", "filename": "v1-5-pruned-emaonly-fp16.safetensors"}}}


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"refusing unsafe model target: {relative}") from exc
    return candidate


def _required_models(workflow: dict[str, Any]) -> list[dict[str, str]]:
    refs, seen = [], set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type, inputs = node.get("class_type"), node.get("inputs")
        if not isinstance(class_type, str) or not isinstance(inputs, dict):
            continue
        for model_type, input_name in MODEL_INPUTS.get(class_type, ()):
            value = inputs.get(input_name)
            if not isinstance(value, str) or not value:
                continue
            key = (model_type, value)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"node": str(node_id), "class_type": class_type, "input": input_name, "type": model_type, "name": value})
    return refs


def _resolve(ref: dict[str, str], registry: dict[str, Any]) -> dict[str, str] | None:
    section = registry.get(ref["type"], {})
    if not isinstance(section, dict):
        return None
    entry = section.get(ref["name"])
    if not isinstance(entry, dict):
        return None
    repo = entry.get("repo") or entry.get("repo_id")
    filename = entry.get("filename") or entry.get("file") or ref["name"]
    if not repo or not filename:
        return None
    spec = {**ref, "repo": repo, "filename": filename, "target_dir": MODEL_DIRS.get(ref["type"], ref["type"]), "target_name": entry.get("target_name") or ref["name"]}
    if entry.get("revision"):
        spec["revision"] = entry["revision"]
    if entry.get("subfolder"):
        spec["subfolder"] = entry["subfolder"]
    return spec


def prepare(workflow: dict[str, Any], *, comfyui_root: Path, cache_dir: Path | None, registry: dict[str, Any]) -> list[dict[str, str]]:
    models_root = comfyui_root / "models"
    specs, unknown = [], []
    for ref in _required_models(workflow):
        spec = _resolve(ref, registry)
        if spec is None:
            unknown.append(ref)
        else:
            specs.append(spec)
    if unknown:
        raise RuntimeError("unknown ComfyUI model loader entries: " + ", ".join(f"{item['class_type']}.{item['input']}={item['name']}" for item in unknown))
    if specs and cache_dir is None:
        raise ValueError("ComfyUI model prepare requires HF_HUB_CACHE or --cache-dir before downloading models")
    if specs and cache_dir is not None:
        workspace = Path(os.environ.get("KURA_WORKSPACE", "/workspace")).resolve()
        try:
            cache_dir.resolve().relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"ComfyUI model prepare cache_dir must be under {workspace}: {cache_dir}") from exc
    from huggingface_hub import hf_hub_download

    for spec in specs:
        downloaded = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"], subfolder=spec.get("subfolder"), revision=spec.get("revision"), cache_dir=str(cache_dir), token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None)
        target = _safe_child(models_root, f"{spec['target_dir']}/{spec['target_name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == Path(downloaded).resolve():
                continue
            raise ValueError(f"refusing to replace existing ComfyUI model target: {target}")
        os.symlink(downloaded, target)
        print(json.dumps({"event": "model_ready", "model": spec["name"], "target": str(target), "source": str(downloaded)}, ensure_ascii=False), flush=True)
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_json")
    parser.add_argument("--registry-json")
    parser.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    args = parser.parse_args()
    workflow = json.loads(Path(args.workflow_json).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise ValueError("workflow_json must contain a ComfyUI API workflow object")
    registry = MODEL_REGISTRY
    if args.registry_json:
        loaded = json.loads(Path(args.registry_json).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("registry_json must contain a model registry object")
        registry = loaded
    specs = prepare(workflow, comfyui_root=Path(args.comfyui_root), cache_dir=Path(args.cache_dir) if args.cache_dir else None, registry=registry)
    print(json.dumps({"event": "models_prepared", "count": len(specs), "models": specs}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def cmd_init(_: argparse.Namespace) -> int:
    root = Path.cwd()
    for relative in ("datasets", "runs", "workflows", "promptsets", "cache/huggingface", "cache/models", "docker/ai-toolkit", "docker/musubi-tuner", "docker/comfyui"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace.yaml"
    if not workspace.exists():
        dump_yaml(workspace, {"schema_version": 1, "name": root.name, "storage": {"host_drive": "", "docker_data_drive": ""}, "docker": {"images": {"ai-toolkit": {"local": "nomadoor/kura-ai-toolkit:dev", "remote": "nomadoor/kura-ai-toolkit:dev", "dockerfile": "docker/ai-toolkit/Dockerfile", "context": "."}, "musubi-tuner": {"local": "nomadoor/kura-musubi-tuner:dev", "remote": "nomadoor/kura-musubi-tuner:dev", "dockerfile": "docker/musubi-tuner/Dockerfile", "context": "."}, "comfyui": {"local": "nomadoor/kura-comfyui:dev", "remote": "nomadoor/kura-comfyui:dev", "dockerfile": "docker/comfyui/Dockerfile", "context": "."}}, "workspace_target": "/workspace", "gpu": True, "mounts": [{"source": "./cache/huggingface", "target": "/workspace/cache/huggingface", "mode": "rw"}]}, "comfyui": {"endpoint": "http://127.0.0.1:8188", "lora_dir": "", "lora_stage_subdir": "Kura_tmp", "lora_stage_mode": "symlink", "lora_stage_cleanup": "remove_after_render", "model_registry": {}, "runpod": {"gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "container_disk_gb": 80, "ports": ["22/tcp"]}}, "runpod": {"default_image": {"ai-toolkit": "ostris/aitoolkit:0.10.22", "musubi-tuner": "nomadoor/kura-musubi-tuner:dev", "comfyui": "nomadoor/kura-comfyui:dev"}, "template_id": "0fqzfjy6f3", "api_key_env": "RUNPOD_API_KEY", "storage_mode": "upload", "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "gpu_count": 1, "container_disk_gb": 150, "volume_in_gb": 0, "workspace_path": "/workspace", "container_cwd": "/app/ai-toolkit", "ports": ["8675/http", "22/tcp"], "backend_ports": {"comfyui": ["22/tcp"]}, "cloud_type": "ANY", "gpu_type_priority": "custom", "interruptible": False}})
    agents = root / "AGENTS.md"
    if not agents.exists():
        agents.write_text("# Repository Guidelines\n\nKura is file-first: use the CLI for mutations and keep secrets out of run artifacts.\n", encoding="utf-8")
    (root / "index.jsonl").touch(exist_ok=True)
    dockerfile = root / "docker/ai-toolkit/Dockerfile"
    if not dockerfile.exists():
        dockerfile.write_text(
            "ARG AI_TOOLKIT_IMAGE=ostris/aitoolkit:0.10.22@sha256:5a810f50de920aaa3439487959ae392bf0d1458345baddee24a7bf33787c0438\nFROM ${AI_TOOLKIT_IMAGE}\n\nCOPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py\nRUN ln -sfn /app/ai-toolkit /opt/ai-toolkit\nWORKDIR /workspace\nCMD [\"/start.sh\"]\n",
            encoding="utf-8",
        )
    runpod_object_job = root / "docker/ai-toolkit/kura_runpod_object_job.py"
    if not runpod_object_job.exists():
        runpod_object_job.write_text(RUNPOD_OBJECT_JOB_TEMPLATE, encoding="utf-8")
    musubi_dockerfile = root / "docker/musubi-tuner/Dockerfile"
    if not musubi_dockerfile.exists():
        musubi_dockerfile.write_text(
            "ARG PYTORCH_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime\nFROM ${PYTORCH_BASE}\n\nARG MUSUBI_TUNER_REF=v0.3.4\nENV DEBIAN_FRONTEND=noninteractive\nENV PIP_DEFAULT_TIMEOUT=180\nENV PIP_RETRIES=5\n\nRUN apt-get update \\\n    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \\\n    && rm -rf /var/lib/apt/lists/*\n\nRUN git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner \\\n    && cd /opt/musubi-tuner \\\n    && git checkout \"$MUSUBI_TUNER_REF\" \\\n    && pip install --no-cache-dir -e .\n\nCOPY docker/musubi-tuner/patch_flux2_diffusers_vae.py /tmp/patch_flux2_diffusers_vae.py\nRUN python /tmp/patch_flux2_diffusers_vae.py \\\n    && rm /tmp/patch_flux2_diffusers_vae.py\n\nCOPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py\nWORKDIR /workspace\nCMD [\"sleep\", \"infinity\"]\n",
            encoding="utf-8",
        )
    musubi_patch = root / "docker/musubi-tuner/patch_flux2_diffusers_vae.py"
    if not musubi_patch.exists():
        musubi_patch.write_text(
            "from pathlib import Path\n\n\nTARGET = Path(\"/opt/musubi-tuner/src/musubi_tuner/flux_2/flux2_utils.py\")\n\n\nOLD = \"\"\"    logger.info(f\\\"Loading state dict from {ckpt_path}\\\")\n    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)\n    info = ae.load_state_dict(sd, strict=True, assign=True)\n\"\"\"\n\n\nNEW = \"\"\"    logger.info(f\\\"Loading state dict from {ckpt_path}\\\")\n    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)\n    if any(\\\".down_blocks.\\\" in key or \\\".up_blocks.\\\" in key or key.endswith(\\\"conv_norm_out.weight\\\") for key in sd):\n        logger.info(\\\"Converting Diffusers-layout Flux2 VAE state dict\\\")\n        from musubi_tuner.ideogram4.ideogram4_autoencoder import convert_diffusers_state_dict\n\n        sd = convert_diffusers_state_dict(sd)\n    info = ae.load_state_dict(sd, strict=True, assign=True)\n\"\"\"\n\n\ndef main() -> None:\n    text = TARGET.read_text()\n    if NEW in text:\n        return\n    if OLD not in text:\n        raise SystemExit(f\"expected load_ae block not found in {TARGET}\")\n    TARGET.write_text(text.replace(OLD, NEW))\n\n\nif __name__ == \"__main__\":\n    main()\n",
            encoding="utf-8",
        )
    comfyui_dockerfile = root / "docker/comfyui/Dockerfile"
    if not comfyui_dockerfile.exists():
        comfyui_dockerfile.write_text(COMFYUI_DOCKERFILE_TEMPLATE, encoding="utf-8")
    comfyui_prepare = root / "docker/comfyui/kura_comfy_prepare.py"
    if not comfyui_prepare.exists():
        comfyui_prepare.write_text(COMFYUI_PREPARE_TEMPLATE, encoding="utf-8")
    print(f"initialized workspace: {root}")
    print("next:")
    print("  1. Put a dataset under datasets/<id>/ with dataset.yaml and items.jsonl.")
    print("  2. Check it with: uv run kura dataset validate datasets/<id>")
    print("  3. Tell your AI agent what LoRA/render run you want and which model to use.")
    print("  4. Watch progress with: uv run kura monitor")
    return 0
