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
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative = key[len(prefix) + 1:]
            if not relative or relative.endswith("/"):
                continue
            target = workspace / relative
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


def cmd_init(_: argparse.Namespace) -> int:
    root = Path.cwd()
    for relative in ("datasets", "experiments", "runs", "workflows", "promptsets", "backends", "executors", "cache/huggingface", "cache/models", "docker/ai-toolkit", "docker/musubi-tuner"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace.yaml"
    if not workspace.exists():
        dump_yaml(workspace, {"schema_version": 1, "name": root.name, "docker": {"images": {"ai-toolkit": {"local": "kura/ai-toolkit:dev", "remote": "nomadoor/kura-ai-toolkit:dev", "dockerfile": "docker/ai-toolkit/Dockerfile", "context": "."}, "musubi-tuner": {"local": "kura/musubi-tuner:dev", "remote": "nomadoor/kura-musubi-tuner:dev", "dockerfile": "docker/musubi-tuner/Dockerfile", "context": "."}}, "workspace_target": "/workspace", "gpu": True, "mounts": [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]}, "comfyui": {"endpoint": "http://127.0.0.1:8188", "lora_dir": "", "lora_stage_subdir": "Kura_tmp", "lora_stage_mode": "symlink", "lora_stage_cleanup": "remove_after_render"}, "runpod": {"default_image": {"ai-toolkit": "ostris/aitoolkit:latest", "musubi-tuner": "nomadoor/kura-musubi-tuner:dev"}, "template_id": "0fqzfjy6f3", "api_key_env": "RUNPOD_API_KEY", "storage_mode": "upload", "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "gpu_count": 1, "container_disk_gb": 150, "volume_in_gb": 0, "workspace_path": "/workspace", "container_cwd": "/app/ai-toolkit", "ports": ["8675/http", "22/tcp"], "cloud_type": "ANY", "gpu_type_priority": "custom", "interruptible": False}})
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
    runpod_object_job = root / "docker/ai-toolkit/kura_runpod_object_job.py"
    if not runpod_object_job.exists():
        runpod_object_job.write_text(RUNPOD_OBJECT_JOB_TEMPLATE, encoding="utf-8")
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
