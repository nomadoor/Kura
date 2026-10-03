"""`kura init`: create a workspace in the current directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from kura.workspace import WORKSPACE_SCHEMA_VERSION, dump_yaml


def cmd_init(_: argparse.Namespace) -> int:
    root = Path.cwd()
    for relative in ("datasets", "runs", "artifacts/training-state", "workflows", "promptsets", "cache/huggingface", "cache/models"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace.yaml"
    if not workspace.exists():
        dump_yaml(workspace, {"schema_version": WORKSPACE_SCHEMA_VERSION, "name": root.name, "storage": {"host_drive": "", "docker_data_drive": ""}, "docker": {"workspace_target": "/workspace", "gpu": True, "mounts": [{"source": "./cache/huggingface", "target": "/workspace/cache/huggingface", "mode": "rw"}]}, "comfyui": {"endpoint": "http://127.0.0.1:8188", "lora_dir": "", "lora_stage_subdir": "Kura_tmp", "lora_stage_mode": "symlink", "lora_stage_cleanup": "remove_after_render", "model_patches_dir": "", "model_patch_stage_subdir": "Kura_tmp", "model_patch_stage_mode": "symlink", "model_patch_stage_cleanup": "remove_after_render", "model_registry": {}, "runpod": {"gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "container_disk_gb": 80, "ports": ["22/tcp"]}}, "runpod": {"template_id": "0fqzfjy6f3", "api_key_env": "RUNPOD_API_KEY", "storage_mode": "upload", "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "gpu_count": 1, "container_disk_gb": 150, "volume_in_gb": 0, "workspace_path": "/workspace", "ports": ["8675/http", "22/tcp"], "backend_ports": {"comfyui": ["22/tcp"]}, "cloud_type": "ANY", "gpu_type_priority": "custom", "interruptible": False}})
    agents = root / "AGENTS.md"
    if not agents.exists():
        agents.write_text("# Repository Guidelines\n\nKura is file-first: use the CLI for mutations and keep secrets out of run artifacts.\n", encoding="utf-8")
    (root / "index.jsonl").touch(exist_ok=True)
    print(f"initialized workspace: {root}")
    print("next:")
    print("  1. Put a dataset under datasets/<id>/ with dataset.yaml and items.jsonl.")
    print("  2. Check it with: uv run kura dataset validate datasets/<id>")
    print("  3. Tell your AI agent what LoRA/render run you want and which model to use.")
    print("  4. Watch progress with: uv run kura monitor")
    return 0
