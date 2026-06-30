#!/usr/bin/env python3
"""Run explicit, real one-step Musubi adapter smoke tests.

This is a developer acceptance test, not a release gate. It downloads or uses
real model files, launches Kura through its normal Docker or RunPod executor,
and requires one actual optimizer step to complete.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SmokeSpec:
    architecture: str
    model_base: str
    dataset_id: str
    model_paths: dict[str, str] | None
    model_downloads: dict[str, dict[str, Any]] | None
    extra_override: dict[str, Any]
    params: dict[str, Any]
    expected_script: str
    expected_outputs: int = 1


SPECS: dict[str, SmokeSpec] = {
    "krea2": SmokeSpec(
        architecture="krea2",
        model_base="krea/Krea-2-Raw",
        dataset_id="flux2-klein-tiny",
        model_paths={
            "dit": "/workspace/cache/models/musubi/krea--Krea-2-Raw/dit/raw.safetensors",
            "vae": "/workspace/cache/models/musubi/Comfy-Org--Qwen-Image_ComfyUI/vae/split_files/vae/qwen_image_vae.safetensors",
            "text_encoder": "/workspace/cache/models/musubi/Comfy-Org--Qwen3-VL/text_encoder/text_encoders/qwen3vl_4b_bf16.safetensors",
        },
        model_downloads=None,
        extra_override={
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "extra_args": ["--blocks_to_swap", "26"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="krea2_train_network.py",
        expected_outputs=2,
    ),
    "qwen_image": SmokeSpec(
        architecture="qwen_image",
        model_base="Comfy-Org/Qwen-Image_ComfyUI",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/diffusion_models/qwen_image_bf16.safetensors"},
            "text_encoder": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"},
            "vae": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/vae/qwen_image_vae.safetensors"},
        },
        extra_override={
            "model_version": "original",
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "fp8_vl": True,
            "extra_args": ["--blocks_to_swap", "45"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="qwen_image_train_network.py",
        expected_outputs=2,
    ),
    "wan": SmokeSpec(
        architecture="wan",
        model_base="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"},
            "vae": {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/vae/wan_2.1_vae.safetensors"},
            "t5": {"repo": "Wan-AI/Wan2.1-I2V-14B-720P", "filename": "models_t5_umt5-xxl-enc-bf16.pth"},
        },
        extra_override={
            "task": "t2v-1.3B",
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="wan_train_network.py",
        expected_outputs=2,
    ),
}


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, env=env, timeout=timeout)


def workspace_root() -> Path:
    root = Path.cwd()
    if not (root / "workspace.yaml").is_file():
        raise SystemExit("workspace.yaml was not found; run from a Kura workspace root")
    return root


def write_run(root: Path, spec: SmokeSpec, *, executor: str, gpu: str) -> str:
    run_id = f"{datetime.now():%Y%m%d-%H%M}_musubi-real-smoke-{spec.architecture}_{secrets.token_hex(2)}"
    run_dir = root / "runs" / run_id
    for relative in ("resolved", "logs", "metrics", "samples", "checkpoints", "outputs"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    override: dict[str, Any] = {
        "architecture": spec.architecture,
        **spec.extra_override,
    }
    if spec.model_paths is not None:
        override["model_paths"] = dict(spec.model_paths)
    if spec.model_downloads is not None:
        override["model_downloads"] = {key: dict(value) for key, value in spec.model_downloads.items()}
    run_yaml = {
        "schema_version": 1,
        "id": run_id,
        "type": "train",
        "experiment": "musubi-real-smoke",
        "created": datetime.now().astimezone().isoformat(),
        "created_by": "agent",
        "parent_run": None,
        "intent": "real one-step Musubi adapter smoke with actual model files",
        "backend": {"name": "musubi-tuner", "version": None, "adapter_version": 1},
        "model": {"base": spec.model_base, "revision": None},
        "datasets": [{"id": spec.dataset_id, "digest": None, "role": None}],
        "params": dict(spec.params),
        "backend_overrides": {"musubi-tuner": override},
        "compute": {"executor": executor, "gpu": gpu},
        "sampling": {"prompts": [], "cadence_steps": None},
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "plan.md").write_text("# Real smoke plan\n\n", encoding="utf-8")
    (run_dir / "notes.md").write_text("# Notes\n\n", encoding="utf-8")
    for relative in ("logs/events.jsonl", "metrics/metrics.jsonl", "samples/samples.jsonl"):
        (run_dir / relative).touch()
    return run_id


def validate_result(root: Path, run_id: str, spec: SmokeSpec) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    log = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8", errors="replace")
    command_path = run_dir / "resolved" / "musubi" / "command.json"
    command_text = command_path.read_text(encoding="utf-8", errors="replace") if command_path.exists() else ""
    outputs = list((run_dir / "outputs").glob("*.safetensors"))
    outputs.extend((run_dir / "downloads").glob("**/outputs/*.safetensors"))
    checks = {
        "completed": status.get("state") == "completed",
        "exit_code_zero": status.get("exit_code") == 0,
        "one_step": status.get("last_step") == 1 and status.get("total_steps") == 1,
        "script_seen": spec.expected_script in log or spec.expected_script in command_text,
        "loss_seen": "avr_loss=" in log,
        "outputs_seen": len(outputs) >= spec.expected_outputs,
    }
    return {
        "run_id": run_id,
        "architecture": spec.architecture,
        "status": status,
        "outputs": [str(path.relative_to(run_dir)) for path in outputs],
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real one-step Musubi adapter smoke through Kura Docker or RunPod.")
    parser.add_argument("architecture", choices=sorted(SPECS))
    parser.add_argument("--executor", choices=("docker", "runpod"), default="docker")
    parser.add_argument("--gpu", default="gpu", help="GPU selector recorded in run.yaml; RunPod accepts a GPU type such as 'NVIDIA A40'")
    parser.add_argument("--no-launch", action="store_true", help="Create and compile the run, but do not launch it")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Launch timeout in seconds")
    parser.add_argument("--hold-for", default="0", help="RunPod review hold after successful download")
    parser.add_argument("--max-lease", default="4h", help="RunPod max lease safety fuse")
    args = parser.parse_args()

    root = workspace_root()
    spec = SPECS[args.architecture]
    run_id = write_run(root, spec, executor=args.executor, gpu=args.gpu)
    env = dict(os.environ)
    env["KURA_NOTIFY"] = "none"
    for command in (
        ["uv", "run", "kura", "run", "compile", run_id],
        ["uv", "run", "kura", "run", "plan", run_id],
    ):
        result = run(command, env=env, timeout=120)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return result.returncode
    if args.no_launch:
        print(json.dumps({"run_id": run_id, "launched": False}, indent=2))
        return 0
    if args.executor == "docker":
        command = ["uv", "run", "kura", "run", "launch", run_id, "--executor", "docker", "--wait"]
    else:
        command = [
            "uv", "run", "kura", "run", "remote", run_id,
            "--hold-for", args.hold_for,
            "--max-lease", args.max_lease,
            "--job-timeout", str(int(args.timeout)),
            "--download-attempts", "20",
            "--download-interval", "30",
            "--notify", "none",
        ]
    result = run(command, env=env, timeout=args.timeout + 600)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    report = validate_result(root, run_id, spec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.returncode == 0 and report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
