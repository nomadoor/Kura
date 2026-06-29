#!/usr/bin/env python3
"""Run explicit, real one-step Musubi adapter smoke tests.

This is a developer acceptance test, not a release gate. It downloads or uses
real model files, launches Kura through its normal Docker executor, and requires
one actual optimizer step to complete.
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
    model_paths: dict[str, str]
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
}


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, env=env, timeout=timeout)


def workspace_root() -> Path:
    root = Path.cwd()
    if not (root / "workspace.yaml").is_file():
        raise SystemExit("workspace.yaml was not found; run from a Kura workspace root")
    return root


def write_run(root: Path, spec: SmokeSpec) -> str:
    run_id = f"{datetime.now():%Y%m%d-%H%M}_musubi-real-smoke-{spec.architecture}_{secrets.token_hex(2)}"
    run_dir = root / "runs" / run_id
    for relative in ("resolved", "logs", "metrics", "samples", "checkpoints", "outputs"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    override = {
        "architecture": spec.architecture,
        "model_paths": dict(spec.model_paths),
        **spec.extra_override,
    }
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
        "compute": {"executor": "docker", "gpu": "gpu"},
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
    outputs = list((run_dir / "outputs").glob("*.safetensors"))
    checks = {
        "completed": status.get("state") == "completed",
        "exit_code_zero": status.get("exit_code") == 0,
        "one_step": status.get("last_step") == 1 and status.get("total_steps") == 1,
        "script_seen": spec.expected_script in log,
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
    parser = argparse.ArgumentParser(description="Run a real one-step Musubi adapter smoke through Kura Docker.")
    parser.add_argument("architecture", choices=sorted(SPECS))
    parser.add_argument("--no-launch", action="store_true", help="Create and compile the run, but do not launch it")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Launch timeout in seconds")
    args = parser.parse_args()

    root = workspace_root()
    spec = SPECS[args.architecture]
    run_id = write_run(root, spec)
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
    result = run(["uv", "run", "kura", "run", "launch", run_id, "--executor", "docker", "--wait"], env=env, timeout=args.timeout)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    report = validate_result(root, run_id, spec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.returncode == 0 and report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
