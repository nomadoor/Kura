#!/usr/bin/env python3
"""Prepare, plan, launch, and validate bounded sd-scripts Tier 1 smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SmokeSpec:
    architecture: str
    mode: str
    roles: tuple[str, ...]
    expected_script: str
    output_kind: str
    paired: bool = False


SPECS = {
    "sd15": SmokeSpec("sd15", "lora", ("base",), "train_network.py", "lora"),
    "sdxl": SmokeSpec("sdxl", "lora", ("base",), "sdxl_train_network.py", "lora"),
    "flux1": SmokeSpec("flux1", "lora", ("dit", "clip_l", "t5xxl", "ae"), "flux_train_network.py", "lora"),
    "anima-lora": SmokeSpec("anima", "lora", ("dit", "qwen3", "vae"), "anima_train_network.py", "lora"),
    "anima-lllite": SmokeSpec("anima", "controlnet_lllite", ("dit", "qwen3", "vae"), "anima_train_control_net_lllite.py", "anima-lllite", paired=True),
}


def run(command: list[str], *, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["KURA_NOTIFY"] = "none"
    return subprocess.run(command, text=True, capture_output=True, check=False, env=env, timeout=timeout)


def workspace_root() -> Path:
    root = Path.cwd()
    if not (root / "workspace.yaml").is_file():
        raise SystemExit("workspace.yaml was not found; run this harness from a Kura workspace root")
    return root


def _png(width: int = 256, height: int = 256, *, conditioning: bool = False) -> bytes:
    rows = b"".join(
        b"\x00" + bytes(channel for x in range(width) for channel in ((255 if conditioning else x % 256), y % 256, (x + y) % 256))
        for y in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def ensure_dataset(root: Path, *, paired: bool) -> str:
    dataset_id = "sd-scripts-lllite-smoke" if paired else "sd-scripts-lora-smoke"
    dataset = root / "datasets" / dataset_id
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "0001.png").write_bytes(_png())
    (images / "0001.txt").write_text("a tiny synthetic smoke image\n", encoding="utf-8")
    item: dict[str, Any] = {"id": "0001", "path": "images/0001.png", "caption": "a tiny synthetic smoke image", "role": "target"}
    layout: dict[str, Any] = {"root": "images", "image_dir": "images"}
    if paired:
        conditioning = dataset / "conditioning"
        conditioning.mkdir(parents=True, exist_ok=True)
        (conditioning / "0001.png").write_bytes(_png(conditioning=True))
        item["control_path"] = "conditioning/0001.png"
        layout["control_dir"] = "conditioning"
    (dataset / "dataset.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "id": dataset_id, "modality": "image-pair" if paired else "image", "description": "Generated one-item sd-scripts real-smoke dataset.", "source": [], "caption": {"strategy": "manual", "version": 1}, "stats": {"count": 1}, "layout": layout, "digest": {"raw": None, "dataset": None}}, sort_keys=False),
        encoding="utf-8",
    )
    (dataset / "items.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataset_id


def dataset_identity(root: Path, dataset_id: str) -> list[dict[str, Any]]:
    dataset = root / "datasets" / dataset_id
    return [
        {"path": path.relative_to(dataset).as_posix(), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(dataset.rglob("*")) if path.is_file()
    ]


def load_models(path: Path, spec: SmokeSpec) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("models file must be a role-to-path/download mapping")
    paths: dict[str, str] = {}
    downloads: dict[str, dict[str, Any]] = {}
    for role, value in payload.items():
        if isinstance(value, str) and value:
            paths[str(role)] = value
        elif isinstance(value, dict):
            downloads[str(role)] = value
        else:
            raise SystemExit(f"invalid model role value: {role}")
    missing = [role for role in spec.roles if role not in paths and role not in downloads]
    if missing:
        raise SystemExit("models file is missing required role(s): " + ", ".join(missing))
    return paths, downloads


def write_run(root: Path, selector: str, spec: SmokeSpec, *, models_file: Path, executor: str, gpu: str) -> tuple[str, str]:
    dataset_id = ensure_dataset(root, paired=spec.paired)
    paths, downloads = load_models(models_file, spec)
    run_id = f"{datetime.now():%Y%m%d-%H%M}_sd-scripts-smoke-{selector}_{secrets.token_hex(2)}"
    run_dir = root / "runs" / run_id
    for relative in ("resolved", "logs", "metrics", "samples", "checkpoints", "outputs", "realizations"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    subset: dict[str, Any] = {"dataset_id": dataset_id, "image_subdir": "images", "num_repeats": 1}
    if spec.paired:
        subset["conditioning_subdir"] = "conditioning"
    resolution = [256, 256] if spec.architecture == "sd15" else [512, 512]
    config: dict[str, Any] = {
        "architecture": spec.architecture,
        "mode": spec.mode,
        "model_paths": paths,
        "model_downloads": downloads,
        "dataset_config": {"general": {"resolution": resolution, "caption_extension": ".txt"}, "datasets": [{"batch_size": 1, "subsets": [subset]}]},
        "learning_rate": 0.0001,
        "optimizer_type": "AdamW8bit",
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "output_name": run_id,
    }
    if spec.mode == "lora":
        config.update({"network_dim": 4, "network_alpha": 1})
    if spec.architecture == "sdxl":
        # Bounded 12 GiB smoke: rank/resolution are deliberately below the
        # upstream 1024px/rank-32 example; the training topology follows its
        # U-Net-only and cache recommendations.
        config.update({
            "network_dim": 8,
            "network_alpha": 4,
            "network_train_unet_only": True,
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs_to_disk": True,
            "disk_cache_estimate_gb": 1,
        })
    if spec.architecture == "flux1":
        # v0.11.1 documents blocks_to_swap=16 for a 12 GiB GPU with fp8_base.
        config.update({
            "network_dim": 16,
            "network_alpha": 1,
            "fp8_base": True,
            "blocks_to_swap": 16,
            "network_train_unet_only": True,
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs_to_disk": True,
            "disk_cache_estimate_gb": 1,
            "timestep_sampling": "flux_shift",
            "guidance_scale": 1.0,
            "model_prediction_type": "raw",
        })
    if spec.architecture == "anima":
        if spec.mode == "lora":
            config.update({
                "network_dim": 8,
                "network_alpha": 1,
                "learning_rate": 0.0001,
                "timestep_sampling": "sigmoid",
                "discrete_flow_shift": 1.0,
                "network_train_unet_only": True,
            })
        else:
            config.update({
                "learning_rate": 0.00005,
                "timestep_sampling": "shift",
                "discrete_flow_shift": 3.0,
                "attn_mode": "sdpa",
                "cond_emb_dim": 32,
                "lllite_cond_dim": 64,
                "lllite_cond_resblocks": 1,
                "lllite_mlp_dim": 64,
                "lllite_target_layers": "self_attn_q",
            })
        config.update({
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs_to_disk": True,
            "disk_cache_estimate_gb": 1,
            "qwen_image_vae_2d": True,
        })
    run_yaml = {
        "schema_version": 2,
        "id": run_id,
        "type": "train",
        "experiment": "sd-scripts-real-smoke",
        "created": datetime.now().astimezone().isoformat(),
        "created_by": "agent",
        "parent_run": None,
        "intent": f"real one-step sd-scripts Tier 1 smoke: {selector}",
        "model": {"base": selector, "revision": None},
        "datasets": [{"id": dataset_id, "digest": None, "role": "conditioning" if spec.paired else None}],
        "recipe": {"steps": 1, "seed": 42},
        "backend": {"name": "sd-scripts", "version": "0.11.1", "adapter_version": 1, "config": config},
        "compute": {"executor": executor, "gpu": gpu},
        "safety": {"allow_large_model_downloads": True},
        "sampling": {"prompts": [], "cadence_steps": None},
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "plan.md").write_text("# sd-scripts real smoke plan\n", encoding="utf-8")
    (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for relative in ("logs/events.jsonl", "metrics/metrics.jsonl", "samples/samples.jsonl"):
        (run_dir / relative).touch()
    (run_dir / "realizations" / "dataset-before.json").write_text(json.dumps(dataset_identity(root, dataset_id), indent=2) + "\n", encoding="utf-8")
    return run_id, dataset_id


def _safetensors_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header.get("__metadata__") if isinstance(header, dict) and isinstance(header.get("__metadata__"), dict) else {}


def validate_result(root: Path, run_id: str, dataset_id: str, spec: SmokeSpec) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    command = (run_dir / "resolved" / "backend-command.lock.json").read_text(encoding="utf-8", errors="replace")
    outputs = sorted((run_dir / "outputs").glob("*.safetensors"))
    before = json.loads((run_dir / "realizations" / "dataset-before.json").read_text(encoding="utf-8"))
    after = dataset_identity(root, dataset_id)
    (run_dir / "realizations" / "dataset-after.json").write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    metadata_ok = True
    if spec.output_kind == "anima-lllite":
        metadata_ok = bool(outputs) and _safetensors_metadata(outputs[0]).get("lllite.version") == "2"
    observation: dict[str, Any] = {}
    observation_ref = status.get("last_observation")
    if isinstance(observation_ref, str):
        observation_path = run_dir / observation_ref
        if observation_path.is_file():
            loaded = json.loads(observation_path.read_text(encoding="utf-8"))
            observation = loaded if isinstance(loaded, dict) else {}
    local_docker = not isinstance(status.get("pod_id"), str)
    stdout_path = run_dir / "logs" / "stdout.log"
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    losses = [float(match) for match in re.findall(r"(?:\bloss:\s*|\bavr_loss=)([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", stdout_text, flags=re.IGNORECASE)]
    checks = {
        "completed": status.get("state") == "completed" and status.get("exit_code") == 0,
        "one_step": status.get("last_step") == 1 and status.get("total_steps") == 1,
        "entrypoint": spec.expected_script in command,
        "output": bool(outputs),
        "output_metadata": metadata_ok,
        "finite_loss": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "dataset_unchanged": before == after,
        "docker_terminal_observed": not local_docker or (observation.get("state") == "completed" and observation.get("exit_code") == 0 and isinstance(observation.get("container_id"), str)),
        "recovery_complete": status.get("host") != "runpod" or status.get("recovery_required") is False,
        "pod_stopped": status.get("host") != "runpod" or isinstance(status.get("pod_stopped_at"), str),
    }
    return {"run_id": run_id, "selector": f"{spec.architecture}/{spec.mode}", "outputs": [path.relative_to(run_dir).as_posix() for path in outputs], "checks": checks, "ok": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and optionally launch an sd-scripts Tier 1 real smoke through Kura.")
    parser.add_argument("selector", choices=sorted(SPECS))
    parser.add_argument("--models", type=Path, help="YAML mapping each required role to a path or immutable download mapping")
    parser.add_argument("--run-id", help="Reuse the already planned run for an approved launch")
    parser.add_argument("--executor", choices=("docker", "runpod"), default="docker")
    parser.add_argument("--gpu", default="gpu")
    parser.add_argument("--launch", action="store_true", help="Launch after doctor, compile, and plan")
    parser.add_argument("--yes", action="store_true", help="Carry the user's explicit launch approval")
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.launch and not args.yes:
        parser.error("--launch requires --yes after the user has approved the displayed plan")
    if args.launch and not args.run_id:
        parser.error("--launch requires --run-id so the approved frozen run is reused")
    if not args.run_id and args.models is None:
        parser.error("--models is required when preparing a new smoke run")
    root = workspace_root()
    for command in (["uv", "run", "kura", "doctor", "disk"], ["uv", "run", "kura", "doctor", "sd-scripts"]):
        result = run(command, timeout=900)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            return result.returncode
    spec = SPECS[args.selector]
    if args.run_id:
        run_id = args.run_id
        run_yaml = yaml.safe_load((root / "runs" / run_id / "run.yaml").read_text(encoding="utf-8"))
        if not isinstance(run_yaml, dict) or run_yaml.get("backend", {}).get("config", {}).get("architecture") != spec.architecture or run_yaml.get("backend", {}).get("config", {}).get("mode", "lora") != spec.mode:
            raise SystemExit("--run-id does not match the selected sd-scripts smoke contract")
        dataset_id = run_yaml["datasets"][0]["id"]
    else:
        assert args.models is not None
        run_id, dataset_id = write_run(root, args.selector, spec, models_file=args.models, executor=args.executor, gpu=args.gpu)
    commands = [["uv", "run", "kura", "run", "plan", run_id]]
    if not args.run_id:
        commands.insert(0, ["uv", "run", "kura", "run", "compile", run_id])
    for command in commands:
        result = run(command, timeout=300)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            return result.returncode
    if not args.launch:
        print(json.dumps({"run_id": run_id, "launched": False, "next": f"after approval: scripts/sd_scripts_real_smoke.py {args.selector} --run-id {run_id} --launch --yes"}, indent=2))
        return 0
    result = run(["uv", "run", "kura", "run", "execute", run_id, "--yes"], timeout=args.timeout)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    report = validate_result(root, run_id, dataset_id, spec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.returncode == 0 and report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
