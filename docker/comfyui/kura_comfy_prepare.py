#!/usr/bin/env python3
"""Prepare ComfyUI models for a workflow inside a RunPod ComfyUI session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
from typing import Any

from huggingface_hub import hf_hub_download


MODEL_INPUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "CheckpointLoaderSimple": (("checkpoints", "ckpt_name"),),
    "VAELoader": (("vae", "vae_name"),),
    "CLIPLoader": (("clip", "clip_name"),),
    "DualCLIPLoader": (("clip", "clip_name1"), ("clip", "clip_name2")),
    "TripleCLIPLoader": (("clip", "clip_name1"), ("clip", "clip_name2"), ("clip", "clip_name3")),
    "UNETLoader": (("diffusion_models", "unet_name"),),
    "ControlNetLoader": (("controlnet", "control_net_name"),),
}

MODEL_DIRS = {
    "checkpoints": "checkpoints",
    "vae": "vae",
    "clip": "clip",
    "diffusion_models": "diffusion_models",
    "controlnet": "controlnet",
}

MODEL_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "checkpoints": {
        "v1-5-pruned-emaonly-fp16.safetensors": {
            "repo": "Comfy-Org/stable-diffusion-v1-5-archive",
            "filename": "v1-5-pruned-emaonly-fp16.safetensors",
        }
    }
}


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"refusing unsafe model target: {relative}") from exc
    return candidate


def _required_models(workflow: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs")
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
    url = entry.get("url") or entry.get("direct_url")
    filename = entry.get("filename") or entry.get("file") or ref["name"]
    if (not repo and not url) or not filename:
        return None
    spec = {
        **ref,
        "filename": filename,
        "target_dir": entry.get("target_dir") or MODEL_DIRS.get(ref["type"], ref["type"]),
        "target_name": entry.get("target_name") or ref["name"],
    }
    if repo:
        spec["repo"] = repo
    if url:
        spec["url"] = url
    if entry.get("revision"):
        spec["revision"] = entry["revision"]
    if entry.get("subfolder"):
        spec["subfolder"] = entry["subfolder"]
    return spec


def _download_model(spec: dict[str, str], cache_dir: Path | None) -> Path:
    if spec.get("url"):
        root = cache_dir or Path("/tmp/kura-comfyui-downloads")
        digest = hashlib.sha256(spec["url"].encode("utf-8")).hexdigest()[:16]
        target = root / "direct" / digest / Path(spec["filename"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            return target
        temporary = target.with_suffix(target.suffix + ".tmp")
        with urllib.request.urlopen(spec["url"], timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(target)
        return target
    return Path(hf_hub_download(
        repo_id=spec["repo"],
        filename=spec["filename"],
        subfolder=spec.get("subfolder"),
        revision=spec.get("revision"),
        cache_dir=str(cache_dir) if cache_dir else None,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    ))


def prepare(workflow: dict[str, Any], *, comfyui_root: Path, cache_dir: Path | None, registry: dict[str, Any]) -> list[dict[str, str]]:
    models_root = comfyui_root / "models"
    specs: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    for ref in _required_models(workflow):
        spec = _resolve(ref, registry)
        if spec is None:
            unknown.append(ref)
        else:
            specs.append(spec)
    if unknown:
        raise RuntimeError("unknown ComfyUI model loader entries: " + ", ".join(f"{item['class_type']}.{item['input']}={item['name']}" for item in unknown))
    for spec in specs:
        downloaded = _download_model(spec, cache_dir)
        target = _safe_child(models_root, f"{spec['target_dir']}/{spec['target_name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == downloaded.resolve():
                continue
            target.unlink()
        os.symlink(downloaded, target)
        print(json.dumps({"event": "model_ready", "model": spec["name"], "target": str(target), "source": str(downloaded)}, ensure_ascii=False), flush=True)
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_json")
    parser.add_argument("--registry-json")
    parser.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    args = parser.parse_args()
    workflow = json.loads(Path(args.workflow_json).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise ValueError("workflow_json must contain a ComfyUI API workflow object")
    registry: dict[str, Any] = MODEL_REGISTRY
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
