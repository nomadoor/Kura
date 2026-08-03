"""ComfyUI workflow model discovery and model registry resolution."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


MODEL_INPUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "CheckpointLoaderSimple": (("checkpoints", "ckpt_name"),),
    "VAELoader": (("vae", "vae_name"),),
    "CLIPLoader": (("clip", "clip_name"),),
    "DualCLIPLoader": (("clip", "clip_name1"), ("clip", "clip_name2")),
    "TripleCLIPLoader": (("clip", "clip_name1"), ("clip", "clip_name2"), ("clip", "clip_name3")),
    "UNETLoader": (("diffusion_models", "unet_name"),),
    "ControlNetLoader": (("controlnet", "control_net_name"),),
    "ModelPatchLoader": (("model_patches", "name"),),
}


COMFYUI_MODEL_DIRS = {
    "checkpoints": "checkpoints",
    "vae": "vae",
    "clip": "clip",
    "diffusion_models": "diffusion_models",
    "controlnet": "controlnet",
    "model_patches": "model_patches",
}


DEFAULT_MODEL_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "checkpoints": {
        "v1-5-pruned-emaonly-fp16.safetensors": {
            "repo": "Comfy-Org/stable-diffusion-v1-5-archive",
            "filename": "v1-5-pruned-emaonly-fp16.safetensors",
        }
    },
    "diffusion_models": {
        "anima-base-v1.0.safetensors": {
            "repo": "circlestone-labs/Anima",
            "filename": "split_files/diffusion_models/anima-base-v1.0.safetensors",
            "revision": "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b",
        }
    },
    "clip": {
        "qwen_3_06b_base.safetensors": {
            "repo": "circlestone-labs/Anima",
            "filename": "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "revision": "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b",
            "target_dir": "text_encoders",
        }
    },
    "vae": {
        "qwen_image_vae.safetensors": {
            "repo": "circlestone-labs/Anima",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "revision": "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b",
        }
    },
}


def merged_registry(*registries: Any) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(DEFAULT_MODEL_REGISTRY)
    for registry in registries:
        sections = _registry_sections(registry)
        for section_name, section in sections.items():
            if not isinstance(section, dict):
                continue
            target = merged.setdefault(section_name, {})
            if isinstance(target, dict):
                target.update(deepcopy(section))
    return merged


def _registry_sections(registry: Any) -> dict[str, Any]:
    if not isinstance(registry, dict):
        return {}
    sections = registry.get("models", registry)
    return sections if isinstance(sections, dict) else {}


def _lookup_model(registry: Any, model_type: str, name: str) -> dict[str, Any] | None:
    sections = _registry_sections(registry)
    section = sections.get(model_type)
    if not isinstance(section, dict):
        return None
    entry = section.get(name)
    if not isinstance(entry, dict):
        return None
    return deepcopy(entry)


def required_model_refs(workflow: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
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


def visible_model_refs(workflow: dict[str, Any], object_info: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split workflow model references by visibility at one ComfyUI endpoint."""

    visible: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for ref in required_model_refs(workflow):
        node = object_info.get(ref["class_type"])
        required = node.get("input", {}).get("required", {}) if isinstance(node, dict) else {}
        raw = required.get(ref["input"]) if isinstance(required, dict) else None
        names = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else []
        (visible if ref["name"] in names else missing).append(ref)
    return visible, missing


def endpoint_fingerprint(object_info: dict[str, Any]) -> dict[str, Any]:
    """Return a stable identity hint for distinguishing ComfyUI instances.

    Model lists are deliberately excluded: users may add models between compile
    and launch. The registered node set still distinguishes a daily instance
    with custom nodes from Kura's minimal smoke image in the common case.
    """

    node_types = sorted(str(name) for name in object_info)
    encoded = json.dumps(node_types, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "kind": "comfyui-object-info-node-set-v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "node_type_count": len(node_types),
    }


def resolve_model_specs(workflow: dict[str, Any], registry: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specs: list[dict[str, Any]] = []
    unknown: list[dict[str, str]] = []
    for ref in required_model_refs(workflow):
        entry = _lookup_model(registry, ref["type"], ref["name"])
        if entry is None:
            unknown.append(ref)
            continue
        repo = entry.get("repo") or entry.get("repo_id")
        url = entry.get("url") or entry.get("direct_url")
        filename = entry.get("filename") or entry.get("file") or ref["name"]
        if (not isinstance(repo, str) or not repo) and (not isinstance(url, str) or not url):
            unknown.append(ref)
            continue
        if not isinstance(filename, str) or not filename:
            unknown.append(ref)
            continue
        target_dir = entry.get("target_dir") if isinstance(entry.get("target_dir"), str) and entry["target_dir"] else COMFYUI_MODEL_DIRS.get(ref["type"], ref["type"])
        spec = {
            **ref,
            "filename": filename,
            "target_dir": target_dir,
            "target_name": entry.get("target_name") if isinstance(entry.get("target_name"), str) else ref["name"],
        }
        if isinstance(repo, str) and repo:
            spec["repo"] = repo
        if isinstance(url, str) and url:
            spec["url"] = url
        if isinstance(entry.get("revision"), str) and entry["revision"]:
            spec["revision"] = entry["revision"]
        if isinstance(entry.get("subfolder"), str) and entry["subfolder"]:
            spec["subfolder"] = entry["subfolder"]
        specs.append(spec)
    return specs, unknown
