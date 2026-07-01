"""ComfyUI workflow model discovery and model registry resolution."""

from __future__ import annotations

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
}


COMFYUI_MODEL_DIRS = {
    "checkpoints": "checkpoints",
    "vae": "vae",
    "clip": "clip",
    "diffusion_models": "diffusion_models",
    "controlnet": "controlnet",
}


DEFAULT_MODEL_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "checkpoints": {
        "v1-5-pruned-emaonly-fp16.safetensors": {
            "repo": "Comfy-Org/stable-diffusion-v1-5-archive",
            "filename": "v1-5-pruned-emaonly-fp16.safetensors",
        }
    }
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
