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


# The keys of one model source (comfyui.model_registry and a workflow sidecar's models).
MODEL_ENTRY_KEYS = ("repo", "url", "filename", "revision", "subfolder", "target_dir", "target_name")
# Older spellings of three of those keys: refused with the new name, and renamed by `kura workspace migrate`.
RETIRED_MODEL_ENTRY_KEYS = {"repo_id": "repo", "direct_url": "url", "file": "filename"}


def merged_registry(*registries: Any) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(DEFAULT_MODEL_REGISTRY)
    for registry in registries:
        sections = registry_folders(registry)
        for section_name, section in sections.items():
            if not isinstance(section, dict):
                continue
            target = merged.setdefault(section_name, {})
            if isinstance(target, dict):
                target.update(deepcopy(section))
    return merged


def registry_folders(registry: Any) -> dict[str, Any]:
    """A registry maps a model folder to model names; the older `models:` wrapper is refused, not read as a folder."""
    if not isinstance(registry, dict):
        return {}
    if "models" in registry:
        raise ValueError(
            "a ComfyUI model registry lists model folders directly (checkpoints:, loras:, ...); remove the inner "
            "models: wrapper (for comfyui.model_registry, `kura workspace migrate` does it)"
        )
    return registry


def _lookup_model(registry: Any, model_type: str, name: str) -> dict[str, Any] | None:
    sections = registry_folders(registry)
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


def visible_model_refs(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    ignored_inputs: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split workflow model references by visibility at one ComfyUI endpoint."""

    visible: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for ref in required_model_refs(workflow):
        if ignored_inputs and (ref["node"], ref["input"]) in ignored_inputs:
            continue
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
        label = f"model {ref['type']}/{ref['name']}"
        unknown_keys = sorted(set(entry) - set(MODEL_ENTRY_KEYS))
        if unknown_keys:
            raise ValueError(
                f"{label} has unsupported key(s): "
                + ", ".join(f"{key} (write {RETIRED_MODEL_ENTRY_KEYS[key]})" if key in RETIRED_MODEL_ENTRY_KEYS else key for key in unknown_keys)
                + "; use " + ", ".join(MODEL_ENTRY_KEYS)
            )
        if ("repo" in entry) == ("url" in entry) or not all(isinstance(value, str) and value for value in entry.values()):
            raise ValueError(f"{label} needs exactly one of repo or url, and every value a non-empty string")
        spec = {
            **ref,
            "filename": entry.get("filename", ref["name"]),
            "target_dir": entry.get("target_dir", COMFYUI_MODEL_DIRS.get(ref["type"], ref["type"])),
            "target_name": entry.get("target_name", ref["name"]),
            **{key: entry[key] for key in ("repo", "url", "revision", "subfolder") if key in entry},
        }
        specs.append(spec)
    return specs, unknown
