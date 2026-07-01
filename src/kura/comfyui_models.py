"""ComfyUI workflow model discovery and model registry resolution."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
from urllib.parse import urlparse


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

NOTE_MODEL_SECTIONS = {
    "checkpoints": ("checkpoints", "checkpoints"),
    "checkpoint": ("checkpoints", "checkpoints"),
    "vae": ("vae", "vae"),
    "clip": ("clip", "clip"),
    "clips": ("clip", "clip"),
    "text_encoder": ("clip", "text_encoders"),
    "text_encoders": ("clip", "text_encoders"),
    "diffusion_model": ("diffusion_models", "diffusion_models"),
    "diffusion_models": ("diffusion_models", "diffusion_models"),
    "unet": ("diffusion_models", "diffusion_models"),
    "controlnet": ("controlnet", "controlnet"),
    "controlnets": ("controlnet", "controlnet"),
}

MARKDOWN_LINK_RE = re.compile(r"^\s*-\s+\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)")


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


def ui_workflow_companion(path: Any) -> Any | None:
    """Return the likely editable ComfyUI workflow next to an API workflow."""
    try:
        stem = path.stem
    except AttributeError:
        return None
    candidates = []
    for suffix in ("_api", "-api"):
        if stem.endswith(suffix):
            candidates.append(path.with_name(stem[: -len(suffix)] + path.suffix))
    for candidate in candidates:
        if candidate != path and candidate.is_file():
            return candidate
    return None


def registry_from_ui_workflow_notes(workflow: dict[str, Any]) -> dict[str, Any]:
    """Extract explicit model links from MarkdownNote nodes in an editable workflow."""
    models: dict[str, dict[str, dict[str, str]]] = {}
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return models
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").lower()
        if "markdown" not in node_type and "note" not in node_type:
            continue
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list):
            continue
        for value in widgets:
            if isinstance(value, str):
                _extract_models_from_markdown(value, models)
    return models


def _extract_models_from_markdown(markdown: str, models: dict[str, dict[str, dict[str, str]]]) -> None:
    current: tuple[str, str] | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        section = stripped[2:].strip().rstrip(":").lower() if stripped.startswith("- ") and "[" not in stripped else ""
        if section in NOTE_MODEL_SECTIONS:
            current = NOTE_MODEL_SECTIONS[section]
            continue
        match = MARKDOWN_LINK_RE.match(line)
        if current is None or match is None:
            continue
        name = match.group("name").strip()
        url = match.group("url").strip()
        if not name or not url:
            continue
        model_type, target_dir = current
        entry = _entry_from_model_url(name, url)
        if entry is None:
            continue
        entry.setdefault("target_dir", target_dir)
        entry.setdefault("target_name", name)
        models.setdefault(model_type, {})[name] = entry


def _entry_from_model_url(name: str, url: str) -> dict[str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    if host in ("huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"):
        for marker in ("blob", "resolve"):
            if marker not in path_parts:
                continue
            marker_index = path_parts.index(marker)
            if marker_index < 2 or marker_index + 2 > len(path_parts):
                continue
            repo = "/".join(path_parts[:marker_index])
            revision = path_parts[marker_index + 1]
            filename = "/".join(path_parts[marker_index + 2 :])
            if not repo or not filename:
                continue
            entry = {"repo": repo, "filename": filename}
            if revision and revision != "main":
                entry["revision"] = revision
            return entry
    return {"url": url, "filename": name}


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
