"""ComfyUI-only render runs. This is deliberately not a general generator plugin API."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from kura import __version__
from kura.comfyui_models import endpoint_fingerprint, merged_registry, resolve_model_specs, visible_model_refs
from kura.fsio import atomic_write_json
from kura.workspace import dump_yaml


# Promptset keys Kura owns directly. Every other key in a promptset item must be
# bound to a workflow node/field through run.yaml `workflow_patches`, so that a
# promptset and a workflow can never silently disagree about which parameters exist.
PROMPTSET_CORE_KEYS = frozenset({"id", "prompt", "negative_prompt", "seeds", "meta"})

# Patch names whose value comes from the run, not from each promptset item.
# `seed` is item-sourced but read from `seeds` / render.default_seed, and
# `negative_prompt` stays optional for promptsets that do not set one.
PATCHES_WITHOUT_ITEM_KEY = frozenset({"lora", "checkpoint", "model_patch", "seed", "negative_prompt"})

# Patch names an item must never carry: writing them there looks like it sets a
# per-case value, but the value would come from the run and the item would be
# ignored. `seeds` is the per-item spelling; `seed` is the binding name.
ITEM_FORBIDDEN_KEYS = frozenset({"lora", "checkpoint", "model_patch", "seed"})

# Core keys that must be bound when the promptset supplies them. Without a
# binding the value is carried into filenames and images.jsonl while the
# workflow renders its own hardcoded one.
CORE_KEYS_REQUIRING_BINDINGS = ("prompt", "negative_prompt", "seed")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return load_yaml(path)


def _validate_train_run_reference(workspace: Path, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("render inputs.train_run must be a non-empty run ID or null")
    train_run = value.strip()
    if Path(train_run).name != train_run or train_run in {".", ".."}:
        raise ValueError("render inputs.train_run must be a run ID, not a path")
    source_path = workspace / "runs" / train_run / "run.yaml"
    if not source_path.is_file():
        raise ValueError(f"render inputs.train_run does not exist: {train_run}")
    source = load_yaml(source_path)
    if source.get("type") != "train":
        raise ValueError(f"render inputs.train_run must reference a train run: {train_run}")
    if source.get("id") not in (None, train_run):
        raise ValueError(f"render inputs.train_run ID does not match its run.yaml: {train_run}")
    return train_run


def _workflow_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".kura.yaml")
    if not sidecar.is_file():
        return {}
    return load_yaml(sidecar)


def event(run_dir: Path, payload: dict[str, Any]) -> None:
    with (run_dir / "logs" / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def status(run_dir: Path, **changes: Any) -> None:
    path = run_dir / "status.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(changes)
    atomic_write_json(path, current)


def write_realization(run_dir: Path, **details: Any) -> None:
    realization_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    path = run_dir / "realizations" / f"{realization_id}.json"
    path.parent.mkdir(exist_ok=True)
    atomic_write_json(path, {"id": realization_id, "timestamp": now(), **details})
    status(run_dir, last_realization=str(path.relative_to(run_dir)))


def _set_path(document: dict[str, Any], node: str, field: str, value: Any) -> None:
    if node not in document or not isinstance(document[node], dict):
        raise ValueError(f"workflow patch node does not exist: {node}")
    target: Any = document[node]
    pieces = field.split(".")
    for piece in pieces[:-1]:
        if not isinstance(target, dict) or piece not in target:
            raise ValueError(f"workflow patch field does not exist: {node}.{field}")
        target = target[piece]
    if not isinstance(target, dict) or pieces[-1] not in target:
        raise ValueError(f"workflow patch field does not exist: {node}.{field}")
    target[pieces[-1]] = value


def is_safe_component(value: Any) -> bool:
    """True when a value can be joined into a path without escaping or colliding."""
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    if set(value) & set("/\\") or value in (".", "..") or value.startswith("."):
        return False
    return value == Path(value).name


def normalized_workflow_fixed(value: Any) -> list[str]:
    """Return the declared fixed core inputs, rejecting YAML shape mistakes."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("render.workflow_fixed must be a list of names, for example: [prompt, seed]")
    return value


def _binding_target(name: Any, patch: Any) -> tuple[str, str, str]:
    # A binding name reaches the filesystem as `resolved/images/<name>/` and as a
    # staged file name, so it is constrained the same way a promptset id is.
    if not is_safe_component(name):
        raise ValueError(f"workflow_patches keys must be plain names, not paths: {name!r}")
    if not isinstance(patch, dict) or not isinstance(patch.get("node"), str) or not isinstance(patch.get("field"), str):
        raise ValueError(f"workflow_patches.{name} requires node and field")
    kind = patch.get("type", "value")
    if kind not in ("value", "image"):
        raise ValueError(f"workflow_patches.{name}.type must be value or image")
    return patch["node"], patch["field"], kind


def image_patch_names(patches: Any) -> list[str]:
    if not isinstance(patches, dict):
        return []
    return [name for name, patch in patches.items() if _binding_target(name, patch)[2] == "image"]


def validate_patch_bindings(workflow: dict[str, Any], patches: Any) -> None:
    """Every declared binding must name a node and field that exist in this workflow."""
    if patches in (None, {}):
        return
    if not isinstance(patches, dict):
        raise ValueError("workflow_patches must be a mapping of name to {node, field}")
    probe = deepcopy(workflow)
    for name, patch in patches.items():
        node, field, _ = _binding_target(name, patch)
        try:
            _set_path(probe, node, field, None)
        except ValueError as exc:
            raise ValueError(f"workflow_patches.{name} does not match this workflow: {exc}") from exc


def _core_binding_required(name: str, items: list[dict[str, Any]], default_seed: Any) -> bool:
    if name == "prompt":
        return True
    if name == "negative_prompt":
        return any(str(item.get("negative_prompt", "")).strip() for item in items)
    return default_seed is not None or any(item.get("seeds") for item in items)


def reconcile_promptset(items: list[dict[str, Any]], patches: Any, *, default_seed: Any = None, workflow_fixed: Any = None) -> None:
    """Fail when a promptset and the declared bindings disagree about which parameters exist.

    Kura cannot know whether an unbound key is an input the user expects to take
    effect or provenance from whatever generated the promptset, so it refuses to
    guess in either direction. Core keys are checked too: an unbound `prompt` or
    `seed` still reaches file names and `samples/images.jsonl` while the workflow
    renders its own hardcoded value, which is the silently-wrong result this
    whole contract exists to prevent.
    """
    bound = set(patches) if isinstance(patches, dict) else set()
    fixed = set(normalized_workflow_fixed(workflow_fixed))
    unknown_fixed = sorted(fixed - set(CORE_KEYS_REQUIRING_BINDINGS))
    if unknown_fixed:
        raise ValueError(
            f"render.workflow_fixed only accepts {', '.join(CORE_KEYS_REQUIRING_BINDINGS)}; remove: {', '.join(unknown_fixed)}"
        )
    conflicting = sorted(fixed & bound)
    if conflicting:
        raise ValueError(
            f"render.workflow_fixed and workflow_patches both claim {', '.join(conflicting)}; a parameter is either bound or fixed by the workflow"
        )
    if "seed" in fixed:
        # Kura must not expand cases along a parameter the workflow controls; every
        # image would be identical while file names and images.jsonl claimed a seed.
        if default_seed is not None:
            raise ValueError("render.workflow_fixed includes seed, so render.default_seed must be null")
        seeded = [item["id"] for item in items if item.get("seeds")]
        if seeded:
            raise ValueError(
                f"render.workflow_fixed includes seed, but these promptset items set seeds: {', '.join(seeded)}. "
                "The workflow would render its own seed for every one of them. Remove the seeds, or bind seed to a workflow node/field."
            )
    for name in ("prompt", "negative_prompt"):
        if name in fixed and any(str(item.get(name, "")).strip() for item in items):
            print(
                f"warning: render.workflow_fixed includes {name}, so promptset {name} values are not rendered and are recorded as null.",
                flush=True,
            )
    for name in CORE_KEYS_REQUIRING_BINDINGS:
        if name in bound or name in fixed:
            continue
        if not _core_binding_required(name, items, default_seed):
            continue
        source = "render.default_seed or promptset seeds" if name == "seed" else f"promptset {name}"
        raise ValueError(
            f"this render uses {source} but run.yaml workflow_patches has no {name} binding, so the workflow would render its own "
            f"value while Kura recorded yours. Bind {name} to a workflow node/field, or list it under render.workflow_fixed to "
            "declare that this workflow deliberately fixes it."
        )
    required = {name for name in bound if name not in PATCHES_WITHOUT_ITEM_KEY}
    for item in items:
        forbidden = sorted(set(item) & ITEM_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(
                f"promptset item {item['id']!r} sets {', '.join(forbidden)}, which comes from the run rather than the item and would be "
                "ignored. Use `seeds` for per-case seeds; the checkpoint is set by inputs.checkpoint."
            )
        unbound = sorted(set(item) - PROMPTSET_CORE_KEYS - bound)
        if unbound:
            raise ValueError(
                f"promptset item {item['id']!r} declares {', '.join(unbound)} but run.yaml workflow_patches has no binding for "
                f"{'them' if len(unbound) > 1 else 'it'}. Either bind each one to a workflow node/field, move it under `meta` if it is "
                "provenance rather than an input, or remove it because this workflow derives that value another way."
            )
        missing = sorted(name for name in required if name not in item)
        if missing:
            raise ValueError(
                f"promptset item {item['id']!r} has no value for bound workflow_patches: {', '.join(missing)}"
            )


def patch_workflow(
    workflow: dict[str, Any],
    patches: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    checkpoint: str,
    model_patch: str | None = None,
    item: dict[str, Any] | None = None,
    image_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    patched = deepcopy(workflow)
    values: dict[str, Any] = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "lora": checkpoint, "checkpoint": checkpoint, "model_patch": model_patch if model_patch is not None else checkpoint}
    values.update(image_values or {})
    for name, patch in patches.items():
        node, field, _ = _binding_target(name, patch)
        if name in values:
            value = values[name]
        elif item is not None and name in item:
            value = item[name]
        elif item is not None:
            raise ValueError(f"promptset item {item.get('id')!r} has no value for workflow_patches.{name}")
        else:
            continue
        _set_path(patched, node, field, value)
    return patched


def _link(node: str, output: int) -> list[Any]:
    return [node, output]


def _as_node_id(value: Any, *, context: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int):
        return str(value)
    raise ValueError(f"{context} requires node")


def _as_output_index(value: Any, default: int, *, context: str) -> int:
    if value is None:
        return default
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{context} output must be a non-negative integer")


def _lora_insert_from_sidecar(sidecar: dict[str, Any]) -> dict[str, Any] | None:
    raw = sidecar.get("lora_insert")
    if raw is None and isinstance(sidecar.get("lora"), dict):
        raw = sidecar["lora"].get("insert")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("lora_insert must be a mapping")
    kind = str(raw.get("kind") or raw.get("type") or "model_clip").strip()
    class_type = "LoraLoaderModelOnly" if kind in ("model_only", "LoraLoaderModelOnly") else "LoraLoader"
    if kind not in ("model_only", "model_clip", "full", "LoraLoaderModelOnly", "LoraLoader"):
        raise ValueError("lora_insert.kind must be one of: model_only, model_clip, full, LoraLoaderModelOnly, LoraLoader")
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    clip = raw.get("clip") if isinstance(raw.get("clip"), dict) else {}
    model_node = _as_node_id(raw.get("model_node", model.get("node")), context="lora_insert.model")
    spec: dict[str, Any] = {
        "class_type": class_type,
        "model_node": model_node,
        "model_output": _as_output_index(raw.get("model_output", model.get("output")), 0, context="lora_insert.model"),
        "strength_model": float(raw.get("strength_model", 0.8)),
    }
    if class_type == "LoraLoader":
        spec["clip_node"] = _as_node_id(raw.get("clip_node", clip.get("node", model_node)), context="lora_insert.clip")
        spec["clip_output"] = _as_output_index(raw.get("clip_output", clip.get("output")), 1, context="lora_insert.clip")
        spec["strength_clip"] = float(raw.get("strength_clip", 0.8))
    return spec


def insert_lora_loader(workflow: dict[str, Any], spec: dict[str, Any] | None, lora_name: str) -> dict[str, Any]:
    if not spec or not lora_name:
        return workflow
    patched = deepcopy(workflow)
    model_node = spec["model_node"]
    if model_node not in patched:
        raise ValueError(f"lora_insert model node does not exist: {model_node}")
    class_type = spec["class_type"]
    model_link = _link(model_node, int(spec.get("model_output", 0)))
    if class_type == "LoraLoader":
        clip_node = spec["clip_node"]
        if clip_node not in patched:
            raise ValueError(f"lora_insert clip node does not exist: {clip_node}")
        clip_link = _link(clip_node, int(spec.get("clip_output", 1)))
    else:
        clip_link = None
    node_id = _next_workflow_node_id(patched)
    inputs: dict[str, Any] = {
        "model": model_link,
        "lora_name": lora_name,
        "strength_model": float(spec.get("strength_model", 0.8)),
    }
    if class_type == "LoraLoader":
        inputs["clip"] = clip_link
        inputs["strength_clip"] = float(spec.get("strength_clip", 0.8))
    patched[node_id] = {"class_type": class_type, "inputs": inputs}
    _replace_links(patched, model_link, _link(node_id, 0), skip_node=node_id)
    if clip_link is not None:
        _replace_links(patched, clip_link, _link(node_id, 1), skip_node=node_id)
    return patched


def checkpoint_application(
    frozen: dict[str, Any],
    workflow: dict[str, Any],
    *,
    lora_name: str = "",
) -> dict[str, Any]:
    """Describe how the frozen checkpoint participates in the workflow."""

    inserted = frozen.get("lora_insert")
    if isinstance(inserted, dict):
        if not lora_name:
            return {"kind": "none"}
        application: dict[str, Any] = {
            "kind": "lora_insert",
            "class_type": inserted.get("class_type"),
            "strength_model": inserted.get("strength_model"),
        }
        if inserted.get("class_type") == "LoraLoader":
            application["strength_clip"] = inserted.get("strength_clip")
        return application

    patches = frozen.get("workflow_patches")
    patches = patches if isinstance(patches, dict) else {}
    for name in ("lora", "model_patch", "checkpoint"):
        binding = patches.get(name)
        if not isinstance(binding, dict):
            continue
        node_id = str(binding.get("node") or "")
        node = workflow.get(node_id)
        node = node if isinstance(node, dict) else {}
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        application = {
            "kind": f"{name}_binding",
            "node": node_id or None,
            "field": binding.get("field"),
            "class_type": node.get("class_type"),
        }
        if name == "lora":
            if isinstance(inputs.get("strength_model"), (int, float)):
                application["strength_model"] = inputs["strength_model"]
            if isinstance(inputs.get("strength_clip"), (int, float)):
                application["strength_clip"] = inputs["strength_clip"]
        return application

    checkpoint = frozen.get("inputs", {}).get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("path"):
        return {"kind": "checkpoint_reference"}
    return {"kind": "none"}


def _next_workflow_node_id(workflow: dict[str, Any]) -> str:
    numeric = [int(node_id) for node_id in workflow if isinstance(node_id, str) and node_id.isdigit()]
    return str(max(numeric, default=0) + 1)


def _replace_links(value: Any, old: list[Any], new: list[Any], *, skip_node: str | None = None, node_id: str | None = None) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_node_id = key if node_id is None and isinstance(key, str) else node_id
            if next_node_id == skip_node:
                continue
            if isinstance(child, list) and child == old:
                value[key] = list(new)
            else:
                _replace_links(child, old, new, skip_node=skip_node, node_id=next_node_id)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, list) and child == old:
                value[index] = list(new)
            else:
                _replace_links(child, old, new, skip_node=skip_node, node_id=node_id)


def _workspace_path(workspace: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _safe_stage_name(run_id: str, source: Path) -> str:
    stem = "".join(character if character.isalnum() or character in "._-" else "-" for character in source.stem)
    digest8 = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
    suffix = source.suffix or ".safetensors"
    tail = f"-{digest8}{suffix}"
    prefix = f"{run_id}-"
    max_prefix = max(0, 220 - len(tail))
    prefix = prefix[:max_prefix]
    max_stem = max(0, 220 - len(prefix) - len(tail))
    return f"{prefix}{stem[:max_stem]}{tail}"


def _lora_stage_plan(workspace: Path, run_dir: Path, frozen: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    if "lora" not in frozen.get("workflow_patches", {}) and not frozen.get("lora_insert"):
        return None
    if str(frozen.get("render", {}).get("lora_stage", "auto")).strip().lower() in ("0", "false", "off", "none", "no"):
        return None
    source = _workspace_path(workspace, checkpoint.get("path"))
    if source is None or not source.is_file() or source.suffix != ".safetensors":
        return None
    comfyui = frozen.get("comfyui", {})
    if not isinstance(comfyui, dict):
        return None
    stage_subdir = str(comfyui.get("lora_stage_subdir") or "Kura_tmp").strip("/\\")
    if not stage_subdir or Path(stage_subdir).is_absolute() or ".." in Path(stage_subdir).parts:
        raise ValueError("comfyui.lora_stage_subdir must be a safe relative directory name")
    mode = str(comfyui.get("lora_stage_mode") or "symlink").strip().lower()
    if mode not in ("symlink", "copy"):
        raise ValueError("comfyui.lora_stage_mode must be symlink or copy")
    cleanup = str(comfyui.get("lora_stage_cleanup") or "remove_after_render").strip().lower()
    if cleanup not in ("remove_after_render", "keep"):
        raise ValueError("comfyui.lora_stage_cleanup must be remove_after_render or keep")
    lora_dir = _workspace_path(workspace, comfyui.get("lora_dir"))
    if lora_dir is None:
        return None
    stage_dir = (lora_dir / stage_subdir).resolve()
    target = stage_dir / _safe_stage_name(run_dir.name, source)
    return {
        "kind": "LoRA",
        "source": str(source),
        "target": str(target),
        "lora_name": f"{stage_subdir}/{target.name}",
        "mode": mode,
        "cleanup": cleanup,
        "created": False,
    }


def _model_patch_stage_plan(workspace: Path, run_dir: Path, frozen: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    if "model_patch" not in frozen.get("workflow_patches", {}):
        return None
    source = _workspace_path(workspace, checkpoint.get("path"))
    if source is None or not source.is_file() or source.suffix != ".safetensors":
        return None
    comfyui = frozen.get("comfyui", {})
    if not isinstance(comfyui, dict):
        return None
    directory = _workspace_path(workspace, comfyui.get("model_patches_dir"))
    if directory is None:
        raise ValueError("comfyui.model_patches_dir is required to stage a model_patch checkpoint")
    stage_subdir = str(comfyui.get("model_patch_stage_subdir") or "Kura_tmp").strip("/\\")
    if not stage_subdir or Path(stage_subdir).is_absolute() or ".." in Path(stage_subdir).parts:
        raise ValueError("comfyui.model_patch_stage_subdir must be a safe relative directory name")
    mode = str(comfyui.get("model_patch_stage_mode") or "symlink").strip().lower()
    if mode not in ("symlink", "copy"):
        raise ValueError("comfyui.model_patch_stage_mode must be symlink or copy")
    cleanup = str(comfyui.get("model_patch_stage_cleanup") or "remove_after_render").strip().lower()
    if cleanup not in ("remove_after_render", "keep"):
        raise ValueError("comfyui.model_patch_stage_cleanup must be remove_after_render or keep")
    target = (directory / stage_subdir).resolve() / _safe_stage_name(run_dir.name, source)
    return {"kind": "model patch", "source": str(source), "target": str(target), "model_patch_name": f"{stage_subdir}/{target.name}", "mode": mode, "cleanup": cleanup, "created": False}


def _freeze_promptset_images(run_dir: Path, promptset_path: Path, items: list[dict[str, Any]], patches: Any) -> list[dict[str, Any]]:
    """Copy every image referenced by an image binding into `resolved/` and repoint the items at it.

    Item paths resolve against the promptset's own directory so a promptset stays
    self-contained, and the frozen copy makes the render reproducible after the
    original file moves or changes.
    """
    names = image_patch_names(patches)
    if not names:
        return []
    base = promptset_path.parent.resolve()
    frozen_root = run_dir / "resolved" / "images"
    records: list[dict[str, Any]] = []
    for name in names:
        for item in items:
            raw = item.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"promptset item {item['id']!r} must set {name} to an image path relative to the promptset directory")
            candidate = Path(raw)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"promptset item {item['id']!r} {name} must be a relative path inside the promptset directory: {raw}")
            source = (base / candidate).resolve()
            if base != source and base not in source.parents:
                raise ValueError(f"promptset item {item['id']!r} {name} escapes the promptset directory: {raw}")
            if not source.is_file():
                raise ValueError(f"promptset item {item['id']!r} {name} does not exist: {source}")
            frozen_root.mkdir(parents=True, exist_ok=True)
            resolved_root = frozen_root.resolve()
            target_dir = (resolved_root / name).resolve()
            if target_dir.parent != resolved_root:
                raise ValueError(f"workflow_patches.{name} does not resolve to a frozen image directory inside the run")
            target_dir.mkdir(parents=True, exist_ok=True)
            target = (target_dir / f"{item['id']}{source.suffix}").resolve()
            if target.parent != target_dir:
                raise ValueError(f"promptset item {item['id']!r} does not resolve to a frozen image inside the run")
            shutil.copyfile(source, target)
            relative = target.relative_to(run_dir.resolve()).as_posix()
            item[name] = relative
            records.append({"patch": name, "prompt_id": item["id"], "source": str(source), "resolved": relative, "digest": digest(target)})
    return records


def _image_stage_plans(workspace: Path, run_dir: Path, frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage frozen promptset images into the ComfyUI input directory Kura was told about.

    Images reach ComfyUI the same way LoRAs and model patches do: through a
    configured directory the user owns. Kura never uploads through the ComfyUI API.
    """
    patches = frozen.get("workflow_patches", {})
    names = image_patch_names(patches)
    if not names:
        return []
    comfyui = frozen.get("comfyui", {})
    if not isinstance(comfyui, dict):
        comfyui = {}
    directory = _workspace_path(workspace, comfyui.get("input_dir"))
    if directory is None:
        raise ValueError("workspace.yaml comfyui.input_dir is required to render a promptset with image bindings")
    stage_subdir = str(comfyui.get("input_stage_subdir") or "Kura_tmp").strip("/\\")
    if not stage_subdir or Path(stage_subdir).is_absolute() or ".." in Path(stage_subdir).parts:
        raise ValueError("comfyui.input_stage_subdir must be a safe relative directory name")
    # Copy, not symlink: ComfyUI resolves LoadImage paths and rejects one that
    # leaves its input directory, so a symlinked input fails validation at queue
    # time even though a symlinked model loads fine.
    mode = str(comfyui.get("input_stage_mode") or "copy").strip().lower()
    if mode not in ("symlink", "copy"):
        raise ValueError("comfyui.input_stage_mode must be symlink or copy")
    if mode == "symlink":
        print(
            "warning: comfyui.input_stage_mode=symlink is set, but ComfyUI rejects symlinked LoadImage inputs "
            "(\"Invalid image file\"). Use copy unless this endpoint is known to accept them.",
            flush=True,
        )
    cleanup = str(comfyui.get("input_stage_cleanup") or "remove_after_render").strip().lower()
    if cleanup not in ("remove_after_render", "keep"):
        raise ValueError("comfyui.input_stage_cleanup must be remove_after_render or keep")
    plans: dict[str, dict[str, Any]] = {}
    for name in names:
        for source in sorted((run_dir / "resolved" / "images" / name).glob("*")):
            if not source.is_file():
                continue
            key = source.relative_to(run_dir).as_posix()
            target = (directory / stage_subdir).resolve() / _safe_stage_name(run_dir.name, source)
            plans[key] = {"kind": "image", "patch": name, "frozen": key, "source": str(source), "target": str(target), "image_name": f"{stage_subdir}/{target.name}", "mode": mode, "cleanup": cleanup, "created": False}
    return list(plans.values())


def _image_values_for_item(item: dict[str, Any], patches: Any, plans: list[dict[str, Any]]) -> dict[str, str]:
    by_frozen = {plan["frozen"]: plan["image_name"] for plan in plans}
    values: dict[str, str] = {}
    for name in image_patch_names(patches):
        frozen_path = item.get(name)
        if not isinstance(frozen_path, str) or frozen_path not in by_frozen:
            raise ValueError(f"promptset item {item.get('id')!r} has no staged image for workflow_patches.{name}")
        values[name] = by_frozen[frozen_path]
    return values


# There is deliberately no staged-image visibility pre-check. ComfyUI's
# `/object_info/LoadImage` listing does not enumerate files inside input
# subdirectories, so a correctly staged image never appears there and the check
# only ever produced a false warning. Unlike a LoRA — where a name ComfyUI
# cannot see is silently ignored and the wrong weights render — an unusable
# image name is rejected outright when the prompt is queued, and that rejection
# names the file. The server-side validation is the gate.


def _dynamically_patched_model_inputs(frozen: dict[str, Any]) -> set[tuple[str, str]]:
    ignored: set[tuple[str, str]] = set()
    patches = frozen.get("workflow_patches")
    if not isinstance(patches, dict):
        return ignored
    patch = patches.get("model_patch")
    if isinstance(patch, dict):
        node = patch.get("node")
        field = patch.get("field")
        if isinstance(node, (str, int)) and isinstance(field, str) and field.startswith("inputs."):
            ignored.add((str(node), field.removeprefix("inputs.")))
    return ignored


def _freeze_comfyui_config(comfyui: Any, *, include_remote: bool) -> dict[str, Any]:
    if not isinstance(comfyui, dict):
        return {}
    allowed = ["lora_dir", "lora_stage_subdir", "lora_stage_mode", "lora_stage_cleanup", "model_patches_dir", "model_patch_stage_subdir", "model_patch_stage_mode", "model_patch_stage_cleanup", "input_dir", "input_stage_subdir", "input_stage_mode", "input_stage_cleanup"]
    if include_remote:
        allowed.extend(("model_registry", "runpod"))
    return {key: deepcopy(comfyui[key]) for key in allowed if key in comfyui}


def _fetch_endpoint_object_info(endpoint: str, *, timeout: float = 15) -> dict[str, Any]:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/object_info", timeout=timeout) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError("ComfyUI object_info must be a JSON object")
    return payload


def _materialize_stage(plan: dict[str, Any]) -> None:
    source = Path(plan["source"])
    target = Path(plan["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            plan["created"] = False
            return
        if target.is_file() and not target.is_symlink() and digest(target) == digest(source):
            plan["created"] = False
            return
        raise ValueError(f"ComfyUI {plan.get('kind', 'file')} stage target already exists with different content: {target}")
    if plan["mode"] == "copy":
        shutil.copy2(source, target)
        plan["created"] = True
        return
    try:
        os.symlink(source, target)
        plan["created"] = True
    except OSError:
        shutil.copy2(source, target)
        plan["mode"] = "copy"
        plan["created"] = True


def _cleanup_stage(plan: dict[str, Any] | None) -> None:
    if not plan or plan.get("cleanup") != "remove_after_render" or not plan.get("created"):
        return
    target = Path(str(plan.get("target", "")))
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
    except OSError:
        pass


def _ensure_model_patch_stage_visible(client: Any, endpoint: str, plan: dict[str, Any] | None) -> None:
    if not plan:
        return
    name = str(plan.get("model_patch_name", ""))
    safe_endpoint = _redact_url_userinfo(endpoint)
    try:
        if name in client.model_patch_names():
            return
        time.sleep(0.5)
        if name in client.model_patch_names():
            return
    except RuntimeError as exc:
        raise ValueError(f"ComfyUI model-patch visibility could not be checked; endpoint={safe_endpoint}; error={exc}") from exc
    raise ValueError(f"ComfyUI model-patch stage is not visible; endpoint={safe_endpoint}; model_patch_name={name}")


def _lora_name_visible(client: Any, lora_name: str) -> bool:
    if not lora_name or not hasattr(client, "lora_names"):
        return True
    return lora_name in client.lora_names()


def _redact_url_userinfo(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if "@" not in parsed.netloc:
        return urllib.parse.urlunparse(parsed)
    host = parsed.netloc.rsplit("@", 1)[1]
    return urllib.parse.urlunparse(parsed._replace(netloc=f"***@{host}"))


def _ensure_lora_stage_visible(client: Any, endpoint: str, plan: dict[str, Any] | None) -> None:
    if not plan:
        return
    safe_endpoint = _redact_url_userinfo(endpoint)
    try:
        if _lora_name_visible(client, str(plan.get("lora_name", ""))):
            return
        time.sleep(0.5)
        if _lora_name_visible(client, str(plan.get("lora_name", ""))):
            return
    except RuntimeError as exc:
        raise ValueError(
            "ComfyUI LoRA visibility could not be checked because object_info is unavailable. "
            f"endpoint={safe_endpoint}; error={exc}. "
            f"Run `uv run kura doctor comfyui --endpoint {safe_endpoint}` to check the endpoint."
        ) from exc
    raise ValueError(
        "ComfyUI LoRA stage is not visible from the configured endpoint. "
        f"endpoint={safe_endpoint}; lora_name={plan.get('lora_name')}; lora_dir={Path(str(plan.get('target'))).parent.parent}. "
        f"Run `uv run kura doctor comfyui --endpoint {safe_endpoint} --probe-stage` to verify staging, "
        "then set comfyui.lora_dir to a LoRA directory used by that ComfyUI instance and recompile the render run."
    )


def _validate_prompt_id(value: Any, line_number: int) -> str:
    """Ids become file names under `resolved/` and `samples/`, so they must be inert.

    An id is joined into output paths; anything that can traverse or collide would
    let a promptset write outside its run or silently overwrite another case.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"promptset:{line_number}: id and prompt are required")
    if not is_safe_component(value):
        raise ValueError(f"promptset:{line_number}: id must be a single safe file name, not a path: {value!r}")
    return value


def promptset(path: Path, *, require_prompt: bool = True) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"promptset:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(item, dict) or (require_prompt and not item.get("prompt")):
            raise ValueError(f"promptset:{line_number}: id and prompt are required")
        item_id = _validate_prompt_id(item.get("id"), line_number)
        if item_id in seen:
            raise ValueError(f"promptset:{line_number}: duplicate id {item_id!r} (already used on line {seen[item_id]})")
        seen[item_id] = line_number
        prompts.append(item)
    return prompts


class ComfyUIClient:
    def __init__(self, endpoint: str, timeout: int) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(f"{self.endpoint}{path}", data=data, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # ComfyUI reports which node and input it rejected in the response body.
            # Without it a caller only sees "HTTP Error 400: Bad Request".
            try:
                detail = exc.read().decode("utf-8", "replace").strip()
            except OSError:
                detail = ""
            raise urllib.error.HTTPError(
                exc.url, exc.code, f"{exc.reason}: {detail[:1000]}" if detail else exc.reason, exc.headers, None
            ) from None

    def lora_names(self) -> set[str]:
        names: set[str] = set()
        errors: list[str] = []
        responded = False
        for class_type in ("LoraLoader", "LoraLoaderModelOnly"):
            try:
                response = self._json(f"/object_info/{class_type}")
                responded = True
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                errors.append(f"{class_type}: {exc}")
                continue
            node = response.get(class_type)
            if not isinstance(node, dict):
                continue
            required = node.get("input", {}).get("required", {})
            raw = required.get("lora_name")
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                names.update(str(item) for item in raw[0])
        if not responded:
            raise RuntimeError("ComfyUI object_info query failed for LoRA loaders: " + "; ".join(errors))
        return names

    def object_info(self) -> dict[str, Any]:
        response = self._json("/object_info")
        if not isinstance(response, dict):
            raise RuntimeError("ComfyUI object_info is not a JSON object")
        return response

    def model_patch_names(self) -> set[str]:
        try:
            response = self._json("/object_info/ModelPatchLoader")
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ModelPatchLoader object_info query failed: {exc}") from exc
        node = response.get("ModelPatchLoader")
        required = node.get("input", {}).get("required", {}) if isinstance(node, dict) else {}
        raw = required.get("name") if isinstance(required, dict) else None
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return {str(item) for item in raw[0]}
        return set()

    def queue(self, workflow: dict[str, Any]) -> str:
        response = self._json("/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
        return prompt_id

    def wait(self, prompt_id: str) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            history = self._json(f"/history/{urllib.parse.quote(prompt_id)}")
            record = history.get(prompt_id, {})
            outputs = record.get("outputs")
            if isinstance(outputs, dict):
                images = [image for output in outputs.values() if isinstance(output, dict) for image in output.get("images", [])]
                return [image for image in images if isinstance(image, dict)]
            time.sleep(1)
        raise TimeoutError(f"ComfyUI prompt timed out after {self.timeout} seconds: {prompt_id}")

    def download(self, image: dict[str, Any]) -> bytes:
        query = urllib.parse.urlencode({key: image.get(key, "") for key in ("filename", "subfolder", "type")})
        with urllib.request.urlopen(f"{self.endpoint}/view?{query}", timeout=30) as response:
            return response.read()


def compile_render(workspace: Path, run_dir: Path) -> None:
    run = load_yaml(run_dir / "run.yaml")
    workspace_config = load_optional_yaml(workspace / "workspace.yaml")
    inputs = run.get("inputs", {})
    train_run = _validate_train_run_reference(workspace, inputs.get("train_run"))
    workflow_path = workspace / inputs.get("workflow", {}).get("path", "")
    promptset_path = workspace / inputs.get("promptset", {}).get("path", "")
    if not workflow_path.is_file() or not promptset_path.is_file():
        raise ValueError("render inputs.workflow.path and inputs.promptset.path must exist")
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"workflow is not valid JSON: {exc}") from exc
    sidecar = _workflow_sidecar(workflow_path)
    lora_insert = _lora_insert_from_sidecar(sidecar) if isinstance(sidecar, dict) else None
    render_settings = run.get("render") if isinstance(run.get("render"), dict) else {}
    workflow_fixed = normalized_workflow_fixed(render_settings.get("workflow_fixed"))
    fixed = set(workflow_fixed)
    patches = run.get("workflow_patches", {})
    items = promptset(promptset_path, require_prompt="prompt" not in fixed)
    validate_patch_bindings(workflow, patches)
    reconcile_promptset(items, patches, default_seed=render_settings.get("default_seed"), workflow_fixed=workflow_fixed)
    executor = run.get("executor") if isinstance(run.get("executor"), dict) else {}
    is_runpod = executor.get("name") == "runpod"
    if is_runpod and image_patch_names(patches):
        raise ValueError("promptset image bindings are not supported for the runpod executor; render image-driven promptsets against a local ComfyUI endpoint")
    frozen = deepcopy(run)
    frozen.setdefault("inputs", {})["train_run"] = train_run
    if lora_insert:
        insert_lora_loader(workflow, lora_insert, "placeholder.safetensors")
        frozen["lora_insert"] = lora_insert
    frozen.setdefault("inputs", {}).setdefault("workflow", {})["digest"] = digest(workflow_path)
    frozen["inputs"].setdefault("promptset", {})["digest"] = digest(promptset_path)
    comfyui = _freeze_comfyui_config(workspace_config.get("comfyui"), include_remote=is_runpod)
    if comfyui:
        frozen["comfyui"] = comfyui
    if is_runpod:
        sidecar_models = sidecar.get("models") if isinstance(sidecar, dict) else {}
        workspace_models = comfyui.get("model_registry") if isinstance(comfyui, dict) else {}
        registry = merged_registry(sidecar_models, workspace_models)
        specs, unknown = resolve_model_specs(workflow, registry)
        if unknown:
            labels = ", ".join(f"{item['class_type']}.{item['input']}={item['name']}" for item in unknown)
            raise ValueError("runpod ComfyUI render has unknown model loader entries; add comfyui.model_registry mappings for: " + labels)
        frozen["comfyui_model_registry"] = registry
        frozen["comfyui_models"] = specs
    else:
        if "model_patch" in frozen.get("workflow_patches", {}) and _model_patch_stage_plan(workspace, run_dir, frozen, inputs.get("checkpoint", {})) is None:
            raise ValueError("local model_patch render requires a readable .safetensors checkpoint and configured comfyui.model_patches_dir")
        endpoint = run.get("generator", {}).get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            try:
                object_info = _fetch_endpoint_object_info(endpoint)
            except Exception as exc:
                print(f"warning: ComfyUI endpoint identity is unavailable at compile time: {_redact_url_userinfo(endpoint)}: {exc}", flush=True)
            else:
                visible, missing = visible_model_refs(workflow, object_info, ignored_inputs=_dynamically_patched_model_inputs(frozen))
                frozen["comfyui_endpoint_identity"] = endpoint_fingerprint(object_info)
                frozen["comfyui_required_models"] = {"visible": visible, "missing": missing}
    checkpoint_path = inputs.get("checkpoint", {}).get("path")
    if checkpoint_path:
        candidate = workspace / checkpoint_path
        if candidate.is_file():
            frozen["inputs"].setdefault("checkpoint", {})["hash"] = digest(candidate)
        elif not inputs.get("checkpoint", {}).get("hash"):
            print("warning: checkpoint hash is unavailable", flush=True)
    # Image freezing mutates resolved/ and the in-memory promptset. Keep it
    # after every validation and digest step so a rejected compile leaves no
    # unreferenced image copies behind.
    resolved = run_dir / "resolved"
    resolved.mkdir(exist_ok=True)
    frozen_images = _freeze_promptset_images(run_dir, promptset_path, items, patches)
    if frozen_images:
        frozen["promptset_images"] = frozen_images
    frozen["_kura"] = {"frozen_at": now(), "artifact": "manifest.lock"}
    dump_yaml(resolved / "manifest.lock.yaml", frozen)
    atomic_write_json(resolved / "workflow_used.json", workflow)
    if "comfyui_models" in frozen:
        atomic_write_json(resolved / "comfyui_models.json", frozen["comfyui_models"])
    if "comfyui_model_registry" in frozen:
        atomic_write_json(resolved / "comfyui_model_registry.json", frozen["comfyui_model_registry"])
    if frozen_images:
        (resolved / "promptset_used.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    else:
        shutil.copyfile(promptset_path, resolved / "promptset_used.jsonl")
    dump_yaml(resolved / "env.lock", {"kura_version": __version__, "generator": "comfyui", "endpoint": run.get("generator", {}).get("endpoint"), "generated_at": now()})
    status(run_dir, state="compiled")


def launch_render(
    workspace: Path,
    run_dir: Path,
    dry_run: bool = False,
    *,
    endpoint_override: str | None = None,
    lora_name_override: str | None = None,
    executor_name: str | None = None,
    manage_lora_stage: bool = True,
) -> int:
    manifest_path = run_dir / "resolved" / "manifest.lock.yaml"
    workflow_used_path = run_dir / "resolved" / "workflow_used.json"
    if not manifest_path.is_file() or not workflow_used_path.is_file():
        raise ValueError("render is not compiled; run kura render compile first")
    frozen = load_yaml(manifest_path)
    resolved_executor = executor_name or frozen.get("executor", {}).get("name")
    if frozen.get("generator", {}).get("name") != "comfyui" or resolved_executor not in ("local", "runpod"):
        raise ValueError("render runs require generator.name=comfyui and executor.name=local or runpod")
    if resolved_executor == "runpod" and "model_patch" in frozen.get("workflow_patches", {}):
        raise ValueError("ComfyUI model patch staging is not supported for the runpod executor")
    if resolved_executor == "runpod" and image_patch_names(frozen.get("workflow_patches", {})):
        raise ValueError("promptset image bindings are not supported for the runpod executor")
    current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    allowed_states = {"compiled"} if resolved_executor == "local" else {"compiled", "running"}
    if current_status.get("state") not in allowed_states:
        raise ValueError("render must be compiled and has already been launched or finalized")
    inputs = frozen.get("inputs", {})
    train_run = inputs.get("train_run")
    workflow_path = workspace / inputs.get("workflow", {}).get("path", "")
    promptset_path = workspace / inputs.get("promptset", {}).get("path", "")
    promptset_used_path = run_dir / "resolved" / "promptset_used.jsonl"
    if not promptset_used_path.is_file():
        raise ValueError("render promptset is not frozen; run kura render compile first")
    workflow_fixed = normalized_workflow_fixed(frozen.get("render", {}).get("workflow_fixed"))
    prompts = promptset(promptset_used_path, require_prompt="prompt" not in workflow_fixed)
    checkpoint = inputs.get("checkpoint", {})
    default_seed = frozen.get("render", {}).get("default_seed")
    if "seed" in workflow_fixed:
        # The workflow owns the seed, so there is nothing to vary and nothing Kura
        # may claim about it. One image per case, and the record says seed=None.
        pairs = [(item, None) for item in prompts]
    else:
        pairs = [(item, seed) for item in prompts for seed in (item.get("seeds") or [default_seed]) if seed is not None]
        if not pairs:
            raise ValueError("promptset has no seeds and render.default_seed is not set")
    endpoint = endpoint_override or frozen["generator"].get("endpoint")
    lora_stage = _lora_stage_plan(workspace, run_dir, frozen, checkpoint) if manage_lora_stage else None
    model_patch_stage = _model_patch_stage_plan(workspace, run_dir, frozen, checkpoint) if manage_lora_stage else None
    image_stages = _image_stage_plans(workspace, run_dir, frozen)
    lora_name = lora_name_override or (lora_stage["lora_name"] if lora_stage else checkpoint.get("path", ""))
    model_patch_name = model_patch_stage["model_patch_name"] if model_patch_stage else checkpoint.get("path", "")
    workflow = json.loads(workflow_used_path.read_text(encoding="utf-8"))
    application = checkpoint_application(frozen, workflow, lora_name=lora_name)
    details = {"train_run": train_run, "endpoint": endpoint, "workflow_path": str(workflow_path), "workflow_digest": inputs.get("workflow", {}).get("digest"), "promptset_path": str(promptset_path), "promptset_digest": inputs.get("promptset", {}).get("digest"), "prompt_count": len(prompts), "total_image_count": len(pairs), "checkpoint": checkpoint, "checkpoint_application": application, "comfyui_lora_name": lora_name, "comfyui_model_patch_name": model_patch_name, "lora_stage": lora_stage, "model_patch_stage": model_patch_stage, "image_stages": image_stages, "executor": resolved_executor, "output_dir": frozen.get("render", {}).get("output_dir"), "patch_mapping": frozen.get("workflow_patches", {}), "resolved_paths": ["resolved/manifest.lock.yaml", "resolved/workflow_used.json", "resolved/promptset_used.jsonl", "resolved/env.lock"]}
    if resolved_executor == "local":
        expected_identity = frozen.get("comfyui_endpoint_identity")
        identity_verified = isinstance(expected_identity, dict) and bool(expected_identity.get("sha256"))
        details["endpoint_identity_verified"] = identity_verified
        if not identity_verified:
            details["launch_blocker"] = (
                "local ComfyUI endpoint identity was not verified at compile time; "
                "verify the intended endpoint is reachable, then create and compile a new render run"
            )
    if dry_run:
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return 0
    output_dir = run_dir / frozen.get("render", {}).get("output_dir", "samples/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_log = run_dir / "samples" / "images.jsonl"
    images_log.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("render generator endpoint is empty")
    client = ComfyUIClient(endpoint, int(frozen.get("render", {}).get("timeout_sec", 600)))
    stdout_log = run_dir / "logs" / "stdout.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(f"render endpoint: {endpoint}\n", encoding="utf-8")
    status(run_dir, state="running", started=now(), ended=None, exit_code=None)
    try:
        if resolved_executor == "local" and hasattr(client, "object_info"):
            object_info = client.object_info()
            expected_identity = frozen.get("comfyui_endpoint_identity")
            observed_identity = endpoint_fingerprint(object_info)
            if not isinstance(expected_identity, dict) or not expected_identity.get("sha256"):
                raise ValueError(
                    "local ComfyUI endpoint identity was not verified at compile time; verify the intended endpoint is reachable, then compile again"
                )
            if expected_identity.get("sha256") != observed_identity.get("sha256"):
                raise ValueError(
                    "ComfyUI endpoint identity changed after compile; create and compile a new render run after verifying the intended endpoint. "
                    f"expected={expected_identity.get('sha256')} observed={observed_identity.get('sha256')} endpoint={_redact_url_userinfo(endpoint)}"
                )
            _, missing = visible_model_refs(workflow, object_info, ignored_inputs=_dynamically_patched_model_inputs(frozen))
            if missing:
                labels = ", ".join(f"{item['class_type']}.{item['input']}={item['name']}" for item in missing)
                raise ValueError(
                    "ComfyUI endpoint cannot see workflow-required models: " + labels + ". "
                    "Verify comfyui.endpoint and the user's ComfyUI model paths. Local render never downloads models."
                )
        if lora_stage:
            _materialize_stage(lora_stage)
            _ensure_lora_stage_visible(client, endpoint, lora_stage)
        if model_patch_stage:
            _materialize_stage(model_patch_stage)
            _ensure_model_patch_stage_visible(client, endpoint, model_patch_stage)
        for plan in image_stages:
            _materialize_stage(plan)
        event(run_dir, {"event": "render_started", "timestamp": now(), "train_run": train_run, "generator": "comfyui", "executor": resolved_executor, "endpoint": endpoint, "lora_stage": lora_stage, "model_patch_stage": model_patch_stage, "image_stages": image_stages})
        generated = 0
        for item, seed in pairs:
            patches = frozen.get("workflow_patches", {})
            patched = patch_workflow(workflow, patches, prompt=item.get("prompt", ""), negative_prompt=item.get("negative_prompt", ""), seed=seed, checkpoint=lora_name, model_patch=model_patch_name, item=item, image_values=_image_values_for_item(item, patches, image_stages))
            patched = insert_lora_loader(patched, frozen.get("lora_insert"), lora_name)
            case_application = checkpoint_application(frozen, patched, lora_name=lora_name)
            prompt_id = client.queue(patched)
            with stdout_log.open("a", encoding="utf-8") as handle:
                handle.write(f"queued {item['id']} seed={seed} prompt_id={prompt_id}\n")
            for index, image in enumerate(client.wait(prompt_id)):
                suffix = Path(image.get("filename", "image.png")).suffix or ".png"
                seed_segment = "" if seed is None else f"_seed{seed}"
                relative = f"samples/images/{item['id']}{seed_segment}_{index}{suffix}"
                image_path = run_dir / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(client.download(image))
                # A parameter the workflow fixes was never Kura's to set, and Kura
                # cannot read it back without a binding. Record it as unknown
                # rather than repeating a promptset value that did not reach the render.
                record = {"file": relative, "train_run": train_run, "prompt_id": item["id"], "prompt": None if "prompt" in workflow_fixed else item.get("prompt", ""), "negative_prompt": None if "negative_prompt" in workflow_fixed else item.get("negative_prompt", ""), "seed": seed, "checkpoint_path": checkpoint.get("path"), "checkpoint_hash": checkpoint.get("hash"), "checkpoint_application": case_application, "comfyui_lora_name": lora_name, "workflow_digest": inputs.get("workflow", {}).get("digest"), "promptset_digest": inputs.get("promptset", {}).get("digest"), "comfyui_prompt_id": prompt_id, "patch_inputs": {name: item[name] for name in patches if name in item and name not in ("prompt", "negative_prompt")}, "workflow_fixed": list(workflow_fixed), "created": now()}
                with images_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                event(run_dir, {"event": "image_generated", "timestamp": now(), "prompt_id": item["id"], "seed": seed, "file": relative})
                generated += 1
        if generated == 0:
            raise RuntimeError("ComfyUI completed without returning any images")
        status(run_dir, state="completed", ended=now(), exit_code=0)
        write_realization(run_dir, train_run=train_run, executor=resolved_executor, generator="comfyui", state="completed", workflow_fixed=list(workflow_fixed), endpoint=endpoint, workflow_digest=inputs.get("workflow", {}).get("digest"), promptset_digest=inputs.get("promptset", {}).get("digest"), checkpoint_hash=checkpoint.get("hash"), comfyui_lora_name=lora_name, comfyui_model_patch_name=model_patch_name, lora_stage=lora_stage, model_patch_stage=model_patch_stage, image_count=generated)
        event(run_dir, {"event": "render_completed", "timestamp": now(), "count": generated})
        return 0
    except Exception as exc:
        stdout_log = run_dir / "logs" / "stdout.log"
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        with stdout_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
        status(run_dir, state="failed", ended=now(), exit_code=1)
        write_realization(run_dir, train_run=train_run, executor=resolved_executor, generator="comfyui", state="failed", workflow_fixed=list(workflow_fixed), endpoint=endpoint, workflow_digest=inputs.get("workflow", {}).get("digest"), promptset_digest=inputs.get("promptset", {}).get("digest"), checkpoint_hash=checkpoint.get("hash"), comfyui_lora_name=lora_name, comfyui_model_patch_name=model_patch_name, lora_stage=lora_stage, model_patch_stage=model_patch_stage, error=str(exc))
        event(run_dir, {"event": "render_failed", "timestamp": now(), "error": str(exc)})
        return 1
    finally:
        _cleanup_stage(lora_stage)
        _cleanup_stage(model_patch_stage)
        for plan in image_stages:
            _cleanup_stage(plan)
