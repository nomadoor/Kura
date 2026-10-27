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
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any

from kura.run_results import render_output_dir
from kura.install_source import kura_provenance
from kura.yamlio import YAMLError, load_run_yaml, load_yaml, load_yaml_mapping
from kura.comfyui_models import endpoint_fingerprint, merged_registry, resolve_model_specs, visible_model_refs
from kura.fsio import atomic_write_json
from kura.records import realization_reference, record as as_record
from kura.fsio import atomic_write_yaml as dump_yaml
from kura.images import effective_image
from kura.run_envelope import gpu_choice_problem, render_executor
from kura.workspace import comfyui_endpoint_problem, plain_relative_folder, validate_workspace_config


# Promptset keys Kura owns directly. Every other key in a promptset item must be
# bound to a workflow node/field through run.yaml `workflow_patches`, so that a
# promptset and a workflow can never silently disagree about which parameters exist.
PROMPTSET_CORE_KEYS = frozenset({"id", "prompt", "negative_prompt", "seeds", "meta"})

# Core keys that must be bound when the promptset supplies them. Without a
# binding the value is carried into filenames and images.jsonl while the
# workflow renders its own hardcoded one.
CORE_KEYS_REQUIRING_BINDINGS = ("prompt", "negative_prompt", "seed")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return load_yaml_mapping(path)


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
    source = load_run_yaml(source_path)
    if not isinstance(source, dict) or source.get("type") != "train":
        raise ValueError(f"render inputs.train_run must reference a train run: {train_run}")
    if source.get("id") not in (None, train_run):
        raise ValueError(f"render inputs.train_run ID does not match its run.yaml: {train_run}")
    return train_run


CASE_KEYS = frozenset({"id", "values", "checkpoint", "meta"})
CHECKPOINT_KEYS = frozenset({"id", "path", "hash"})
ASSET_PATCH_NAMES = frozenset({"lora", "model_patch"})


def _validated_checkpoint(value: Any, *, context: str, require_id: bool) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = sorted(str(key) for key in value if key not in CHECKPOINT_KEYS)
    if unknown:
        raise ValueError(f"{context} has unsupported keys: {', '.join(unknown)}")
    checkpoint_id = value.get("id")
    if (require_id or checkpoint_id is not None) and not is_safe_component(checkpoint_id):
        raise ValueError(f"{context}.id must be a single safe file name, not a path: {checkpoint_id!r}")
    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{context}.path must be a non-empty path")
    checkpoint_hash = value.get("hash")
    if checkpoint_hash is not None and not isinstance(checkpoint_hash, str):
        raise ValueError(f"{context}.hash must be a string or null")
    return deepcopy(value)


def authored_cases(path: Path) -> list[dict[str, Any]]:
    """Load the canonical one-row-per-logical-render-case surface."""
    cases: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cases:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(item, dict):
            raise ValueError(f"cases:{line_number}: each row must be a mapping")
        unknown = sorted(str(key) for key in item if key not in CASE_KEYS)
        if unknown:
            raise ValueError(f"cases:{line_number}: unsupported keys: {', '.join(unknown)}")
        case_id = _validate_prompt_id(
            item.get("id"),
            line_number,
            surface="cases",
            require_prompt_wording=False,
        )
        if case_id in seen:
            raise ValueError(f"cases:{line_number}: duplicate id {case_id!r} (already used on line {seen[case_id]})")
        values = item.get("values")
        if not isinstance(values, dict):
            raise ValueError(f"cases:{line_number}: values must be a mapping")
        if not all(isinstance(name, str) and is_safe_component(name) for name in values):
            raise ValueError(f"cases:{line_number}: values keys must be plain names")
        meta = item.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"cases:{line_number}: meta must be a mapping")
        checkpoint = _validated_checkpoint(item.get("checkpoint"), context=f"cases:{line_number}: checkpoint", require_id=True)
        normalized = {"id": case_id, "values": deepcopy(values)}
        if checkpoint is not None:
            normalized["checkpoint"] = checkpoint
        if meta:
            normalized["meta"] = deepcopy(meta)
        cases.append(normalized)
        seen[case_id] = line_number
    if not cases:
        raise ValueError("inputs.cases must contain at least one case")
    return cases


def load_resolved_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"resolved/cases.jsonl:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("index"), int) or not isinstance(item.get("values"), dict):
            raise ValueError(f"resolved/cases.jsonl:{line_number}: invalid resolved case")
        cases.append(item)
    if not cases:
        raise ValueError("resolved/cases.jsonl contains no cases")
    return cases


def _workflow_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".kura.yaml")
    if not sidecar.is_file():
        return {}
    values = load_yaml_mapping(sidecar)
    unknown = sorted(str(key) for key in values if key not in SIDECAR_KEYS)
    if unknown:
        raise ValueError(f"{sidecar.name} has unsupported key(s): {', '.join(unknown)}; use {', '.join(sorted(SIDECAR_KEYS))}")
    return values


def event(run_dir: Path, payload: dict[str, Any]) -> None:
    """Append a render event durably; the case progress in status follows from these."""
    from kura.executors.common import append_run_event

    append_run_event(run_dir, payload)


def _append_runtime_warning(stdout_log: Path, message: str) -> None:
    """Leave best-effort diagnostics when a failure-record fallback also breaks."""
    try:
        with stdout_log.open("a", encoding="utf-8") as handle:
            handle.write(f"warning: {message}\n")
    except OSError:
        pass


def status(run_dir: Path, **changes: Any) -> None:
    """Change the render's status under the run's status lock, fenced against a replaced runner."""
    from kura.executors.common import _mutate_run_status

    _mutate_run_status(run_dir, lambda current: current.update(changes))


def _write_realization_file(run_dir: Path, *, timestamp: str | None = None, **details: Any) -> str:
    realization_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    path = run_dir / "realizations" / f"{realization_id}.json"
    path.parent.mkdir(exist_ok=True)
    atomic_write_json(path, as_record("realization", {"id": realization_id, "timestamp": timestamp or now(), **details}))
    return realization_reference(realization_id)


def write_realization(run_dir: Path, *, status_changes: dict[str, Any] | None = None, **details: Any) -> None:
    """Write the render's realization, then point status at it with `status_changes` in the same step.

    The realization's timestamp is the status's `ended`, so both name the same instant.
    """
    changes = status_changes or {}
    ended = changes.get("ended") if isinstance(changes.get("ended"), str) else None
    status(run_dir, **changes, last_realization=_write_realization_file(run_dir, timestamp=ended, **details))


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
    if "checkpoint" in patches:
        raise ValueError(
            "workflow_patches.checkpoint would send a checkpoint's file path to ComfyUI as a base model name; "
            "Kura stages only LoRA and model-patch files, so choose the base model in the workflow itself"
        )
    probe = deepcopy(workflow)
    for name, patch in patches.items():
        node, field, _ = _binding_target(name, patch)
        try:
            _set_path(probe, node, field, None)
        except ValueError as exc:
            raise ValueError(f"workflow_patches.{name} does not match this workflow: {exc}") from exc


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


# The strength a trained LoRA is rendered at unless the run says otherwise.
# LoRAs are conventionally trained to look right at this strength, and a
# workflow's own LoRA node holds whatever value was last left in it.
DEFAULT_LORA_STRENGTH = 0.8


def bound_lora_strength_inputs(workflow: dict[str, Any], patches: Any) -> list[str]:
    """The strength inputs of the `workflow_patches.lora` node Kura may set.

    An input linked to another node, or set per case by another binding, is
    not Kura's to set.
    """
    binding = patches.get("lora") if isinstance(patches, dict) else None
    if not isinstance(binding, dict) or not isinstance(binding.get("node"), str):
        return []
    node = workflow.get(binding["node"])
    inputs = node.get("inputs") if isinstance(node, dict) and isinstance(node.get("inputs"), dict) else {}
    bound = {(patch.get("node"), patch.get("field")) for name, patch in patches.items() if name != "lora" and isinstance(patch, dict)}
    return [
        key for key in ("strength_model", "strength_clip")
        if isinstance(inputs.get(key), (int, float)) and not isinstance(inputs.get(key), bool) and (binding["node"], f"inputs.{key}") not in bound
    ]


def apply_bound_lora_strength(workflow: dict[str, Any], patches: dict[str, Any], strength: float) -> dict[str, Any]:
    """Set the strength of the workflow LoRA node bound by `workflow_patches.lora`."""
    keys = bound_lora_strength_inputs(workflow, patches)
    for key in keys:
        workflow[patches["lora"]["node"]]["inputs"][key] = strength
    return workflow


def _link(node: str, output: int) -> list[Any]:
    return [node, output]


# lora_insert keys by kind: model_only inserts LoraLoaderModelOnly, model_clip inserts LoraLoader.
LORA_INSERT_KEYS = {
    "model_only": frozenset({"kind", "model_node", "model_output", "strength_model"}),
    "model_clip": frozenset({"kind", "model_node", "model_output", "clip_node", "clip_output", "strength_model", "strength_clip"}),
}


def _lora_insert_from_sidecar(sidecar: dict[str, Any]) -> dict[str, Any] | None:
    """The one form of a sidecar's lora_insert, as the spec insert_lora_loader applies."""
    raw = sidecar.get("lora_insert")
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("kind") not in LORA_INSERT_KEYS:
        raise ValueError("lora_insert must be a mapping whose kind is model_only or model_clip")
    kind = raw["kind"]
    unknown = sorted(str(key) for key in raw if key not in LORA_INSERT_KEYS[kind])
    if unknown:
        raise ValueError(f"lora_insert kind {kind} takes {', '.join(sorted(LORA_INSERT_KEYS[kind]))}, not {', '.join(unknown)}")
    for key in ("model_node", "clip_node"):
        if key in raw or key == "model_node":
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"lora_insert.{key} must be a workflow node id written as a string, such as '37', not {raw.get(key)!r}")
    for key in ("model_output", "clip_output"):
        if key in raw and (isinstance(raw[key], bool) or not isinstance(raw[key], int) or raw[key] < 0):
            raise ValueError(f"lora_insert.{key} must be a non-negative integer, not {raw[key]!r}")
    for key in ("strength_model", "strength_clip"):
        if key in raw and (isinstance(raw[key], bool) or not isinstance(raw[key], (int, float))):
            raise ValueError(f"lora_insert.{key} must be a number, not {raw[key]!r}")
    spec: dict[str, Any] = {
        "class_type": "LoraLoaderModelOnly" if kind == "model_only" else "LoraLoader",
        "model_node": raw["model_node"],
        "model_output": raw.get("model_output", 0),
        "strength_model": raw.get("strength_model", DEFAULT_LORA_STRENGTH),
    }
    if kind == "model_clip":
        spec["clip_node"] = raw.get("clip_node", raw["model_node"])
        spec["clip_output"] = raw.get("clip_output", 1)
        spec["strength_clip"] = raw.get("strength_clip", DEFAULT_LORA_STRENGTH)
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
    checkpoint_record: dict[str, Any] | None = None,
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
            render_settings = frozen.get("render") if isinstance(frozen.get("render"), dict) else {}
            if lora_name and isinstance(render_settings.get("lora_strength"), (int, float)):
                inputs = dict(inputs)
                for key in bound_lora_strength_inputs(workflow, patches):
                    inputs[key] = float(render_settings["lora_strength"])
            if isinstance(inputs.get("strength_model"), (int, float)):
                application["strength_model"] = inputs["strength_model"]
            if isinstance(inputs.get("strength_clip"), (int, float)):
                application["strength_clip"] = inputs["strength_clip"]
        return application

    checkpoint = checkpoint_record if checkpoint_record is not None else frozen.get("inputs", {}).get("checkpoint")
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


@lru_cache(maxsize=8)
def _endpoint_os(endpoint: str | None) -> str | None:
    """The OS a ComfyUI endpoint runs on (`posix` or `nt`), or None when it does not say."""
    if not isinstance(endpoint, str) or not endpoint:
        return None
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/system_stats", timeout=5) as response:
            stats = json.loads(response.read())
    except Exception:
        return None
    system = stats.get("system") if isinstance(stats, dict) else None
    value = system.get("os") if isinstance(system, dict) else None
    return value if value in ("posix", "nt") else None


def _stage_mode(value: Any, name: str, endpoint: str | None) -> str:
    """How to expose a staged file to ComfyUI: as configured, or decided here.

    `auto` (the default) links when ComfyUI runs on Linux or in WSL, which can
    follow a link Kura makes, and copies when ComfyUI runs on Windows, which
    cannot follow a link made in WSL, when it does not report its OS, or when
    Kura itself runs on Windows, where making a link needs privileges.
    """
    # Read from a frozen manifest, which an older Kura may have written in another spelling.
    mode = str(value or "auto").strip().lower()
    if mode not in ("auto", "symlink", "copy"):
        raise ValueError(f"comfyui.{name} must be auto, symlink, or copy")
    if mode != "auto":
        return mode
    if os.name != "nt" and _endpoint_os(endpoint) == "posix":
        return "symlink"
    return "copy"


def _frozen_stage(comfyui: dict[str, Any], subdir_key: str, cleanup_key: str) -> tuple[str, str]:
    """A stage subfolder and cleanup as a frozen manifest holds them.

    Compile freezes both in their one form. A manifest an older Kura froze may hold
    null or an empty string (the default), a folder wrapped in slashes, or a cleanup
    in another case; it is read as that Kura read it, never rewritten.
    """
    subdir = str(comfyui.get(subdir_key) or "Kura_tmp").strip("/\\")
    if not plain_relative_folder(subdir):
        raise ValueError(f"comfyui.{subdir_key} must be a relative folder name such as Kura_tmp, not {subdir!r}")
    cleanup = str(comfyui.get(cleanup_key) or "remove_after_render").strip().lower()
    if cleanup not in ("remove_after_render", "keep"):
        raise ValueError(f"comfyui.{cleanup_key} must be remove_after_render or keep, not {cleanup!r}")
    return subdir, cleanup


# How a manifest an older Kura froze wrote `render.lora_stage: off` (no longer a setting).
_OLD_LORA_STAGE_OFF = ("0", "false", "off", "none", "no")


def _lora_stage_plan(workspace: Path, run_dir: Path, frozen: dict[str, Any], checkpoint: dict[str, Any], endpoint: str | None = None) -> dict[str, Any] | None:
    """Where a checkpoint is staged so ComfyUI loads it as a LoRA; None when the workflow takes no LoRA.

    A checkpoint path is always a file path: one Kura cannot stage is refused, never passed on as a LoRA name.
    """
    if not checkpoint or ("lora" not in frozen.get("workflow_patches", {}) and not frozen.get("lora_insert")):
        return None
    render = frozen.get("render") if isinstance(frozen.get("render"), dict) else {}
    if "lora_stage" in render and str(render["lora_stage"]).strip().lower() in _OLD_LORA_STAGE_OFF:
        raise ValueError(
            "this render was compiled by an older Kura with render.lora_stage off, which sent the checkpoint path to ComfyUI "
            "as the LoRA name; Kura now always stages the checkpoint, so create a new render run to render it"
        )
    source = _workspace_path(workspace, checkpoint.get("path"))
    if source is None or not source.is_file() or source.suffix != ".safetensors":
        raise ValueError(f"a LoRA render needs each checkpoint to be a .safetensors file; {checkpoint.get('path')!r} is not")
    comfyui = frozen.get("comfyui") if isinstance(frozen.get("comfyui"), dict) else {}
    lora_dir = _workspace_path(workspace, comfyui.get("lora_dir"))
    if lora_dir is None:
        raise ValueError("a local LoRA render stages each checkpoint into ComfyUI's LoRA folder; set comfyui.lora_dir in workspace.yaml")
    stage_subdir, cleanup = _frozen_stage(comfyui, "lora_stage_subdir", "lora_stage_cleanup")
    stage_dir = (lora_dir / stage_subdir).resolve()
    mode = _stage_mode(comfyui.get("lora_stage_mode"), "lora_stage_mode", endpoint)
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


def _model_patch_stage_plan(workspace: Path, run_dir: Path, frozen: dict[str, Any], checkpoint: dict[str, Any], endpoint: str | None = None) -> dict[str, Any] | None:
    if "model_patch" not in frozen.get("workflow_patches", {}):
        return None
    if not checkpoint:
        return None
    source = _workspace_path(workspace, checkpoint.get("path"))
    if source is None or not source.is_file() or source.suffix != ".safetensors":
        raise ValueError(f"a model_patch render needs each checkpoint to be a .safetensors file; {checkpoint.get('path')!r} is not")
    comfyui = frozen.get("comfyui") if isinstance(frozen.get("comfyui"), dict) else {}
    directory = _workspace_path(workspace, comfyui.get("model_patches_dir"))
    if directory is None:
        raise ValueError("a local model_patch render stages each checkpoint into ComfyUI's model patch folder; set comfyui.model_patches_dir in workspace.yaml")
    stage_subdir, cleanup = _frozen_stage(comfyui, "model_patch_stage_subdir", "model_patch_stage_cleanup")
    stage_dir = (directory / stage_subdir).resolve()
    mode = _stage_mode(comfyui.get("model_patch_stage_mode"), "model_patch_stage_mode", endpoint)
    target = stage_dir / _safe_stage_name(run_dir.name, source)
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
    stage_subdir, cleanup = _frozen_stage(comfyui, "input_stage_subdir", "input_stage_cleanup")
    # Copy, not symlink: ComfyUI resolves LoadImage paths and rejects one that
    # leaves its input directory, so a symlinked input fails validation at queue
    # time even though a symlinked model loads fine.
    # Read from a frozen manifest, which an older Kura may have written in another spelling.
    mode = str(comfyui.get("input_stage_mode") or "copy").strip().lower()
    if mode == "auto":
        mode = "copy"  # ComfyUI rejects symlinked LoadImage inputs
    if mode not in ("symlink", "copy"):
        raise ValueError("comfyui.input_stage_mode must be auto, symlink, or copy")
    if mode == "symlink":
        print(
            "warning: comfyui.input_stage_mode=symlink is set, but ComfyUI rejects symlinked LoadImage inputs "
            "(\"Invalid image file\"). Use copy unless this endpoint is known to accept them.",
            flush=True,
        )
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


STAGE_KINDS = (
    ("lora_dir", "lora_stage_subdir", "lora_stage_cleanup"),
    ("model_patches_dir", "model_patch_stage_subdir", "model_patch_stage_cleanup"),
    ("input_dir", "input_stage_subdir", "input_stage_cleanup"),
)


def remove_leftover_stages(workspace: Path, run_dir: Path) -> list[str]:
    """Remove files a killed render of this run left in ComfyUI's folders.

    Staged names always start with the run id, so only this run's files match;
    a stage kind set to `keep` is left alone.
    """
    try:
        frozen = load_yaml(run_dir / "resolved" / "manifest.lock.yaml") or {}
    except (OSError, YAMLError):
        return []
    if not isinstance(frozen, dict):
        return []
    comfyui = frozen.get("comfyui") if isinstance(frozen.get("comfyui"), dict) else {}
    removed = []
    for dir_key, subdir_key, cleanup_key in STAGE_KINDS:
        base = _workspace_path(workspace, comfyui.get(dir_key))
        try:
            subdir, cleanup = _frozen_stage(comfyui, subdir_key, cleanup_key)
        except ValueError:
            # A manifest frozen by an older Kura was never validated; cleanup never leaves its base.
            continue
        if cleanup == "keep" or base is None:
            continue
        # Run ids end in a random hex suffix (`<time>_<slug>_<hex>`), so `<id>-` never starts another run's id.
        for path in sorted((base / subdir).glob(f"{run_dir.name}-*")):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(str(path))
    return removed


def _freeze_comfyui_config(comfyui: Any, *, include_remote: bool) -> dict[str, Any]:
    if not isinstance(comfyui, dict):
        return {}
    allowed = ["lora_dir", "lora_stage_subdir", "lora_stage_mode", "lora_stage_cleanup", "model_patches_dir", "model_patch_stage_subdir", "model_patch_stage_mode", "model_patch_stage_cleanup", "input_dir", "input_stage_subdir", "input_stage_mode", "input_stage_cleanup"]
    if include_remote:
        allowed.append("model_registry")
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


def _materialize_visible(plan: dict[str, Any], ensure_visible: Any) -> None:
    """Stage a file and confirm ComfyUI lists it, copying when a link was not listed."""
    _materialize_stage(plan)
    try:
        ensure_visible(plan)
    except ValueError:
        if plan.get("mode") != "symlink" or not plan.get("created"):
            raise
        Path(str(plan["target"])).unlink(missing_ok=True)
        plan.update({"mode": "copy", "created": False, "fell_back_from": "symlink"})
        _materialize_stage(plan)
        ensure_visible(plan)


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
    try:
        if name in client.model_patch_names():
            return
        time.sleep(0.5)
        if name in client.model_patch_names():
            return
    except RuntimeError as exc:
        raise ValueError(f"ComfyUI model-patch visibility could not be checked; endpoint={endpoint}; error={exc}") from exc
    raise ValueError(f"ComfyUI model-patch stage is not visible; endpoint={endpoint}; model_patch_name={name}")


def _lora_name_visible(client: Any, lora_name: str) -> bool:
    if not lora_name or not hasattr(client, "lora_names"):
        return True
    return lora_name in client.lora_names()


def _ensure_lora_stage_visible(client: Any, endpoint: str, plan: dict[str, Any] | None) -> None:
    if not plan:
        return
    try:
        if _lora_name_visible(client, str(plan.get("lora_name", ""))):
            return
        time.sleep(0.5)
        if _lora_name_visible(client, str(plan.get("lora_name", ""))):
            return
    except RuntimeError as exc:
        raise ValueError(
            "ComfyUI LoRA visibility could not be checked because object_info is unavailable. "
            f"endpoint={endpoint}; error={exc}. "
            f"Run `kura doctor comfyui --endpoint {endpoint}` to check the endpoint."
        ) from exc
    raise ValueError(
        "ComfyUI LoRA stage is not visible from the configured endpoint. "
        f"endpoint={endpoint}; lora_name={plan.get('lora_name')}; lora_dir={Path(str(plan.get('target'))).parent.parent}. "
        f"Run `kura doctor comfyui --endpoint {endpoint} --probe-stage` to verify staging, "
        "then set comfyui.lora_dir to a LoRA directory used by that ComfyUI instance and recompile the render run."
    )


def _validate_prompt_id(
    value: Any,
    line_number: int,
    *,
    surface: str = "promptset",
    require_prompt_wording: bool = True,
) -> str:
    """Ids become file names under `resolved/` and `samples/`, so they must be inert.

    An id is joined into output paths; anything that can traverse or collide would
    let a promptset write outside its run or silently overwrite another case.
    """
    if not isinstance(value, str) or not value:
        required = "id and prompt are required" if require_prompt_wording else "id is required"
        raise ValueError(f"{surface}:{line_number}: {required}")
    if not is_safe_component(value):
        raise ValueError(f"{surface}:{line_number}: id must be a single safe file name, not a path: {value!r}")
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


def _reconcile_case_values(cases: list[dict[str, Any]], patches: Any, *, default_seed: Any, workflow_fixed: Any, has_checkpoint_consumer: bool, requires_checkpoint: bool) -> None:
    bound = set(patches) if isinstance(patches, dict) else set()
    fixed = set(normalized_workflow_fixed(workflow_fixed))
    unknown_fixed = sorted(fixed - set(CORE_KEYS_REQUIRING_BINDINGS))
    if unknown_fixed:
        raise ValueError(f"render.workflow_fixed only accepts {', '.join(CORE_KEYS_REQUIRING_BINDINGS)}; remove: {', '.join(unknown_fixed)}")
    conflicting = sorted(fixed & bound)
    if conflicting:
        raise ValueError(f"render.workflow_fixed and workflow_patches both claim {', '.join(conflicting)}")
    if "seed" in fixed and default_seed is not None:
        raise ValueError("render.workflow_fixed includes seed, so render.default_seed must be null")
    if default_seed is not None and "seed" not in bound:
        raise ValueError("render.default_seed needs a workflow_patches.seed binding; without one the workflow renders its own seed")
    required = bound - ASSET_PATCH_NAMES - {"negative_prompt"}
    for case in cases:
        values = case["values"]
        forbidden_assets = sorted(set(values) & ASSET_PATCH_NAMES)
        if forbidden_assets:
            raise ValueError(f"case {case['id']!r} sets asset values directly: {', '.join(forbidden_assets)}; use case.checkpoint")
        forbidden_fixed = sorted(set(values) & fixed)
        if forbidden_fixed:
            raise ValueError(f"case {case['id']!r} sets workflow-owned values: {', '.join(forbidden_fixed)}")
        if "negative_prompt" in bound and "negative_prompt" not in values:
            values["negative_prompt"] = ""
        if "seed" in bound and "seed" not in values and default_seed is not None:
            values["seed"] = default_seed
        unbound = sorted(set(values) - bound)
        if unbound:
            raise ValueError(f"case {case['id']!r} declares values with no workflow_patches binding: {', '.join(unbound)}")
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"case {case['id']!r} has no value for workflow_patches: {', '.join(missing)}")
        if case.get("checkpoint") and not has_checkpoint_consumer:
            raise ValueError(f"case {case['id']!r} has a checkpoint but this workflow has no lora, checkpoint, or model_patch binding and no lora_insert")
        if requires_checkpoint and not case.get("checkpoint"):
            raise ValueError(f"case {case['id']!r} requires a checkpoint because this workflow applies a lora, checkpoint, or model_patch")


def _legacy_render_cases(items: list[dict[str, Any]], checkpoint: dict[str, Any] | None, *, patches: Any, default_seed: Any, workflow_fixed: list[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    bound = set(patches) if isinstance(patches, dict) else set()
    for item in items:
        seeds = [None] if "seed" in workflow_fixed else (item.get("seeds") or [default_seed])
        for seed in seeds:
            if seed is None and "seed" not in workflow_fixed:
                continue
            case_id = item["id"] if seed is None else f"{item['id']}_seed{seed}"
            if not is_safe_component(case_id) or case_id in seen:
                raise ValueError(f"legacy promptset expands to an unsafe or duplicate case id: {case_id!r}")
            values = {name: deepcopy(value) for name, value in item.items() if name not in PROMPTSET_CORE_KEYS}
            if "prompt" in bound and "prompt" not in workflow_fixed:
                values["prompt"] = item.get("prompt", "")
            if "negative_prompt" in bound and "negative_prompt" not in workflow_fixed:
                values["negative_prompt"] = item.get("negative_prompt", "")
            if "seed" in bound and "seed" not in workflow_fixed:
                values["seed"] = seed
            case = {"id": case_id, "source_id": item["id"], "values": values}
            if checkpoint is not None:
                case["checkpoint"] = deepcopy(checkpoint)
            if isinstance(item.get("meta"), dict) and item["meta"]:
                case["meta"] = deepcopy(item["meta"])
            cases.append(case)
            seen.add(case_id)
    if not cases:
        raise ValueError("promptset has no seeds and render.default_seed is not set")
    return cases


def _freeze_case_images(run_dir: Path, source_path: Path, cases: list[dict[str, Any]], patches: Any) -> list[dict[str, Any]]:
    flattened = [{"id": case["id"], **deepcopy(case["values"])} for case in cases]
    records = _freeze_promptset_images(run_dir, source_path, flattened, patches)
    frozen_by_id = {item["id"]: item for item in flattened}
    image_names = image_patch_names(patches)
    for case in cases:
        item = frozen_by_id[case["id"]]
        for name in image_names:
            case["values"][name] = item[name]
    return records


def _freeze_case_checkpoints(workspace: Path, cases: list[dict[str, Any]]) -> None:
    identities: dict[str, tuple[str, Any]] = {}
    for case in cases:
        checkpoint = case.get("checkpoint")
        if not isinstance(checkpoint, dict):
            continue
        candidate = _workspace_path(workspace, checkpoint.get("path"))
        if candidate is None or not candidate.is_file():
            raise ValueError(f"case {case['id']} checkpoint {checkpoint.get('path')!r} is not a file; a checkpoint path names a file Kura can hash")
        observed = digest(candidate)
        if checkpoint.get("hash") is not None and checkpoint["hash"] != observed:
            raise ValueError(
                f"case {case['id']} checkpoint hash {checkpoint['hash']!r} does not match {checkpoint['path']} ({observed}); "
                "fix the hash or the path, or remove the hash to record the file's"
            )
        checkpoint["hash"] = observed
        checkpoint_id = checkpoint.get("id")
        if not isinstance(checkpoint_id, str):
            continue
        identity = (os.path.normpath(str(Path(str(checkpoint.get("path"))).expanduser())), checkpoint.get("hash"))
        previous = identities.get(checkpoint_id)
        if previous is not None and previous != identity:
            raise ValueError(
                f"case checkpoint id {checkpoint_id!r} refers to different artifacts; "
                "one checkpoint id must map to one path and hash across the case queue"
            )
        identities[checkpoint_id] = identity


def _indexed_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": case["id"],
            "index": index,
            **({"source_id": case["source_id"]} if "source_id" in case else {}),
            "values": case["values"],
            **({"checkpoint": case["checkpoint"]} if "checkpoint" in case else {}),
            **({"meta": case["meta"]} if "meta" in case else {}),
        }
        for index, case in enumerate(cases, 1)
    ]


def _runtime_checkpoint_provenance(runtime_cases: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for runtime_case in runtime_cases:
        checkpoint = runtime_case["checkpoint"]
        identity = (
            checkpoint.get("id"),
            checkpoint.get("path"),
            checkpoint.get("hash"),
            runtime_case["lora_name"],
            runtime_case["model_patch_name"],
        )
        if identity in seen or not any(identity):
            continue
        seen.add(identity)
        checkpoints.append({
            "id": checkpoint.get("id"),
            "path": checkpoint.get("path"),
            "hash": checkpoint.get("hash"),
            "checkpoint_application": runtime_case["checkpoint_application"],
            "comfyui_lora_name": runtime_case["lora_name"],
            "comfyui_model_patch_name": runtime_case["model_patch_name"],
            "lora_stage": runtime_case["lora_stage"],
            "model_patch_stage": runtime_case["model_patch_stage"],
        })
    details: dict[str, Any] = {"checkpoints": checkpoints} if checkpoints else {}
    if len(checkpoints) == 1:
        checkpoint = checkpoints[0]
        details.update({
            "checkpoint_hash": checkpoint["hash"],
            "comfyui_lora_name": checkpoint["comfyui_lora_name"],
            "comfyui_model_patch_name": checkpoint["comfyui_model_patch_name"],
            "lora_stage": checkpoint["lora_stage"],
            "model_patch_stage": checkpoint["model_patch_stage"],
        })
    return details


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
            from kura.executors.common import sleep_checking_stop

            sleep_checking_stop(1)
        raise TimeoutError(f"ComfyUI prompt timed out after {self.timeout} seconds: {prompt_id}")

    def cancel(self, prompt_id: str) -> None:
        """Withdraw Kura's own prompt: delete it if queued, interrupt it only if it is the one running.

        The queue belongs to the user; nothing else in it is touched.
        """
        self._json("/queue", {"delete": [prompt_id]})
        queue = self._json("/queue")
        running = queue.get("queue_running") if isinstance(queue, dict) else None
        if isinstance(running, list) and any(isinstance(item, list) and len(item) > 1 and item[1] == prompt_id for item in running):
            # A server that knows prompt_id interrupts only this prompt, so one
            # that started after the queue read is left alone.
            self._json("/interrupt", {"prompt_id": prompt_id})

    def download(self, image: dict[str, Any]) -> bytes:
        query = urllib.parse.urlencode({key: image.get(key, "") for key in ("filename", "subfolder", "type")})
        with urllib.request.urlopen(f"{self.endpoint}/view?{query}", timeout=30) as response:
            return response.read()


# Facts compile writes into the run lock. In run.yaml they would be copied and
# then overwritten, so a value there would be silently ignored.
COMPILED_RENDER_KEYS = ("lora_insert", "comfyui", "comfyui_models", "comfyui_model_registry", "comfyui_endpoint_identity", "comfyui_required_models", "promptset_images", "_kura")
RENDER_SETTINGS = ("output_dir", "timeout_sec", "default_seed", "workflow_fixed", "lora_strength")
# Every key a render run.yaml may hold, by level. `workflow_patches` is closed by
# validate_patch_bindings, and `inputs.cases` is checked where compile reads it.
RENDER_RUN_SCHEMA: dict[str, frozenset[str]] = {
    "": frozenset({
        "schema_version", "id", "type", "created", "intent", "inputs", "generator", "executor", "compute",
        "workflow_patches", "render",
        # Recorded as written (the lora-evaluation skill's plan); Kura reads nothing in it.
        "evaluation",
    }),
    "inputs": frozenset({"train_run", "workflow", "cases", "promptset", "checkpoint"}),
    "inputs.workflow": frozenset({"path", "digest"}),
    "render": frozenset(RENDER_SETTINGS),
    "generator": frozenset({"name", "endpoint"}),
    "executor": frozenset({"name"}),
    "compute": frozenset({"gpu"}),
}
# Levels a render run.yaml must write as a mapping; the others may be left out.
_REQUIRED_RENDER_LEVELS = frozenset({"inputs", "inputs.workflow"})
SIDECAR_KEYS = frozenset({"source", "provenance", "models", "lora_insert"})
DEFAULT_RENDER_TIMEOUT_SEC = 600


def _refuse_missing_models(missing: list[dict[str, Any]], endpoint: str) -> None:
    """A local render needs every model its workflow loads; compile and launch both refuse here."""
    if missing:
        labels = ", ".join(f"{item['class_type']}.{item['input']}={item['name']}" for item in missing)
        raise ValueError(
            f"ComfyUI at {endpoint} does not have model(s) this workflow loads: {labels}. "
            "Add them to that ComfyUI's model folders or change the workflow; a local render never downloads models."
        )


def _refuse_runpod_model_patch(executor: Any, patches: Any) -> None:
    """A RunPod render cannot stage a model patch; compile refuses it before any Pod, launch for a run compiled earlier."""
    if executor == "runpod" and isinstance(patches, dict) and "model_patch" in patches:
        raise ValueError("ComfyUI model patch staging is not supported for the runpod executor; render a model_patch workflow locally")


def render_timeout_sec(frozen: dict[str, Any]) -> int:
    """How long one ComfyUI request may take; local and RunPod renders both read it here."""
    render = frozen.get("render") if isinstance(frozen.get("render"), dict) else {}
    # Compile accepts only a whole number of at least 1; int() reads a manifest an older Kura froze.
    return int(render.get("timeout_sec", DEFAULT_RENDER_TIMEOUT_SEC))


def _validate_render_run_fields(run: dict[str, Any]) -> None:
    """Refuse a key no render reader uses, and a value Kura decides from in any other form, in one message."""
    problems: list[str] = []
    for level, keys in RENDER_RUN_SCHEMA.items():
        holder: Any = run
        for part in level.split(".") if level else ():
            holder = holder.get(part) if isinstance(holder, dict) else None
        required = level in _REQUIRED_RENDER_LEVELS and (level == "inputs" or isinstance(run.get("inputs"), dict))
        if level and (holder is not None or required) and not isinstance(holder, dict):
            problems.append(f"{level} must be a mapping of {', '.join(sorted(keys))}, not {holder!r}")
            continue
        for key in holder if isinstance(holder, dict) else ():
            if key in keys:
                continue
            if not level and key in COMPILED_RENDER_KEYS:
                hint = "; to change the inserted LoRA's strength, set render.lora_strength" if key == "lora_insert" else ""
                problems.append(f"{key} is written by compile from the workflow, its sidecar, and workspace.yaml{hint}")
            else:
                problems.append(f"{level + '.' if level else ''}{key} is not a render run setting")
    generator = run.get("generator") if isinstance(run.get("generator"), dict) else {}
    if generator.get("name", "comfyui") != "comfyui":
        problems.append(f"generator.name must be comfyui, not {generator['name']!r}")
    if problem := comfyui_endpoint_problem(generator.get("endpoint"), where="generator.endpoint"):
        problems.append(problem)
    executor = run.get("executor") if isinstance(run.get("executor"), dict) else {}
    if executor.get("name", "local") not in ("local", "runpod"):
        problems.append(f"executor.name must be local or runpod, not {executor['name']!r}")
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    if "gpu" in compute:
        problem = gpu_choice_problem(compute["gpu"], runpod=executor.get("name") == "runpod")
        if problem:
            problems.append(problem)
    render = run.get("render") if isinstance(run.get("render"), dict) else {}
    timeout = render.get("timeout_sec", DEFAULT_RENDER_TIMEOUT_SEC)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        problems.append(f"render.timeout_sec must be a whole number of seconds of at least 1, not {timeout!r}")
    workflow = (run.get("inputs") or {}).get("workflow") if isinstance(run.get("inputs"), dict) else None
    if isinstance(workflow, dict) and workflow.get("digest") is not None and not isinstance(workflow["digest"], str):
        problems.append(f"inputs.workflow.digest must be a string or null, not {workflow['digest']!r}")
    if "output_dir" in render and not plain_relative_folder(render["output_dir"]):
        problems.append(f"render.output_dir must be a folder inside the run, such as samples/images (no leading or trailing slash, . or ..), not {render['output_dir']!r}")
    if problems:
        raise ValueError("render run.yaml: " + "; ".join(problems))


def _render_lora_strength(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -10 <= float(value) <= 10:
        raise ValueError("render.lora_strength must be a number, for example 0.8 or 1.0")
    return float(value)


def compile_render(workspace: Path, run_dir: Path) -> None:
    run = load_yaml_mapping(run_dir / "run.yaml")
    workspace_config = load_optional_yaml(workspace / "workspace.yaml")
    if workspace_config:
        validate_workspace_config(workspace_config)
    _validate_render_run_fields(run)
    inputs = run.get("inputs", {})
    if "promptset" in inputs:
        # Runs compiled with one stay readable; a new one authors inputs.cases.
        raise ValueError("render inputs.promptset is replaced by inputs.cases; write one row per case in a cases.jsonl")
    descriptor = inputs.get("cases")
    if not isinstance(descriptor, dict):
        raise ValueError("render requires inputs.cases, a mapping with path and digest")
    unknown_descriptor = sorted(str(key) for key in descriptor if key not in {"path", "digest"})
    if unknown_descriptor:
        raise ValueError(f"render inputs.cases has unsupported keys: {', '.join(unknown_descriptor)}")
    if not isinstance(descriptor.get("path"), str) or not descriptor["path"].strip():
        raise ValueError("render inputs.cases.path must be a non-empty path")
    if descriptor.get("digest") is not None and not isinstance(descriptor.get("digest"), str):
        raise ValueError("render inputs.cases.digest must be a string or null")
    train_run = _validate_train_run_reference(workspace, inputs.get("train_run"))
    workflow_path = workspace / inputs.get("workflow", {}).get("path", "")
    source_path = workspace / descriptor["path"]
    if not workflow_path.is_file() or not source_path.is_file():
        raise ValueError("render inputs.workflow.path and inputs.cases.path must exist")
    for name, path in (("workflow", workflow_path), ("cases", source_path)):
        # An authored digest asserts the file; null asserts nothing and compile records the file's.
        authored, actual = inputs[name].get("digest"), digest(path)
        if authored is not None and authored != actual:
            raise ValueError(
                f"render inputs.{name}.digest {authored} does not match {inputs[name]['path']} ({actual}); "
                "set it to null to render the file as it is now, or restore the file it names"
            )
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"workflow is not valid JSON: {exc}") from exc
    sidecar = _workflow_sidecar(workflow_path)
    lora_insert = _lora_insert_from_sidecar(sidecar) if isinstance(sidecar, dict) else None
    render_settings = run.get("render") if isinstance(run.get("render"), dict) else {}
    lora_strength = _render_lora_strength(render_settings.get("lora_strength"))
    bound_lora = isinstance((run.get("workflow_patches") or {}).get("lora"), dict) if isinstance(run.get("workflow_patches"), dict) else False
    if lora_strength is not None and lora_insert is None and not bound_lora:
        raise ValueError(
            "render.lora_strength sets the strength of the trained LoRA, and this workflow neither inserts it from a sidecar "
            "lora_insert nor binds a LoRA node in workflow_patches.lora"
        )
    if lora_strength is not None and lora_insert is not None:
        lora_insert["strength_model"] = lora_strength
        if "strength_clip" in lora_insert:
            lora_insert["strength_clip"] = lora_strength
    workflow_fixed = normalized_workflow_fixed(render_settings.get("workflow_fixed"))
    patches = run.get("workflow_patches")
    if patches is None:
        patches = {}
    if not isinstance(patches, dict):
        raise ValueError("workflow_patches must be a mapping of name to {node, field}")
    validate_patch_bindings(workflow, patches)
    shared_checkpoint = _validated_checkpoint(inputs.get("checkpoint"), context="render inputs.checkpoint", require_id=False)
    cases = authored_cases(source_path)
    if shared_checkpoint is not None and any(case.get("checkpoint") for case in cases):
        raise ValueError("render inputs.checkpoint cannot be combined with an explicit case checkpoint")
    if shared_checkpoint is not None:
        for case in cases:
            case["checkpoint"] = deepcopy(shared_checkpoint)
    has_checkpoint_consumer = bool(lora_insert) or any(name in patches for name in ASSET_PATCH_NAMES)
    requires_checkpoint = any(name in patches for name in ASSET_PATCH_NAMES)
    _reconcile_case_values(cases, patches, default_seed=render_settings.get("default_seed"), workflow_fixed=workflow_fixed, has_checkpoint_consumer=has_checkpoint_consumer, requires_checkpoint=requires_checkpoint)
    _freeze_case_checkpoints(workspace, cases)
    is_runpod = render_executor(run) == "runpod"
    _refuse_runpod_model_patch(render_executor(run), patches)
    if is_runpod and any(case.get("checkpoint") for case in cases) and not lora_insert and "lora" not in patches:
        raise ValueError(
            "RunPod case checkpoints currently require a lora workflow binding or sidecar lora_insert; "
            "dynamic checkpoint and model_patch staging are local-executor features"
        )
    frozen = deepcopy(run)
    frozen["workflow_patches"] = deepcopy(patches)
    if bound_lora:
        if bound_lora_strength_inputs(workflow, patches):
            if not isinstance(frozen.get("render"), dict):
                frozen["render"] = {}
            frozen["render"]["lora_strength"] = DEFAULT_LORA_STRENGTH if lora_strength is None else lora_strength
        elif lora_strength is not None and lora_insert is None:
            raise ValueError(
                "render.lora_strength cannot reach the LoRA node bound by workflow_patches.lora: its strength inputs are "
                "linked to other nodes or bound per case; set the strength where those inputs come from"
            )
    frozen.setdefault("inputs", {})["train_run"] = train_run
    if lora_insert:
        insert_lora_loader(workflow, lora_insert, "placeholder.safetensors")
        frozen["lora_insert"] = lora_insert
    frozen.setdefault("inputs", {}).setdefault("workflow", {})["digest"] = digest(workflow_path)
    frozen["inputs"]["cases"]["digest"] = digest(source_path)
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
        # A local render stages every checkpoint; each plan refuses one it cannot stage.
        for case in cases:
            checkpoint = case.get("checkpoint") if isinstance(case.get("checkpoint"), dict) else {}
            _lora_stage_plan(workspace, run_dir, frozen, checkpoint)
            _model_patch_stage_plan(workspace, run_dir, frozen, checkpoint)
        endpoint = run.get("generator", {}).get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            try:
                object_info = _fetch_endpoint_object_info(endpoint)
            except Exception as exc:
                print(f"warning: ComfyUI endpoint identity is unavailable at compile time: {endpoint}: {exc}", flush=True)
            else:
                visible, missing = visible_model_refs(workflow, object_info, ignored_inputs=_dynamically_patched_model_inputs(frozen))
                _refuse_missing_models(missing, endpoint)
                frozen["comfyui_endpoint_identity"] = endpoint_fingerprint(object_info)
                frozen["comfyui_required_models"] = {"visible": visible, "missing": missing}
    # Image freezing mutates resolved/ and the in-memory cases. Keep it
    # after every validation and digest step so a rejected compile leaves no
    # unreferenced image copies behind.
    resolved = run_dir / "resolved"
    resolved.mkdir(exist_ok=True)
    frozen_images = _freeze_case_images(run_dir, source_path, cases, patches)
    if frozen_images:
        frozen["promptset_images"] = frozen_images
    frozen["_kura"] = {"frozen_at": now(), "artifact": "manifest.lock"}
    indexed_cases = _indexed_cases(cases)
    dump_yaml(resolved / "manifest.lock.yaml", frozen)
    atomic_write_json(resolved / "workflow_used.json", workflow)
    if "comfyui_models" in frozen:
        atomic_write_json(resolved / "comfyui_models.json", frozen["comfyui_models"])
    if "comfyui_model_registry" in frozen:
        atomic_write_json(resolved / "comfyui_model_registry.json", frozen["comfyui_model_registry"])
    (resolved / "cases.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in indexed_cases), encoding="utf-8")
    env = {**kura_provenance(), "generator": "comfyui", "endpoint": run.get("generator", {}).get("endpoint"), "generated_at": now()}
    if is_runpod:
        # The image a RunPod render launches, frozen as a training compile freezes its own (`launch_image` reads it).
        image = effective_image(workspace_config or {}, "comfyui")
        env.update({"selected_image": image["reference"], "image_origin": image["origin"]})
    dump_yaml(resolved / "env.lock", env)
    status(run_dir, state="compiled", last_step=0, total_steps=len(indexed_cases), current_case_id=None)


def launch_render(
    workspace: Path,
    run_dir: Path,
    dry_run: bool = False,
    *,
    endpoint_override: str | None = None,
    lora_name_override: str | None = None,
    lora_name_overrides: dict[str, str] | None = None,
    image_name_overrides: dict[str, str] | None = None,
    executor_name: str | None = None,
    manage_lora_stage: bool = True,
    controlled_by: dict[str, Any] | None = None,
    check_only: bool = False,
) -> int:
    manifest_path = run_dir / "resolved" / "manifest.lock.yaml"
    workflow_used_path = run_dir / "resolved" / "workflow_used.json"
    if not manifest_path.is_file() or not workflow_used_path.is_file():
        raise ValueError("render is not compiled; run kura render compile first")
    frozen = load_yaml_mapping(manifest_path)
    resolved_executor = executor_name or render_executor(frozen)
    if frozen.get("generator", {}).get("name") != "comfyui" or resolved_executor not in ("local", "runpod"):
        raise ValueError("render runs require generator.name=comfyui and executor.name=local or runpod")
    _refuse_runpod_model_patch(resolved_executor, frozen.get("workflow_patches", {}))
    current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    allowed_states = {"compiled"} if resolved_executor == "local" else {"compiled", "running"}
    if current_status.get("state") not in allowed_states:
        raise ValueError("render must be compiled and has already been launched or finalized")
    inputs = frozen.get("inputs", {})
    train_run = inputs.get("train_run")
    workflow_path = workspace / inputs.get("workflow", {}).get("path", "")
    cases_path = run_dir / "resolved" / "cases.jsonl"
    workflow_fixed = normalized_workflow_fixed(frozen.get("render", {}).get("workflow_fixed"))
    if cases_path.is_file():
        cases = load_resolved_cases(cases_path)
    else:
        # Released compiled renders predate resolved/cases.jsonl. Adapt only their
        # already-frozen promptset; never return to the mutable authored source.
        promptset_used = run_dir / "resolved" / "promptset_used.jsonl"
        if not promptset_used.is_file():
            raise ValueError("render cases are not frozen; run kura render compile first")
        legacy_items = promptset(promptset_used, require_prompt="prompt" not in workflow_fixed)
        raw_checkpoint = inputs.get("checkpoint")
        legacy_checkpoint = raw_checkpoint if isinstance(raw_checkpoint, dict) and raw_checkpoint.get("path") else None
        legacy_cases = _legacy_render_cases(legacy_items, legacy_checkpoint, patches=frozen.get("workflow_patches", {}), default_seed=frozen.get("render", {}).get("default_seed"), workflow_fixed=workflow_fixed)
        cases = _indexed_cases(legacy_cases)
    endpoint = endpoint_override or frozen["generator"].get("endpoint")
    # A run compiled before such endpoints were refused cannot reach ComfyUI either.
    if problem := comfyui_endpoint_problem(endpoint, where="generator.endpoint"):
        raise ValueError(f"{problem}; compile a new run from a corrected run.yaml")
    image_names = image_patch_names(frozen.get("workflow_patches", {}))
    if resolved_executor == "runpod":
        expected_images: set[str] = set()
        for case in cases:
            values = case.get("values") if isinstance(case.get("values"), dict) else {}
            for name in image_names:
                value = values.get(name)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"runpod remote image mapping cannot resolve case {case.get('id')!r} "
                        f"workflow_patches.{name}"
                    )
                expected_images.add(value)
        provided_images = set(image_name_overrides or {})
        if expected_images != provided_images:
            missing = sorted(expected_images - provided_images)
            unexpected = sorted(provided_images - expected_images)
            raise ValueError(
                "runpod remote image mapping must exactly cover every frozen image; "
                f"missing={missing} unexpected={unexpected}"
            )
        image_stages = []
        for frozen_path, image_name in sorted((image_name_overrides or {}).items()):
            image_path = Path(image_name)
            if (
                not image_name
                or image_path.is_absolute()
                or ".." in image_path.parts
                or "\\" in image_name
            ):
                raise ValueError(f"runpod remote image mapping has an unsafe ComfyUI input name: {image_name!r}")
            image_stages.append({
                "kind": "image",
                "frozen": frozen_path,
                "image_name": image_name,
                "remote": True,
            })
    else:
        if image_name_overrides is not None:
            raise ValueError("remote image mapping is only valid for the runpod executor")
        image_stages = _image_stage_plans(workspace, run_dir, frozen)
    workflow = json.loads(workflow_used_path.read_text(encoding="utf-8"))
    runtime_cases: list[dict[str, Any]] = []
    lora_stages: dict[str, dict[str, Any]] = {}
    model_patch_stages: dict[str, dict[str, Any]] = {}
    for case in cases:
        checkpoint = case.get("checkpoint") if isinstance(case.get("checkpoint"), dict) else {}
        lora_stage = _lora_stage_plan(workspace, run_dir, frozen, checkpoint, endpoint) if manage_lora_stage else None
        model_patch_stage = _model_patch_stage_plan(workspace, run_dir, frozen, checkpoint, endpoint) if manage_lora_stage else None
        if lora_stage:
            lora_stage = lora_stages.setdefault(str(lora_stage["target"]), lora_stage)
        if model_patch_stage:
            model_patch_stage = model_patch_stages.setdefault(str(model_patch_stage["target"]), model_patch_stage)
        checkpoint_id = checkpoint.get("id")
        case_override = lora_name_overrides.get(checkpoint_id) if lora_name_overrides and isinstance(checkpoint_id, str) else None
        # A RunPod render names its remotely staged files; a local one names what it staged here.
        lora_name = case_override or lora_name_override or (lora_stage["lora_name"] if lora_stage else None)
        model_patch_name = model_patch_stage["model_patch_name"] if model_patch_stage else None
        runtime_cases.append({
            "case": case,
            "checkpoint": checkpoint,
            "lora_stage": lora_stage,
            "model_patch_stage": model_patch_stage,
            "lora_name": lora_name,
            "model_patch_name": model_patch_name,
            "checkpoint_application": checkpoint_application(
                frozen, workflow, lora_name=lora_name, checkpoint_record=checkpoint,
            ),
        })
    case_queue = [
        {
            "id": item["case"]["id"],
            "index": item["case"]["index"],
            "values": item["case"]["values"],
            "checkpoint": item["case"].get("checkpoint"),
            "meta": item["case"].get("meta"),
            "checkpoint_application": item["checkpoint_application"],
            "comfyui_lora_name": item["lora_name"],
            "comfyui_model_patch_name": item["model_patch_name"],
        }
        for item in runtime_cases
    ]
    source = inputs.get("cases") if isinstance(inputs.get("cases"), dict) else inputs.get("promptset", {})
    source_digest = source.get("digest") if isinstance(source, dict) else None
    legacy_digest_details = {"promptset_digest": source_digest} if isinstance(inputs.get("promptset"), dict) else {}
    details = {"train_run": train_run, "endpoint": endpoint, "workflow_path": str(workflow_path), "workflow_digest": inputs.get("workflow", {}).get("digest"), "cases_digest": source_digest, "case_count": len(cases), "case_queue": case_queue, "image_stages": image_stages, "executor": resolved_executor, "output_dir": frozen.get("render", {}).get("output_dir"), "patch_mapping": frozen.get("workflow_patches", {}), "resolved_paths": ["resolved/manifest.lock.yaml", "resolved/workflow_used.json", "resolved/cases.jsonl", "resolved/env.lock"]}
    details.update(_runtime_checkpoint_provenance(runtime_cases))
    if isinstance(inputs.get("promptset"), dict):
        details["promptset_digest"] = source_digest
    if len(runtime_cases) == 1:
        only = runtime_cases[0]
        details.update({
            "checkpoint": only["checkpoint"],
            "checkpoint_application": only["checkpoint_application"],
            "comfyui_lora_name": only["lora_name"],
            "comfyui_model_patch_name": only["model_patch_name"],
        })
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
    output_dir = render_output_dir(run_dir, frozen)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_log = run_dir / "samples" / "images.jsonl"
    images_log.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("render generator endpoint is empty")
    client = ComfyUIClient(endpoint, render_timeout_sec(frozen))
    # The user's ComfyUI is checked before the run is marked running, so a stopped
    # or different ComfyUI leaves the run compiled and launchable again.
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
                f"expected={expected_identity.get('sha256')} observed={observed_identity.get('sha256')} endpoint={endpoint}"
            )
        _refuse_missing_models(visible_model_refs(workflow, object_info, ignored_inputs=_dynamically_patched_model_inputs(frozen))[1], endpoint)
    if check_only:
        return 0
    stdout_log = run_dir / "logs" / "stdout.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(f"render endpoint: {endpoint}\n", encoding="utf-8")
    from kura.executors.common import StaleRunnerEpoch

    # The event first: status says running because of it.
    started_at = now()
    event(run_dir, {"event": "render_started", "timestamp": started_at, "train_run": train_run, "generator": "comfyui", "executor": resolved_executor, "endpoint": endpoint, "case_count": len(cases), "image_stages": image_stages})
    status(run_dir, state="running", started=started_at, ended=None, exit_code=None, error=None, last_step=0, total_steps=len(cases), current_case_id=None)
    active_runtime_case: dict[str, Any] | None = None
    queued_prompt_id: str | None = None
    generated = 0
    completed_cases = 0
    try:
        for lora_stage in lora_stages.values():
            _materialize_visible(lora_stage, lambda plan: _ensure_lora_stage_visible(client, endpoint, plan))
        for model_patch_stage in model_patch_stages.values():
            _materialize_visible(model_patch_stage, lambda plan: _ensure_model_patch_stage_visible(client, endpoint, plan))
        if resolved_executor == "local":
            for plan in image_stages:
                _materialize_stage(plan)
        for runtime_case in runtime_cases:
            active_runtime_case = runtime_case
            case = runtime_case["case"]
            checkpoint = runtime_case["checkpoint"]
            lora_name = runtime_case["lora_name"]
            model_patch_name = runtime_case["model_patch_name"]
            values = case["values"]
            patches = frozen.get("workflow_patches", {})
            image_values = _image_values_for_item({"id": case["id"], **values}, patches, image_stages)
            patched = patch_workflow(workflow, patches, prompt=values.get("prompt", ""), negative_prompt=values.get("negative_prompt", ""), seed=values.get("seed"), checkpoint=lora_name, model_patch=model_patch_name, item={"id": case["id"], **values}, image_values=image_values)
            patched = insert_lora_loader(patched, frozen.get("lora_insert"), lora_name)
            if lora_name:
                render_settings = frozen.get("render") if isinstance(frozen.get("render"), dict) else {}
                patched = apply_bound_lora_strength(patched, patches, float(render_settings.get("lora_strength", DEFAULT_LORA_STRENGTH)))
            case_application = checkpoint_application(frozen, patched, lora_name=lora_name, checkpoint_record=checkpoint)
            applied_values = {name: image_values.get(name, values.get(name)) for name in patches if name not in ASSET_PATCH_NAMES}
            for name in patches:
                if name in ("lora", "checkpoint"):
                    applied_values[name] = lora_name
                elif name == "model_patch":
                    applied_values[name] = model_patch_name
            from kura.executors.common import check_stop

            check_stop()
            event(run_dir, {"event": "render_case_started", "timestamp": now(), "case_id": case["id"], "index": case["index"], "total": len(cases)})
            status(run_dir, current_case_id=case["id"])
            prompt_id = client.queue(patched)
            queued_prompt_id = prompt_id
            with stdout_log.open("a", encoding="utf-8") as handle:
                if isinstance(inputs.get("promptset"), dict):
                    handle.write(f"queued {case.get('source_id', case['id'])} seed={values.get('seed')} prompt_id={prompt_id}\n")
                else:
                    handle.write(f"queued case={case['id']} index={case['index']}/{len(cases)} prompt_id={prompt_id}\n")
            images = client.wait(prompt_id)
            queued_prompt_id = None
            if not images:
                raise RuntimeError(f"ComfyUI completed without returning any images (case {case['id']!r})")
            for image_index, image in enumerate(images):
                suffix = Path(image.get("filename", "image.png")).suffix or ".png"
                relative = f"samples/images/{case['id']}_{image_index}{suffix}"
                image_path = run_dir / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(client.download(image))
                legacy_prompt_id = case.get("source_id", case["id"])
                record = {"file": relative, "train_run": train_run, "case_id": case["id"], "case_index": case["index"], "case": case, "applied_values": applied_values, "prompt_id": legacy_prompt_id, "prompt": values.get("prompt") if "prompt" not in workflow_fixed else None, "negative_prompt": values.get("negative_prompt") if "negative_prompt" not in workflow_fixed else None, "seed": values.get("seed") if "seed" not in workflow_fixed else None, "checkpoint_id": checkpoint.get("id"), "checkpoint_path": checkpoint.get("path"), "checkpoint_hash": checkpoint.get("hash"), "checkpoint_application": case_application, "comfyui_lora_name": lora_name, "workflow_digest": inputs.get("workflow", {}).get("digest"), "cases_digest": source_digest, "comfyui_prompt_id": prompt_id, "patch_inputs": {name: values[name] for name in patches if name in values and name not in ("prompt", "negative_prompt", "seed")}, "workflow_fixed": list(workflow_fixed), "created": now()}
                with images_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                event(run_dir, {"event": "image_generated", "timestamp": now(), "case_id": case["id"], "case_index": case["index"], "file": relative})
                generated += 1
            completed_cases += 1
            event(run_dir, {"event": "render_case_completed", "timestamp": now(), "case_id": case["id"], "index": case["index"], "total": len(cases), "image_count": len(images)})
            status(run_dir, last_step=completed_cases, total_steps=len(cases), current_case_id=None)
        if generated == 0:
            raise RuntimeError("ComfyUI completed without returning any images")
        write_realization(run_dir, status_changes={"state": "completed", "ended": now(), "exit_code": 0, "last_step": len(cases), "total_steps": len(cases), "current_case_id": None}, **({"controlled_by": controlled_by} if controlled_by else {}), train_run=train_run, executor=resolved_executor, generator="comfyui", state="completed", workflow_fixed=list(workflow_fixed), endpoint=endpoint, workflow_digest=inputs.get("workflow", {}).get("digest"), cases_digest=source_digest, **legacy_digest_details, **_runtime_checkpoint_provenance(runtime_cases), case_count=len(cases), completed_case_count=completed_cases, image_count=generated)
        event(run_dir, {"event": "render_completed", "timestamp": now(), "count": generated})
        return 0
    except KeyboardInterrupt:
        # Ctrl-C, or a stop request reaching a runner follower: images already written stay,
        # Kura's own prompt is withdrawn, and the render is not continued later.
        if queued_prompt_id is not None:
            try:
                client.cancel(queued_prompt_id)
            except Exception as cancel_exc:
                _append_runtime_warning(stdout_log, f"could not withdraw prompt {queued_prompt_id}: {type(cancel_exc).__name__}: {cancel_exc}")
        interrupted_at = now()
        try:
            write_realization(run_dir, status_changes={"state": "interrupted", "ended": interrupted_at, "exit_code": None, "last_step": completed_cases, "total_steps": len(cases), "current_case_id": None}, **({"controlled_by": controlled_by} if controlled_by else {}), train_run=train_run, executor=resolved_executor, generator="comfyui", state="interrupted", workflow_fixed=list(workflow_fixed), endpoint=endpoint, case_count=len(cases), completed_case_count=completed_cases, generated_image_count=generated)
            event(run_dir, {"event": "render_interrupted", "timestamp": interrupted_at, "completed_case_count": completed_cases, "image_count": generated})
        except StaleRunnerEpoch:
            raise
        except Exception as record_exc:
            _append_runtime_warning(stdout_log, f"failed to record the interruption: {type(record_exc).__name__}: {record_exc}")
        raise
    except StaleRunnerEpoch:
        # A newer runner controls this run now; this follower records nothing more.
        raise
    except Exception as exc:
        failed_case_id = active_runtime_case["case"]["id"] if active_runtime_case is not None else None
        failed_at = now()
        try:
            with stdout_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
        except OSError:
            pass
        failed_status = {"state": "failed", "ended": failed_at, "exit_code": 1, "last_step": completed_cases, "total_steps": len(cases), "current_case_id": failed_case_id}
        try:
            # The record first; status then points at it.
            failed_status["last_realization"] = _write_realization_file(run_dir, timestamp=failed_at, **({"controlled_by": controlled_by} if controlled_by else {}), train_run=train_run, executor=resolved_executor, generator="comfyui", state="failed", workflow_fixed=list(workflow_fixed), endpoint=endpoint, workflow_digest=inputs.get("workflow", {}).get("digest"), cases_digest=source_digest, **legacy_digest_details, **_runtime_checkpoint_provenance(runtime_cases), case_count=len(cases), completed_case_count=completed_cases, failed_case_id=failed_case_id, generated_image_count=generated, error=str(exc))
        except Exception as realization_exc:
            _append_runtime_warning(
                stdout_log,
                "failed to persist render failure realization: "
                f"{type(realization_exc).__name__}: {realization_exc}",
            )
        try:
            status(run_dir, **failed_status)
        except StaleRunnerEpoch:
            raise
        except Exception as status_exc:
            # The failure is already recorded; status keeps what it had, Pod id included, so
            # the Pod is still deleted. The run then ends as interrupted (the Pod stop, or the
            # runner settling a follower that returned), not as failed.
            _append_runtime_warning(
                stdout_log,
                f"failed to persist render failure status: {type(status_exc).__name__}: {status_exc}",
            )
        try:
            event(run_dir, {"event": "render_failed", "timestamp": failed_at, "error": str(exc)})
        except OSError as event_exc:
            _append_runtime_warning(
                stdout_log,
                "failed to persist render failure event: "
                f"{type(event_exc).__name__}: {event_exc}",
            )
        return 1
    finally:
        for plan in lora_stages.values():
            _cleanup_stage(plan)
        for plan in model_patch_stages.values():
            _cleanup_stage(plan)
        if resolved_executor == "local":
            for plan in image_stages:
                _cleanup_stage(plan)
