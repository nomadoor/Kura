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
from kura.comfyui_models import merged_registry, resolve_model_specs
from kura.fsio import atomic_write_json
from kura.workspace import dump_yaml


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


def patch_workflow(workflow: dict[str, Any], patches: dict[str, Any], *, prompt: str, negative_prompt: str, seed: int, checkpoint: str) -> dict[str, Any]:
    patched = deepcopy(workflow)
    values = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "lora": checkpoint, "checkpoint": checkpoint}
    for name, value in values.items():
        patch = patches.get(name)
        if patch is None:
            continue
        if not isinstance(patch, dict) or not isinstance(patch.get("node"), str) or not isinstance(patch.get("field"), str):
            raise ValueError(f"workflow_patches.{name} requires node and field")
        _set_path(patched, patch["node"], patch["field"], value)
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
        "source": str(source),
        "target": str(target),
        "lora_name": f"{stage_subdir}/{target.name}",
        "mode": mode,
        "cleanup": cleanup,
        "created": False,
    }


def _freeze_comfyui_config(comfyui: Any) -> dict[str, Any]:
    if not isinstance(comfyui, dict):
        return {}
    allowed = ("lora_dir", "lora_stage_subdir", "lora_stage_mode", "lora_stage_cleanup", "model_registry", "runpod")
    return {key: deepcopy(comfyui[key]) for key in allowed if key in comfyui}


def _materialize_lora_stage(plan: dict[str, Any]) -> None:
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
        raise ValueError(f"ComfyUI LoRA stage target already exists with different content: {target}")
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


def _cleanup_lora_stage(plan: dict[str, Any] | None) -> None:
    if not plan or plan.get("cleanup") != "remove_after_render" or not plan.get("created"):
        return
    target = Path(str(plan.get("target", "")))
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
    except OSError:
        pass


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
            f"Run `uv run kura doctor comfyui --endpoint {endpoint}` to check the endpoint."
        ) from exc
    raise ValueError(
        "ComfyUI LoRA stage is not visible from the configured endpoint. "
        f"endpoint={endpoint}; lora_name={plan.get('lora_name')}; lora_dir={Path(str(plan.get('target'))).parent.parent}. "
        f"Run `uv run kura doctor comfyui --endpoint {endpoint} --probe-stage` to verify staging, "
        "then set comfyui.lora_dir to a LoRA directory used by that ComfyUI instance and recompile the render run."
    )


def promptset(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"promptset:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
            raise ValueError(f"promptset:{line_number}: id and prompt are required")
        prompts.append(item)
    return prompts


class ComfyUIClient:
    def __init__(self, endpoint: str, timeout: int) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(f"{self.endpoint}{path}", data=data, headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

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
    promptset(promptset_path)
    patch_workflow(
        workflow, run.get("workflow_patches", {}), prompt="", negative_prompt="", seed=0,
        checkpoint=inputs.get("checkpoint", {}).get("path", ""),
    )
    resolved = run_dir / "resolved"
    resolved.mkdir(exist_ok=True)
    frozen = deepcopy(run)
    if lora_insert:
        insert_lora_loader(workflow, lora_insert, "placeholder.safetensors")
        frozen["lora_insert"] = lora_insert
    frozen.setdefault("inputs", {}).setdefault("workflow", {})["digest"] = digest(workflow_path)
    frozen["inputs"].setdefault("promptset", {})["digest"] = digest(promptset_path)
    comfyui = _freeze_comfyui_config(workspace_config.get("comfyui"))
    if comfyui:
        frozen["comfyui"] = comfyui
    executor = run.get("executor") if isinstance(run.get("executor"), dict) else {}
    if executor.get("name") == "runpod":
        sidecar_models = sidecar.get("models") if isinstance(sidecar, dict) else {}
        workspace_models = comfyui.get("model_registry") if isinstance(comfyui, dict) else {}
        registry = merged_registry(sidecar_models, workspace_models)
        specs, unknown = resolve_model_specs(workflow, registry)
        if unknown:
            labels = ", ".join(f"{item['class_type']}.{item['input']}={item['name']}" for item in unknown)
            raise ValueError("runpod ComfyUI render has unknown model loader entries; add comfyui.model_registry mappings for: " + labels)
        frozen["comfyui_model_registry"] = registry
        frozen["comfyui_models"] = specs
    checkpoint_path = inputs.get("checkpoint", {}).get("path")
    if checkpoint_path:
        candidate = workspace / checkpoint_path
        if candidate.is_file():
            frozen["inputs"].setdefault("checkpoint", {})["hash"] = digest(candidate)
        elif not inputs.get("checkpoint", {}).get("hash"):
            print("warning: checkpoint hash is unavailable", flush=True)
    frozen["_kura"] = {"frozen_at": now(), "artifact": "manifest.lock"}
    dump_yaml(resolved / "manifest.lock.yaml", frozen)
    atomic_write_json(resolved / "workflow_used.json", workflow)
    if "comfyui_models" in frozen:
        atomic_write_json(resolved / "comfyui_models.json", frozen["comfyui_models"])
    if "comfyui_model_registry" in frozen:
        atomic_write_json(resolved / "comfyui_model_registry.json", frozen["comfyui_model_registry"])
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
    current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    allowed_states = {"compiled"} if resolved_executor == "local" else {"compiled", "running"}
    if current_status.get("state") not in allowed_states:
        raise ValueError("render must be compiled and has already been launched or finalized")
    inputs = frozen.get("inputs", {})
    workflow_path = workspace / inputs.get("workflow", {}).get("path", "")
    promptset_path = workspace / inputs.get("promptset", {}).get("path", "")
    promptset_used_path = run_dir / "resolved" / "promptset_used.jsonl"
    if not promptset_used_path.is_file():
        raise ValueError("render promptset is not frozen; run kura render compile first")
    prompts = promptset(promptset_used_path)
    checkpoint = inputs.get("checkpoint", {})
    default_seed = frozen.get("render", {}).get("default_seed")
    pairs = [(item, seed) for item in prompts for seed in item.get("seeds", [default_seed]) if seed is not None]
    if not pairs:
        raise ValueError("promptset has no seeds and render.default_seed is not set")
    endpoint = endpoint_override or frozen["generator"].get("endpoint")
    lora_stage = _lora_stage_plan(workspace, run_dir, frozen, checkpoint) if manage_lora_stage else None
    lora_name = lora_name_override or (lora_stage["lora_name"] if lora_stage else checkpoint.get("path", ""))
    details = {"endpoint": endpoint, "workflow_path": str(workflow_path), "workflow_digest": inputs.get("workflow", {}).get("digest"), "promptset_path": str(promptset_path), "promptset_digest": inputs.get("promptset", {}).get("digest"), "prompt_count": len(prompts), "total_image_count": len(pairs), "checkpoint": checkpoint, "comfyui_lora_name": lora_name, "lora_stage": lora_stage, "executor": resolved_executor, "output_dir": frozen.get("render", {}).get("output_dir"), "patch_mapping": frozen.get("workflow_patches", {}), "resolved_paths": ["resolved/manifest.lock.yaml", "resolved/workflow_used.json", "resolved/promptset_used.jsonl", "resolved/env.lock"]}
    if dry_run:
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return 0
    workflow = json.loads(workflow_used_path.read_text(encoding="utf-8"))
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
        if lora_stage:
            _materialize_lora_stage(lora_stage)
            _ensure_lora_stage_visible(client, endpoint, lora_stage)
        event(run_dir, {"event": "render_started", "timestamp": now(), "generator": "comfyui", "executor": resolved_executor, "endpoint": endpoint, "lora_stage": lora_stage})
        generated = 0
        for item, seed in pairs:
            patched = patch_workflow(workflow, frozen.get("workflow_patches", {}), prompt=item["prompt"], negative_prompt=item.get("negative_prompt", ""), seed=seed, checkpoint=lora_name)
            patched = insert_lora_loader(patched, frozen.get("lora_insert"), lora_name)
            prompt_id = client.queue(patched)
            with stdout_log.open("a", encoding="utf-8") as handle:
                handle.write(f"queued {item['id']} seed={seed} prompt_id={prompt_id}\n")
            for index, image in enumerate(client.wait(prompt_id)):
                suffix = Path(image.get("filename", "image.png")).suffix or ".png"
                relative = f"samples/images/{item['id']}_seed{seed}_{index}{suffix}"
                image_path = run_dir / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(client.download(image))
                record = {"file": relative, "prompt_id": item["id"], "prompt": item["prompt"], "negative_prompt": item.get("negative_prompt", ""), "seed": seed, "checkpoint_path": checkpoint.get("path"), "checkpoint_hash": checkpoint.get("hash"), "comfyui_lora_name": lora_name, "workflow_digest": inputs.get("workflow", {}).get("digest"), "promptset_digest": inputs.get("promptset", {}).get("digest"), "comfyui_prompt_id": prompt_id, "created": now()}
                with images_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                event(run_dir, {"event": "image_generated", "timestamp": now(), "prompt_id": item["id"], "seed": seed, "file": relative})
                generated += 1
        if generated == 0:
            raise RuntimeError("ComfyUI completed without returning any images")
        status(run_dir, state="completed", ended=now(), exit_code=0)
        write_realization(run_dir, executor=resolved_executor, generator="comfyui", state="completed", endpoint=endpoint, workflow_digest=inputs.get("workflow", {}).get("digest"), promptset_digest=inputs.get("promptset", {}).get("digest"), checkpoint_hash=checkpoint.get("hash"), comfyui_lora_name=lora_name, lora_stage=lora_stage, image_count=generated)
        event(run_dir, {"event": "render_completed", "timestamp": now(), "count": generated})
        return 0
    except Exception as exc:
        stdout_log = run_dir / "logs" / "stdout.log"
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        with stdout_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
        status(run_dir, state="failed", ended=now(), exit_code=1)
        write_realization(run_dir, executor=resolved_executor, generator="comfyui", state="failed", endpoint=endpoint, workflow_digest=inputs.get("workflow", {}).get("digest"), promptset_digest=inputs.get("promptset", {}).get("digest"), checkpoint_hash=checkpoint.get("hash"), comfyui_lora_name=lora_name, lora_stage=lora_stage, error=str(exc))
        event(run_dir, {"event": "render_failed", "timestamp": now(), "error": str(exc)})
        return 1
    finally:
        _cleanup_lora_stage(lora_stage)
