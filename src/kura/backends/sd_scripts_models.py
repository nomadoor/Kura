"""sd-scripts explicit model roles, downloads, and provenance projections."""

from __future__ import annotations

import json
from typing import Any

from kura.container_scripts import script_source
from kura.provenance import artifact_pinning
from kura.run_envelope import backend_config


ROLE_CONTRACTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("sd15", "lora"): ("base",),
    ("sdxl", "lora"): ("base",),
    ("flux1", "lora"): ("dit", "clip_l", "t5xxl", "ae"),
    ("anima", "lora"): ("dit", "qwen3", "vae"),
    ("anima", "controlnet_lllite"): ("dit", "qwen3", "vae"),
}
OPTIONAL_ROLES = {"vae", "llm_adapter", "t5_tokenizer"}


def sd_scripts_native(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "sd-scripts")


def sd_scripts_architecture(run: dict[str, Any]) -> str:
    value = sd_scripts_native(run).get("architecture")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sd-scripts backend.config.architecture is required")
    normalized = value.lower().replace("-", "_").replace(".", "_")
    aliases = {
        "sd_1_5": "sd15", "stable_diffusion_1_5": "sd15", "stable_diffusion15": "sd15",
        "sdxl_1_0": "sdxl", "flux": "flux1", "flux_1": "flux1",
    }
    return aliases.get(normalized, normalized)


def sd_scripts_mode(run: dict[str, Any]) -> str:
    value = sd_scripts_native(run).get("mode", "lora")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sd-scripts backend.config.mode must be a string")
    normalized = value.lower().replace("-", "_")
    return {"controlnet_lllite": "controlnet_lllite", "lllite": "controlnet_lllite"}.get(normalized, normalized)


def _safe_filename(filename: str) -> str:
    if filename.startswith("/") or any(part in ("", ".", "..") for part in filename.split("/")):
        raise ValueError(f"invalid Hugging Face filename for sd-scripts: {filename}")
    return filename


def _component(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in value).strip(".-")
    if not clean:
        raise ValueError(f"invalid sd-scripts model cache component: {value}")
    return clean


def _cache_path(repo_id: str, role: str, filename: str) -> str:
    return f"/workspace/cache/models/sd-scripts/{_component(repo_id.replace('/', '--'))}/{_component(role)}/{filename}"


def explicit_model_paths(run: dict[str, Any]) -> dict[str, str]:
    values = sd_scripts_native(run).get("model_paths")
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("sd-scripts model_paths must be a mapping")
    return {key: value for key, value in values.items() if isinstance(key, str) and isinstance(value, str) and value}


def sd_scripts_model_download_specs(run: dict[str, Any], existing_paths: dict[str, str] | None = None) -> tuple[list[dict[str, str]], dict[str, str]]:
    downloads = sd_scripts_native(run).get("model_downloads")
    if downloads is None:
        return [], {}
    if not isinstance(downloads, dict):
        raise ValueError("sd-scripts model_downloads must be a mapping")
    existing_paths = existing_paths or {}
    specs: list[dict[str, str]] = []
    paths: dict[str, str] = {}
    for role, value in downloads.items():
        if role in existing_paths:
            continue
        if not isinstance(role, str) or not isinstance(value, dict):
            raise ValueError("sd-scripts model_downloads must map roles to download mappings")
        repo_id = value.get("repo_id") or value.get("repo")
        filenames_value = value.get("filenames")
        filenames = [item for item in filenames_value if isinstance(item, str) and item] if isinstance(filenames_value, list) else []
        filename = value.get("filename") or value.get("file") or (filenames[0] if filenames else None)
        if not isinstance(repo_id, str) or not repo_id or not isinstance(filename, str) or not filename:
            raise ValueError(f"sd-scripts model_downloads.{role} requires repo_id and filename")
        filename = _safe_filename(filename)
        filenames = [_safe_filename(item) for item in (filenames or [filename])]
        for item_filename in filenames:
            spec = {"key": role, "repo_id": repo_id, "filename": item_filename, "link_path": _cache_path(repo_id, role, item_filename)}
            for key in ("revision", "repo_type"):
                if isinstance(value.get(key), str) and value[key]:
                    spec[key] = value[key]
            specs.append(spec)
        paths[role] = _cache_path(repo_id, role, filename)
    return specs, paths


def sd_scripts_model_paths(run: dict[str, Any]) -> dict[str, str]:
    paths = explicit_model_paths(run)
    _, downloaded = sd_scripts_model_download_specs(run, paths)
    paths.update(downloaded)
    required = ROLE_CONTRACTS.get((sd_scripts_architecture(run), sd_scripts_mode(run)))
    if required is None:
        return paths
    missing = [role for role in required if not paths.get(role)]
    if missing:
        raise ValueError("sd-scripts model_paths/model_downloads missing required role(s): " + ", ".join(missing))
    return paths


def sd_scripts_download_commands(run: dict[str, Any], existing_paths: dict[str, str] | None = None) -> tuple[list[list[str]], dict[str, str]]:
    specs, paths = sd_scripts_model_download_specs(run, existing_paths)
    if not specs:
        return [], paths
    return [["python", "-c", script_source("hf_download.py"), json.dumps(specs, ensure_ascii=False)]], paths


def requirements_sd_scripts(run: dict[str, Any], download_estimate: dict[str, Any] | None = None, *, declared: bool = False) -> list[dict[str, Any]]:
    estimate = download_estimate or {}
    if declared:
        explicit = explicit_model_paths(run)
        specs, _ = sd_scripts_model_download_specs(run, explicit)
        estimate = {"items": [{"key": item["key"], "repo_id": item["repo_id"], "filename": item["filename"], "revision": item.get("revision"), "runtime_reference": item["link_path"], "size_status": "not-measured", "measurement_scope": "compile", "size_bytes": None, "cached": False} for item in specs]}
    requirements: list[dict[str, Any]] = []
    for item in estimate.get("items") if isinstance(estimate.get("items"), list) else []:
        if not isinstance(item, dict):
            continue
        identity = {"kind": "huggingface-file", "repo_id": item.get("repo_id"), "filename": item.get("filename")}
        if item.get("revision"):
            identity["revision"] = item["revision"]
        requirements.append({"role": item.get("key") or "model", "acquisition": "kura", "identity": identity, "runtime_reference": item.get("runtime_reference"), "expected_format": "backend-role-file", "measurement": {"scope": item.get("measurement_scope") or "controller", "status": item.get("size_status") or "unknown", "size_bytes": item.get("size_bytes"), "cached": bool(item.get("cached"))}, "pinning": artifact_pinning(identity, observable=True)})
    for role, path in sorted(explicit_model_paths(run).items()):
        identity = {"kind": "path", "path": path}
        requirements.append({"role": role, "acquisition": "local-path", "identity": identity, "runtime_reference": path, "expected_format": "backend-role-file", "measurement": {"scope": "compile", "status": "declared"}, "pinning": artifact_pinning(identity, observable=True)})
    return requirements


def sd_scripts_model_lock(run: dict[str, Any]) -> dict[str, Any]:
    paths = sd_scripts_model_paths(run)
    explicit = explicit_model_paths(run)
    downloads = sd_scripts_native(run).get("model_downloads") if isinstance(sd_scripts_native(run).get("model_downloads"), dict) else {}
    models: list[dict[str, Any]] = []
    for role, path in sorted(paths.items()):
        item: dict[str, Any] = {"role": role, "path": path, "expected_format": "path" if role == "t5_tokenizer" else "safetensors"}
        if role in explicit:
            item["source"] = "model_paths"
        elif isinstance(downloads.get(role), dict):
            source = downloads[role]
            item.update({key: source[key] for key in ("repo", "repo_id", "filename", "file", "revision") if isinstance(source.get(key), str)})
        models.append(item)
    return {"schema_version": 1, "backend": "sd-scripts", "architecture": sd_scripts_architecture(run), "mode": sd_scripts_mode(run), "models": models, "output": {"compatibility": "comfyui"}}
