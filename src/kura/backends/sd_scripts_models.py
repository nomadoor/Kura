"""sd-scripts explicit model roles, downloads, and provenance projections."""

from __future__ import annotations

import json
from typing import Any

from kura.backends.shared import download_specs_for, model_downloads, recorded_model_source
from kura.backends.shared import explicit_model_paths as shared_explicit_model_paths
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


def _component(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in value).strip(".-")
    if not clean:
        raise ValueError(f"invalid sd-scripts model cache component: {value}")
    return clean


def _cache_path(repo_id: str, role: str, filename: str) -> str:
    return f"/workspace/cache/models/sd-scripts/{_component(repo_id.replace('/', '--'))}/{_component(role)}/{filename}"


def explicit_model_paths(run: dict[str, Any]) -> dict[str, str]:
    return shared_explicit_model_paths(sd_scripts_native(run), label="sd-scripts")


def _sd_scripts_download_entries(run: dict[str, Any]) -> list[dict[str, Any]]:
    return model_downloads(
        sd_scripts_native(run).get("model_downloads"), label="sd-scripts", explicit=explicit_model_paths(run), cache_path=_cache_path,
    )


def sd_scripts_model_download_specs(run: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    return download_specs_for(_sd_scripts_download_entries(run))


def sd_scripts_model_paths(run: dict[str, Any]) -> dict[str, str]:
    paths = explicit_model_paths(run)
    paths.update(sd_scripts_model_download_specs(run)[1])
    required = ROLE_CONTRACTS.get((sd_scripts_architecture(run), sd_scripts_mode(run)))
    if required is None:
        return paths
    missing = [role for role in required if not paths.get(role)]
    if missing:
        raise ValueError("sd-scripts model_paths/model_downloads missing required role(s): " + ", ".join(missing))
    return paths


def sd_scripts_download_commands(run: dict[str, Any]) -> tuple[list[list[str]], dict[str, str]]:
    specs, paths = sd_scripts_model_download_specs(run)
    if not specs:
        return [], paths
    return [["python", "-c", script_source("hf_download.py"), json.dumps(specs, ensure_ascii=False)]], paths


def requirements_sd_scripts(run: dict[str, Any], download_estimate: dict[str, Any] | None = None, *, declared: bool = False) -> list[dict[str, Any]]:
    estimate = download_estimate or {}
    if declared:
        specs, _ = sd_scripts_model_download_specs(run)
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
    downloaded = {entry["role"]: entry for entry in _sd_scripts_download_entries(run)}
    models: list[dict[str, Any]] = []
    for role, path in sorted(paths.items()):
        item: dict[str, Any] = {"role": role, "path": path, "expected_format": "path" if role == "t5_tokenizer" else "safetensors"}
        if role in explicit:
            item["source"] = "model_paths"
        elif role in downloaded:
            item.update(recorded_model_source(downloaded[role]))
        models.append(item)
    return {"schema_version": 1, "backend": "sd-scripts", "architecture": sd_scripts_architecture(run), "mode": sd_scripts_mode(run), "models": models, "output": {"compatibility": "comfyui"}}
