"""Musubi model bundle, download, and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kura.container_scripts import script_source
from kura.backends.common import _musubi_backend_override, _truthy
from kura.provenance import artifact_pinning

MUSUBI_ADAPTER_SCRIPTS: dict[str, tuple[str, ...]] = {
    "flux2": (
        "flux_2_train_network.py",
        "flux_2_cache_latents.py",
        "flux_2_cache_text_encoder_outputs.py",
    ),
    "wan": (
        "wan_train_network.py",
        "wan_cache_latents.py",
        "wan_cache_text_encoder_outputs.py",
    ),
    "krea2": (
        "krea2_train_network.py",
        "krea2_cache_latents.py",
        "krea2_cache_text_encoder_outputs.py",
    ),
    "qwen_image": (
        "qwen_image_train_network.py",
        "qwen_image_cache_latents.py",
        "qwen_image_cache_text_encoder_outputs.py",
    ),
    "zimage": (
        "zimage_train_network.py",
        "zimage_cache_latents.py",
        "zimage_cache_text_encoder_outputs.py",
    ),
    "flux_kontext": (
        "flux_kontext_train_network.py",
        "flux_kontext_cache_latents.py",
        "flux_kontext_cache_text_encoder_outputs.py",
    ),
    "ideogram4": (
        "ideogram4_train_network.py",
        "ideogram4_cache_latents.py",
        "ideogram4_cache_text_encoder_outputs.py",
    ),
    "hidream_o1": (
        "hidream_o1_train_network.py",
        "hidream_o1_cache_pixel.py",
        "hidream_o1_cache_text_encoder_outputs.py",
    ),
    "hunyuan_video": (
        "hv_train_network.py",
        "cache_latents.py",
        "cache_text_encoder_outputs.py",
    ),
    "hunyuan_video_1_5": (
        "hv_1_5_train_network.py",
        "hv_1_5_cache_latents.py",
        "hv_1_5_cache_text_encoder_outputs.py",
    ),
    "framepack": (
        "fpack_train_network.py",
        "fpack_cache_latents.py",
        "fpack_cache_text_encoder_outputs.py",
    ),
    "kandinsky5": (
        "kandinsky5_train_network.py",
        "kandinsky5_cache_text_encoder_outputs.py",
        "kandinsky5_cache_latents.py",
    ),
}


def _musubi_model_paths(run: dict[str, Any]) -> dict[str, str]:
    override = _musubi_backend_override(run)
    clean = _musubi_explicit_model_paths(override)
    downloads = _musubi_model_downloads(run, existing_paths=clean)
    clean.update(downloads[1])
    if not clean:
        raise ValueError("Musubi Tuner requires model_paths, model_downloads, or a known model.base bundle")
    return clean


def _musubi_explicit_model_paths(override: dict[str, Any]) -> dict[str, str]:
    paths = override.get("model_paths")
    clean: dict[str, str] = {}
    if isinstance(paths, dict):
        for key, value in paths.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                clean[key] = value
    return clean


def _safe_download_filename(filename: str) -> str:
    if filename.startswith("/") or any(part in ("", ".", "..") for part in filename.split("/")):
        raise ValueError(f"invalid Hugging Face filename for Musubi Tuner: {filename}")
    return filename


def _safe_cache_component(value: str) -> str:
    component = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in value.strip())
    component = component.strip(".-")
    if not component or component in (".", ".."):
        raise ValueError(f"invalid Hugging Face cache component for Musubi Tuner: {value}")
    return component


def _musubi_model_cache_path(repo_id: str, key: str, filename: str) -> str:
    repo_component = _safe_cache_component(repo_id.replace("/", "--"))
    key_component = _safe_cache_component(key)
    return f"/workspace/cache/models/musubi/{repo_component}/{key_component}/{filename}"


def _flux2_klein_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model = run.get("model", {})
    base = str(model.get("base") or "").lower().replace("_", "-")
    override = _musubi_backend_override(run)
    architecture = _musubi_architecture(run)
    if architecture not in ("flux2", "flux_2"):
        return {}
    model_version = str(override.get("model_version") or "").lower().replace("_", "-")
    bundle = str(override.get("model_bundle") or "auto").lower().replace("_", "-")
    if bundle in ("none", "off", "false"):
        return {}
    is_base_4b = (
        bundle in ("flux2-klein-base-4b", "flux2-klein-base-4b-comfy", "comfy-flux2-klein-base-4b")
        or model_version == "klein-base-4b"
        or base in {"black-forest-labs/flux.2-klein-base-4b", "flux.2-klein-base-4b", "flux2-klein-base-4b"}
    )
    is_distilled_4b = (
        bundle in ("flux2-klein-4b", "flux2-klein-4b-comfy", "comfy-flux2-klein-4b")
        or model_version == "klein-4b"
        or base in {"black-forest-labs/flux.2-klein-4b", "flux.2-klein-4b", "flux2-klein-4b"}
    )
    is_base_9b = (
        bundle in ("flux2-klein-base-9b", "bfl-flux2-klein-base-9b")
        or model_version == "klein-base-9b"
        or base in {"black-forest-labs/flux.2-klein-base-9b", "flux.2-klein-base-9b", "flux2-klein-base-9b"}
    )
    is_distilled_9b = (
        bundle in ("flux2-klein-9b", "bfl-flux2-klein-9b")
        or model_version == "klein-9b"
        or base in {"black-forest-labs/flux.2-klein-9b", "flux.2-klein-9b", "flux2-klein-9b"}
    )
    if is_base_4b or is_distilled_4b:
        repo = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
        dit_name = "flux-2-klein-4b.safetensors" if is_distilled_4b else "flux-2-klein-base-4b.safetensors"
        return {
            "dit": {"repo": repo, "filename": f"split_files/diffusion_models/{dit_name}"},
            "vae": {"repo": repo, "filename": "split_files/vae/flux2-vae.safetensors"},
            "text_encoder": {"repo": repo, "filename": "split_files/text_encoders/qwen_3_4b.safetensors"},
        }
    if is_base_9b or is_distilled_9b:
        model_repo = "black-forest-labs/FLUX.2-klein-base-9B" if is_base_9b else "black-forest-labs/FLUX.2-klein-9B"
        dit_name = "flux-2-klein-base-9b.safetensors" if is_base_9b else "flux-2-klein-9b.safetensors"
        text_files = [f"text_encoder/model-0000{index}-of-00004.safetensors" for index in range(1, 5)]
        return {
            "dit": {"repo": model_repo, "filename": dit_name},
            "vae": {"repo": model_repo, "filename": "vae/diffusion_pytorch_model.safetensors"},
            "text_encoder": {"repo": model_repo, "filename": text_files[0], "filenames": [*text_files, "text_encoder/model.safetensors.index.json"]},
        }
    return {}


def _krea2_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    override = _musubi_backend_override(run)
    architecture = _musubi_architecture(run)
    if architecture not in ("krea2", "krea_2"):
        return {}
    bundle = str(override.get("model_bundle") or "auto").lower().replace("_", "-")
    if bundle in ("none", "off", "false"):
        return {}
    downloads: dict[str, dict[str, Any]] = {
        "dit": {"repo": "krea/Krea-2-Raw", "filename": "raw.safetensors"},
        "vae": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/vae/qwen_image_vae.safetensors"},
        "text_encoder": {"repo": "Comfy-Org/Qwen3-VL", "filename": "text_encoders/qwen3vl_4b_bf16.safetensors"},
    }
    if _truthy(override.get("include_turbo_dit")):
        downloads["turbo_dit"] = {"repo": "krea/Krea-2-Turbo", "filename": "turbo.safetensors"}
    return downloads


def _known_musubi_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    downloads = _flux2_klein_bundle(run)
    downloads.update(_krea2_bundle(run))
    return downloads


def musubi_model_download_specs(run: dict[str, Any], existing_paths: dict[str, str] | None = None) -> tuple[list[dict[str, str]], dict[str, str]]:
    override = _musubi_backend_override(run)
    existing_paths = existing_paths or {}
    resolved_downloads: dict[str, Any] = _known_musubi_bundle(run)
    downloads = override.get("model_downloads")
    if isinstance(downloads, dict):
        resolved_downloads.update(downloads)
    if not resolved_downloads:
        return [], {}
    download_specs: list[dict[str, str]] = []
    paths: dict[str, str] = {}
    for key, value in resolved_downloads.items():
        if key in existing_paths:
            continue
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("Musubi Tuner model_downloads must map model keys to download mappings")
        repo_id = value.get("repo_id") or value.get("repo")
        filenames_value = value.get("filenames")
        filenames = [item for item in filenames_value if isinstance(item, str) and item] if isinstance(filenames_value, list) else []
        filename = value.get("filename") or value.get("file") or (filenames[0] if filenames else None)
        if not isinstance(repo_id, str) or not repo_id or not isinstance(filename, str) or not filename:
            raise ValueError(f"Musubi Tuner model_downloads.{key} requires repo_id and filename")
        filename = _safe_download_filename(filename)
        filenames = [_safe_download_filename(item) for item in (filenames or [filename])]
        if value.get("local_dir"):
            raise ValueError("Musubi Tuner model_downloads.local_dir is not supported; use HF_HOME cache or explicit model_paths")
        for item_filename in filenames:
            item = {
                "key": key,
                "repo_id": repo_id,
                "filename": item_filename,
                "link_path": _musubi_model_cache_path(repo_id, key, item_filename),
            }
            revision = value.get("revision")
            if isinstance(revision, str) and revision:
                item["revision"] = revision
            repo_type = value.get("repo_type")
            if isinstance(repo_type, str) and repo_type:
                item["repo_type"] = repo_type
            download_specs.append(item)
        paths[key] = _musubi_model_cache_path(repo_id, key, filename)
    return download_specs, paths


def _musubi_model_downloads(run: dict[str, Any], existing_paths: dict[str, str] | None = None) -> tuple[list[list[str]], dict[str, str]]:
    download_specs, paths = musubi_model_download_specs(run, existing_paths=existing_paths)
    if not download_specs:
        return [], {}
    code = script_source("hf_download.py")
    return [["python", "-c", code, json.dumps(download_specs, ensure_ascii=False)]], paths


def requirements_musubi(run: dict[str, Any], download_estimate: dict[str, Any] | None = None, *, declared: bool = False) -> list[dict[str, Any]]:
    estimate = download_estimate or {}
    native = _musubi_backend_override(run)
    if declared:
        paths = native.get("model_paths") if isinstance(native.get("model_paths"), dict) else {}
        existing = {key: value for key, value in paths.items() if isinstance(key, str) and isinstance(value, str)}
        specs, _ = musubi_model_download_specs(run, existing_paths=existing)
        estimate = {"items": [{"key": item.get("key"), "repo_id": item.get("repo_id"), "filename": item.get("filename"), "revision": item.get("revision"), "runtime_reference": item.get("link_path"), "size_status": "not-measured", "measurement_scope": "compile", "size_bytes": None, "cached": False} for item in specs]}
    requirements: list[dict[str, Any]] = []
    for item in estimate.get("items") if isinstance(estimate.get("items"), list) else []:
        if not isinstance(item, dict):
            continue
        identity = {"kind": "huggingface-file", "repo_id": item.get("repo_id"), "filename": item.get("filename")}
        if item.get("revision"):
            identity["revision"] = item["revision"]
        measurement = {"scope": item.get("measurement_scope") or "controller", "status": item.get("size_status") or "unknown", "size_bytes": item.get("size_bytes"), "cached": bool(item.get("cached"))}
        if item.get("size_detail"):
            measurement["detail"] = item["size_detail"]
        requirements.append({"role": item.get("key") or "model", "acquisition": "kura", "identity": identity, "runtime_reference": item.get("runtime_reference"), "expected_format": "backend-role-file", "measurement": measurement, "pinning": artifact_pinning(identity, observable=True)})
    model_paths = native.get("model_paths")
    if isinstance(model_paths, dict):
        for role, path in sorted(model_paths.items()):
            if isinstance(role, str) and isinstance(path, str) and path:
                identity = {"kind": "path", "path": path}
                requirements.append({"role": role, "acquisition": "local-path", "identity": identity, "runtime_reference": path, "expected_format": "backend-role-file", "measurement": {"scope": "compile", "status": "declared"}, "pinning": artifact_pinning(identity, observable=True)})
    return requirements


def _musubi_architecture(run: dict[str, Any]) -> str:
    override = _musubi_backend_override(run)
    value = override.get("architecture") or override.get("model_arch")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Musubi backend.config.architecture is required")
    return value.lower().replace("-", "_")


def _unsupported_musubi_adapter_error(architecture: str) -> ValueError:
    return ValueError(
        "unsupported Kura built-in Musubi adapter: "
        f"{architecture}. Musubi Tuner may support this architecture upstream, "
        "but Kura does not generate its command automatically yet. "
        "Use backend.config.command for an explicit command, "
        "or add a Kura adapter."
    )


def _musubi_flux2_model_version(run: dict[str, Any]) -> str:
    override = _musubi_backend_override(run)
    model_version = str(override.get("model_version") or "").lower().replace("_", "-")
    if model_version:
        return model_version

    model = run.get("model", {})
    base = str(model.get("base") or "").lower().replace("_", "-")
    bundle = str(override.get("model_bundle") or "").lower().replace("_", "-")
    candidates = {base, bundle}
    if candidates & {"black-forest-labs/flux.2-klein-base-9b", "flux.2-klein-base-9b", "flux2-klein-base-9b", "bfl-flux2-klein-base-9b"}:
        return "klein-base-9b"
    if candidates & {"black-forest-labs/flux.2-klein-9b", "flux.2-klein-9b", "flux2-klein-9b", "bfl-flux2-klein-9b"}:
        return "klein-9b"
    if candidates & {"black-forest-labs/flux.2-klein-base-4b", "flux.2-klein-base-4b", "flux2-klein-base-4b", "flux2-klein-base-4b-comfy", "comfy-flux2-klein-base-4b"}:
        return "klein-base-4b"
    if candidates & {"black-forest-labs/flux.2-klein-4b", "flux.2-klein-4b", "flux2-klein-4b", "flux2-klein-4b-comfy", "comfy-flux2-klein-4b"}:
        return "klein-4b"
    raise ValueError("Musubi FLUX.2 requires backend.config.model_version or a recognized model.base/model_bundle; refusing to default to 4B")


def _musubi_model_expectations(run: dict[str, Any]) -> dict[str, str]:
    architecture = _musubi_architecture(run)
    override = _musubi_backend_override(run)
    model_version = str(override.get("model_version") or "").lower().replace("_", "-")
    model_base = str(run.get("model", {}).get("base") or "").lower().replace("_", "-")
    if model_version == "dev" or "flux.2-dev" in model_base or "flux2-dev" in model_base:
        text_encoder_format = "safetensors"
    else:
        text_encoder_format = "qwen3_8b_text_encoder" if "9b" in model_version or "9b" in model_base else "qwen3_4b_text_encoder"
    defaults: dict[str, dict[str, str]] = {
        "flux2": {
            "dit": "flux2_dit",
            "vae": "flux2_ae_or_vae",
            "text_encoder": text_encoder_format,
        },
        "flux_2": {
            "dit": "flux2_dit",
            "vae": "flux2_ae_or_vae",
            "text_encoder": text_encoder_format,
        },
        "wan": {
            "dit": "safetensors",
            "dit_high_noise": "safetensors",
            "vae": "safetensors",
            "t5": "file",
            "clip": "file",
        },
        "krea2": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
            "turbo_dit": "safetensors",
        },
        "krea_2": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
            "turbo_dit": "safetensors",
        },
        "qwen_image": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "qwen": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "zimage": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "z_image": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "flux_kontext": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder1": "safetensors",
            "text_encoder2": "safetensors",
        },
        "flux1_kontext": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder1": "safetensors",
            "text_encoder2": "safetensors",
        },
        "ideogram4": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "ideogram_4": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "hidream_o1": {
            "dit": "safetensors",
        },
        "hidream": {
            "dit": "safetensors",
        },
        "hunyuan_video": {
            "dit": "safetensors",
            "vae": "file",
            "text_encoder1": "hf_model_id_or_path",
            "text_encoder2": "hf_model_id_or_path",
        },
        "hunyuanvideo": {
            "dit": "safetensors",
            "vae": "file",
            "text_encoder1": "hf_model_id_or_path",
            "text_encoder2": "hf_model_id_or_path",
        },
        "hunyuan_video_1_5": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "hf_model_id_or_path",
            "byt5": "hf_model_id_or_path",
            "image_encoder": "hf_model_id_or_path",
        },
        "framepack": {
            "dit": "safetensors",
            "vae": "file",
            "text_encoder1": "hf_model_id_or_path",
            "text_encoder2": "hf_model_id_or_path",
            "image_encoder": "hf_model_id_or_path",
        },
        "frame_pack": {
            "dit": "safetensors",
            "vae": "file",
            "text_encoder1": "hf_model_id_or_path",
            "text_encoder2": "hf_model_id_or_path",
            "image_encoder": "hf_model_id_or_path",
        },
        "kandinsky5": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder_qwen": "hf_model_id_or_path",
            "text_encoder_clip": "hf_model_id_or_path",
        },
        "kandinsky_5": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder_qwen": "hf_model_id_or_path",
            "text_encoder_clip": "hf_model_id_or_path",
        },
    }
    expectations = dict(defaults.get(architecture, {}))
    user_expectations = override.get("model_expectations")
    if isinstance(user_expectations, dict):
        for key, value in user_expectations.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                expectations[key] = value
    return expectations


def _musubi_model_sources(run: dict[str, Any], paths: dict[str, str]) -> dict[str, dict[str, str]]:
    override = _musubi_backend_override(run)
    explicit_paths = _musubi_explicit_model_paths(override)
    downloads = _known_musubi_bundle(run)
    user_downloads = override.get("model_downloads")
    if isinstance(user_downloads, dict):
        downloads.update({key: value for key, value in user_downloads.items() if isinstance(key, str) and isinstance(value, dict)})
    sources: dict[str, dict[str, str]] = {}
    for role, path in paths.items():
        source: dict[str, str] = {"path": path}
        if role in explicit_paths:
            sources[role] = {**source, "source": "model_paths"}
            continue
        download = downloads.get(role)
        if isinstance(download, dict):
            repo = download.get("repo_id") or download.get("repo")
            filename = download.get("filename") or download.get("file")
            if isinstance(repo, str):
                source["repo"] = repo
            if isinstance(filename, str):
                source["filename"] = filename
            if isinstance(download.get("revision"), str):
                source["revision"] = download["revision"]
        else:
            source["source"] = "model_paths"
        sources[role] = source
    return sources


def _musubi_model_lock(run: dict[str, Any]) -> dict[str, Any]:
    paths = _musubi_model_paths(run)
    expectations = _musubi_model_expectations(run)
    sources = _musubi_model_sources(run, paths)
    return {
        "schema_version": 1,
        "backend": "musubi-tuner",
        "architecture": _musubi_architecture(run),
        "models": [
            {
                "role": role,
                "path": path,
                "expected_format": expectations.get(role, "safetensors"),
                **{key: value for key, value in sources.get(role, {}).items() if key != "path"},
            }
            for role, path in sorted(paths.items())
        ],
        "output": _musubi_output_compatibility(run),
    }


def _musubi_output_compatibility(run: dict[str, Any]) -> dict[str, str]:
    override = _musubi_backend_override(run)
    value = override.get("output_compatibility") or override.get("output_format") or "comfyui"
    return {"lora_format": str(value)}


def _safetensors_validator_code() -> str:
    return script_source("safetensors_validator.py")


def _musubi_model_validation_command(run: dict[str, Any], paths: dict[str, str]) -> list[str]:
    expectations = _musubi_model_expectations(run)
    spec = {
        "architecture": _musubi_architecture(run),
        "models": [
            {"role": role, "path": path, "expected_format": expectations.get(role, "safetensors")}
            for role, path in sorted(paths.items())
            if role in expectations or path.endswith(".safetensors")
        ],
    }
    return ["python", "-c", _safetensors_validator_code(), json.dumps(spec, ensure_ascii=False)]


def _musubi_lora_validation_command(run: dict[str, Any], output_dir: str, output_name: str) -> list[str]:
    compatibility = _musubi_output_compatibility(run)["lora_format"].lower()
    spec = {
        "architecture": _musubi_architecture(run),
        "lora": {
            "pattern": f"{output_dir.rstrip('/')}/{output_name}*.safetensors",
            "compatibility": compatibility,
        },
    }
    return ["python", "-c", _safetensors_validator_code(), json.dumps(spec, ensure_ascii=False)]
