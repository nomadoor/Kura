"""Musubi model bundle, download, and validation helpers."""

from __future__ import annotations

import json
from typing import Any

from kura.container_scripts import script_source
from kura.backends.common import _musubi_architecture, _musubi_backend_override
from kura.backends.shared import _truthy, download_specs_for, explicit_model_paths, model_downloads, recorded_model_source
from kura.provenance import artifact_pinning

MUSUBI_ADAPTER_SCRIPTS: dict[str, tuple[str, ...]] = {
    "minimax_h3": (
        "minimax_h3_train_network.py",
        "minimax_h3_cache_latents.py",
        "minimax_h3_cache_text_encoder_outputs.py",
    ),
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


def _normalize_musubi_model_version(value: Any, *, default: str = "") -> str:
    """Return the one normalized spelling used by Musubi projection and argv."""
    return str(value or default).strip().lower().replace("_", "-")


def _musubi_model_paths(run: dict[str, Any]) -> dict[str, str]:
    override = _musubi_backend_override(run)
    clean = _musubi_explicit_model_paths(override)
    clean.update(musubi_model_download_specs(run)[1])
    if not clean:
        raise ValueError("Musubi Tuner requires model_paths, model_downloads, or a known model.base bundle")
    return clean


def _musubi_explicit_model_paths(override: dict[str, Any]) -> dict[str, str]:
    return explicit_model_paths(override, label="Musubi Tuner")


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


# Every accepted name for a FLUX.2 Klein variant, read from model_bundle and model.base alike.
_FLUX2_KLEIN_NAMES: dict[str, str] = {
    name: variant
    for variant, names in {
        "klein-base-4b": ("flux2-klein-base-4b-comfy", "comfy-flux2-klein-base-4b"),
        "klein-4b": ("flux2-klein-4b-comfy", "comfy-flux2-klein-4b"),
        "klein-base-9b": ("bfl-flux2-klein-base-9b",),
        "klein-9b": ("bfl-flux2-klein-9b",),
    }.items()
    for name in (
        *names,
        f"black-forest-labs/flux.2-{variant}",
        f"flux.2-{variant}",
        f"flux2-{variant}",
    )
}


def _flux2_klein_variant(run: dict[str, Any]) -> str | None:
    """Decide which FLUX.2 Klein variant a run trains: model_version, then model_bundle, then model.base."""
    override = _musubi_backend_override(run)
    model_version = _musubi_model_version(run, default="")
    if model_version:
        return model_version if model_version in _FLUX2_KLEIN_NAMES.values() else None
    for name in (override.get("model_bundle"), run.get("model", {}).get("base")):
        variant = _FLUX2_KLEIN_NAMES.get(_normalize_musubi_model_version(name))
        if variant:
            return variant
    return None


def _flux2_klein_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    override = _musubi_backend_override(run)
    if _musubi_architecture(run) != "flux2":
        return {}
    if _normalize_musubi_model_version(override.get("model_bundle"), default="auto") in ("none", "off", "false"):
        return {}
    variant = _flux2_klein_variant(run)
    if variant in ("klein-base-4b", "klein-4b"):
        repo = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
        dit_name = f"flux-2-{variant}.safetensors"
        return {
            "dit": {"repo": repo, "filename": f"split_files/diffusion_models/{dit_name}"},
            "vae": {"repo": repo, "filename": "split_files/vae/flux2-vae.safetensors"},
            "text_encoder": {"repo": repo, "filename": "split_files/text_encoders/qwen_3_4b.safetensors"},
        }
    if variant in ("klein-base-9b", "klein-9b"):
        model_repo = "black-forest-labs/FLUX.2-klein-base-9B" if variant == "klein-base-9b" else "black-forest-labs/FLUX.2-klein-9B"
        dit_name = f"flux-2-{variant}.safetensors"
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
    if architecture != "krea2":
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


def _minimax_h3_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    override = _musubi_backend_override(run)
    architecture = _musubi_architecture(run)
    if architecture != "minimax_h3":
        return {}
    bundle = str(override.get("model_bundle") or "auto").lower().replace("_", "-")
    if bundle in ("none", "off", "false"):
        return {}
    aliases = {
        "minimax-h3-pruned-int8": "auto",
        "minimax-h3-fl2va-pruned-int8": "fl2va",
        "minimax-h3-ref2va-pruned-int8": "ref2va",
    }
    bundle = aliases.get(bundle, bundle)
    if bundle not in ("auto", "fl2va", "ref2va"):
        raise ValueError(
            "Musubi MiniMax-H3 model_bundle must be auto, minimax-h3-fl2va-pruned-int8, "
            "minimax-h3-ref2va-pruned-int8, or none"
        )
    task = str(override.get("task") or "t2va").lower()
    transformer_family = "ref2va" if bundle == "ref2va" or (bundle == "auto" and task == "ref2va") else "fl2va"
    repo = "Comfy-Org/MiniMax-H3"
    return {
        "dit": {
            "repo": repo,
            "filename": f"diffusion_models/minimax_h3_{transformer_family}_pruned_int8_convrot.safetensors",
        },
        "video_vae": {"repo": repo, "filename": "vae/minimax_h3_video_vae_fp16.safetensors"},
        "audio_vae": {"repo": repo, "filename": "vae/minimax_h3_audio_vae_fp32.safetensors"},
        "text_encoder": {
            "repo": repo,
            "filename": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        },
    }


def _known_musubi_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    downloads = _flux2_klein_bundle(run)
    downloads.update(_krea2_bundle(run))
    downloads.update(_minimax_h3_bundle(run))
    return downloads


def _musubi_download_entries(run: dict[str, Any]) -> list[dict[str, Any]]:
    override = _musubi_backend_override(run)
    authored = override.get("model_downloads")
    # A known bundle fills the roles the user did not declare; a malformed declaration goes to the owner as is.
    downloads = {**_known_musubi_bundle(run), **authored} if isinstance(authored, dict) else authored if authored is not None else _known_musubi_bundle(run)
    return model_downloads(
        downloads, label="Musubi Tuner", explicit=_musubi_explicit_model_paths(override), cache_path=_musubi_model_cache_path,
    )


def musubi_model_download_specs(run: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    # MiniMax-H3 reads its weights through hard links inside the cache.
    extra = {"link_mode": "hardlink"} if _musubi_architecture(run) == "minimax_h3" else {}
    return download_specs_for(_musubi_download_entries(run), **extra)


def _musubi_model_downloads(run: dict[str, Any]) -> tuple[list[list[str]], dict[str, str]]:
    download_specs, paths = musubi_model_download_specs(run)
    if not download_specs:
        return [], {}
    code = script_source("hf_download.py")
    return [["python", "-c", code, json.dumps(download_specs, ensure_ascii=False)]], paths


def requirements_musubi(run: dict[str, Any], download_estimate: dict[str, Any] | None = None, *, declared: bool = False) -> list[dict[str, Any]]:
    estimate = download_estimate or {}
    native = _musubi_backend_override(run)
    if declared:
        specs, _ = musubi_model_download_specs(run)
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


def _unsupported_musubi_adapter_error(architecture: str) -> ValueError:
    return ValueError(
        "unsupported Kura built-in Musubi adapter: "
        f"{architecture}. Musubi Tuner may support this architecture upstream, "
        "but Kura does not generate its command automatically yet. "
        "Use backend.config.command for an explicit command, "
        "or add a Kura adapter."
    )


def _musubi_flux2_model_version(run: dict[str, Any]) -> str:
    model_version = _musubi_model_version(run, default="")
    if model_version:
        return model_version
    variant = _flux2_klein_variant(run)
    if variant:
        return variant
    raise ValueError("Musubi FLUX.2 requires backend.config.model_version or a recognized model.base/model_bundle; refusing to default to 4B")


def _musubi_model_version(run: dict[str, Any], *, default: str = "original") -> str:
    """Normalize the authored model version identically for every Musubi consumer."""
    override = _musubi_backend_override(run)
    return _normalize_musubi_model_version(
        override.get("model_version"), default=default,
    )


def _musubi_model_expectations(run: dict[str, Any]) -> dict[str, str]:
    architecture = _musubi_architecture(run)
    override = _musubi_backend_override(run)
    text_encoder_format = "safetensors"
    if architecture == "flux2":
        model_version = _musubi_flux2_model_version(run)
        if model_version != "dev":
            text_encoder_format = "qwen3_8b_text_encoder" if "9b" in model_version else "qwen3_4b_text_encoder"
    defaults: dict[str, dict[str, str]] = {
        "flux2": {
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
        "minimax_h3": {
            "dit": "safetensors",
            "video_vae": "safetensors",
            "audio_vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "qwen_image": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "zimage": {
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
        "ideogram4": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
        },
        "hidream_o1": {
            "dit": "safetensors",
        },
        "hunyuan_video": {
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
        "kandinsky5": {
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
    downloaded = {entry["role"]: entry for entry in _musubi_download_entries(run)}
    sources: dict[str, dict[str, str]] = {}
    for role, path in paths.items():
        if role in explicit_paths or role not in downloaded:
            sources[role] = {"path": path, "source": "model_paths"}
        else:
            sources[role] = {"path": path, **recorded_model_source(downloaded[role])}
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
    value = override.get("output_compatibility") or "comfyui"
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
