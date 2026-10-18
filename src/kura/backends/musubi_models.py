"""Musubi model bundle, download, and validation helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from kura.container_scripts import script_source
from kura.backends.common import MUSUBI_ARCHITECTURES, _musubi_architecture, _musubi_backend_override
from kura.backends.shared import _truthy, download_specs_for, explicit_model_paths, family_card_hint, model_downloads, model_source_facts, recorded_model_source
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


def _musubi_model_paths(run: dict[str, Any]) -> dict[str, str]:
    override = _musubi_backend_override(run)
    clean = _musubi_explicit_model_paths(override)
    clean.update(musubi_model_download_specs(run)[1])
    if not clean:
        raise ValueError(f"Musubi Tuner requires model_paths, model_downloads, or model_bundle; {family_card_hint(_musubi_architecture(run))}")
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


# The --model_version values the pinned Musubi Tuner accepts for FLUX.2 (flux_2/flux2_utils.py)
# and Qwen-Image (qwen_image/qwen_image_utils.py:1558-1571).
FLUX2_MODEL_VERSIONS = ("klein-base-4b", "klein-4b", "klein-base-9b", "klein-9b", "dev")
QWEN_IMAGE_MODEL_VERSIONS = ("original", "layered", "edit", "edit-2509", "edit-2511")
MODEL_VERSIONS = {"flux2": FLUX2_MODEL_VERSIONS, "qwen_image": QWEN_IMAGE_MODEL_VERSIONS}

# The expected_format values the container validator reads (container_scripts/safetensors_validator.py).
MODEL_FORMATS = (
    "safetensors", "file", "hf_model_id_or_path", "flux2_dit", "flux2_vae", "flux2_ae", "flux2_ae_or_vae",
    "qwen3_4b_text_encoder", "qwen3_8b_text_encoder",
)


def _flux2_klein_files(variant: str) -> dict[str, dict[str, Any]]:
    if variant in ("klein-base-4b", "klein-4b"):
        repo = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
        return {
            "dit": {"repo": repo, "filename": f"split_files/diffusion_models/flux-2-{variant}.safetensors"},
            "vae": {"repo": repo, "filename": "split_files/vae/flux2-vae.safetensors"},
            "text_encoder": {"repo": repo, "filename": "split_files/text_encoders/qwen_3_4b.safetensors"},
        }
    model_repo = "black-forest-labs/FLUX.2-klein-base-9B" if variant == "klein-base-9b" else "black-forest-labs/FLUX.2-klein-9B"
    text_files = [f"text_encoder/model-0000{index}-of-00004.safetensors" for index in range(1, 5)]
    return {
        "dit": {"repo": model_repo, "filename": f"flux-2-{variant}.safetensors"},
        "vae": {"repo": model_repo, "filename": "vae/diffusion_pytorch_model.safetensors"},
        "text_encoder": {"repo": model_repo, "filename": [*text_files, "text_encoder/model.safetensors.index.json"]},
    }


def _krea2_files(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    downloads: dict[str, dict[str, Any]] = {
        "dit": {"repo": "krea/Krea-2-Raw", "filename": "raw.safetensors"},
        "vae": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/vae/qwen_image_vae.safetensors"},
        "text_encoder": {"repo": "Comfy-Org/Qwen3-VL", "filename": "text_encoders/qwen3vl_4b_bf16.safetensors"},
    }
    if _truthy(_musubi_backend_override(run).get("include_turbo_dit")):
        downloads["turbo_dit"] = {"repo": "krea/Krea-2-Turbo", "filename": "turbo.safetensors"}
    return downloads


def _minimax_h3_files(family: str) -> dict[str, dict[str, Any]]:
    repo = "Comfy-Org/MiniMax-H3"
    return {
        "dit": {"repo": repo, "filename": f"diffusion_models/minimax_h3_{family}_pruned_int8_convrot.safetensors"},
        "video_vae": {"repo": repo, "filename": "vae/minimax_h3_video_vae_fp16.safetensors"},
        "audio_vae": {"repo": repo, "filename": "vae/minimax_h3_audio_vae_fp32.safetensors"},
        "text_encoder": {"repo": repo, "filename": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"},
    }


# Each model_bundle by its one name: the architecture it is for and the files it downloads.
MUSUBI_BUNDLES: dict[str, tuple[str, Any]] = {
    **{f"flux2-{variant}": ("flux2", lambda run, variant=variant: _flux2_klein_files(variant))
       for variant in ("klein-base-4b", "klein-4b", "klein-base-9b", "klein-9b")},
    "krea2-raw": ("krea2", _krea2_files),
    "minimax-h3-fl2va-pruned-int8": ("minimax_h3", lambda run: _minimax_h3_files("fl2va")),
    "minimax-h3-ref2va-pruned-int8": ("minimax_h3", lambda run: _minimax_h3_files("ref2va")),
}
# A Klein variant named in a model.base such as black-forest-labs/FLUX.2-klein-base-4B,
# flux-2-klein-9b.safetensors, or flux2-klein-9b_fp8.
_KLEIN_IN_BASE = re.compile(r"flux[.\-_]?2[-_]klein[-_](base[-_])?([49])b", re.IGNORECASE)


def _known_musubi_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The files the named model_bundle downloads; a run that names none downloads none."""
    override = _musubi_backend_override(run)
    name = override.get("model_bundle")
    if name is None:
        return {}
    architecture = _musubi_architecture(run)
    if name not in MUSUBI_BUNDLES:
        raise ValueError(f"Musubi model_bundle {name!r} is not one Kura knows; use one of: " + ", ".join(MUSUBI_BUNDLES))
    bundle_architecture, files = MUSUBI_BUNDLES[name]
    if bundle_architecture != architecture:
        raise ValueError(f"Musubi model_bundle {name!r} is for architecture {bundle_architecture}, not {architecture}")
    if "model_downloads" in override:
        raise ValueError(
            f"Musubi model_bundle {name!r} decides every downloaded file; remove model_downloads, "
            "or remove model_bundle and list each role in model_downloads"
        )
    if architecture == "flux2" and name != f"flux2-{_musubi_flux2_model_version(run)}":
        matching = f"flux2-{override['model_version']}"
        raise ValueError(
            f"Musubi model_bundle {name!r} downloads another FLUX.2 variant than model_version "
            f"{override['model_version']!r}; "
            + (f"use model_bundle {matching}" if matching in MUSUBI_BUNDLES else "no bundle downloads it; list its files in model_downloads")
        )
    return files(run)


def _musubi_download_entries(run: dict[str, Any]) -> list[dict[str, Any]]:
    override = _musubi_backend_override(run)
    # One place decides every downloaded file: the named bundle or model_downloads, never both.
    downloads = _known_musubi_bundle(run) or override.get("model_downloads")
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
    """The FLUX.2 variant comes from model_version alone; a model.base naming another Klein variant conflicts."""
    model_version = _musubi_model_version(run, default="")
    if not model_version:
        raise ValueError(
            "Musubi FLUX.2 requires backend.config.model_version, one of " + ", ".join(FLUX2_MODEL_VERSIONS)
        )
    named = _KLEIN_IN_BASE.search(str((run.get("model") or {}).get("base") or ""))
    variant = f"klein-{'base-' if named.group(1) else ''}{named.group(2)}b" if named else None
    if variant and variant != model_version:
        raise ValueError(
            f"Musubi FLUX.2 model.base {run['model']['base']!r} names variant {variant}, "
            f"but model_version is {model_version!r}; make them agree"
        )
    return model_version


def _musubi_model_version(run: dict[str, Any], *, default: str = "original") -> str:
    """The authored model_version, which must be one the architecture's trainer accepts."""
    override = _musubi_backend_override(run)
    value = override.get("model_version", default)
    accepted = MODEL_VERSIONS.get(_musubi_architecture(run), ())
    if "model_version" in override and value not in accepted:
        raise ValueError(f"Musubi backend.config.model_version {value!r} is not accepted here; use one of: " + ", ".join(accepted))
    return value


def _musubi_role_formats(architecture: str, flux2_model_version: str | None = None) -> dict[str, str]:
    """The model roles `architecture` reads and the format Kura checks for each."""
    text_encoder_format = "safetensors"
    if architecture == "flux2" and flux2_model_version != "dev":
        text_encoder_format = "qwen3_8b_text_encoder" if "9b" in str(flux2_model_version) else "qwen3_4b_text_encoder"
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
            "base_weights": "safetensors",
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
    return dict(defaults.get(architecture, {}))


def _musubi_model_expectations(run: dict[str, Any]) -> dict[str, str]:
    architecture = _musubi_architecture(run)
    override = _musubi_backend_override(run)
    expectations = _musubi_role_formats(architecture, _musubi_flux2_model_version(run) if architecture == "flux2" else None)
    authored = override.get("model_expectations", {})
    if not isinstance(authored, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in authored.items()):
        raise ValueError("Musubi model_expectations must map model roles to a format name")
    unknown_roles = sorted(set(authored) - set(expectations))
    unknown_formats = sorted({value for value in authored.values() if value not in MODEL_FORMATS})
    if unknown_roles or unknown_formats:
        raise ValueError(
            "Musubi model_expectations"
            + (f" names role(s) {', '.join(unknown_roles)} that {architecture} does not use (roles: {', '.join(sorted(expectations))})" if unknown_roles else "")
            + (f" names format(s) {', '.join(unknown_formats)}; use one of: {', '.join(MODEL_FORMATS)}" if unknown_formats else "")
        )
    return {**expectations, **authored}


def musubi_model_facts(architecture_filter: str | None = None) -> dict[str, Any]:
    """The model roles each architecture (or the one given) reads, with the format Kura checks, and
    the entry shapes that name them. Which roles a run needs depends on its task and options;
    compile names any missing one."""
    roles: dict[str, dict[str, Any]] = {}
    for architecture in (architecture_filter,) if architecture_filter else sorted(MUSUBI_ARCHITECTURES):
        for version in FLUX2_MODEL_VERSIONS if architecture == "flux2" else (None,):
            key = f"architecture={architecture}" + (f", model_version={version}" if version else "")
            roles[key] = {"formats": _musubi_role_formats(architecture, version)}
    return {
        "model_roles": roles,
        "model_sources": model_source_facts(),
    }


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
    return {"lora_format": override.get("output_compatibility", "comfyui")}


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
    compatibility = _musubi_output_compatibility(run)["lora_format"]
    spec = {
        "architecture": _musubi_architecture(run),
        "lora": {
            "pattern": f"{output_dir.rstrip('/')}/{output_name}*.safetensors",
            "compatibility": compatibility,
        },
    }
    return ["python", "-c", _safetensors_validator_code(), json.dumps(spec, ensure_ascii=False)]
