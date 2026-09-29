"""Musubi-specific backend adapter helpers."""

from __future__ import annotations

from typing import Any

from kura.run_envelope import backend_config


# Authored aliases are normalized once before command, model, task, or dataset
# tables see an architecture. These names remain Musubi-local vocabulary.
MUSUBI_ARCHITECTURE_ALIASES = {
    "flux_2": "flux2",
    "krea_2": "krea2",
    "minimaxh3": "minimax_h3",
    "qwen": "qwen_image",
    "z_image": "zimage",
    "flux1_kontext": "flux_kontext",
    "ideogram_4": "ideogram4",
    "hidream": "hidream_o1",
    "hunyuanvideo": "hunyuan_video",
    "frame_pack": "framepack",
    "kandinsky_5": "kandinsky5",
}

# Pinned Musubi v0.3.5 dataset helpers use these short names
# (dataset/architectures.py at 4e7c714, lines 1-35).
MUSUBI_NATIVE_DATASET_ARCHITECTURES = {
    "hunyuan_video": "hv",
    "wan": "wan",
    "framepack": "fp",
    "flux_kontext": "fk",
    "qwen_image": "qi",
    "kandinsky5": "k5",
    "hunyuan_video_1_5": "hv15",
    "zimage": "zi",
    "hidream_o1": "ho1",
    "ideogram4": "i4",
    "krea2": "kr2",
    "minimax_h3": "mmh3",
}


def canonical_musubi_architecture(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    return MUSUBI_ARCHITECTURE_ALIASES.get(normalized, normalized)


def musubi_native_dataset_architecture(value: str) -> str:
    canonical = canonical_musubi_architecture(value)
    native = MUSUBI_NATIVE_DATASET_ARCHITECTURES.get(canonical)
    if native is None:
        raise ValueError(
            f"Musubi architecture {canonical!r} has no pinned native dataset architecture"
        )
    return native


def _musubi_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "musubi-tuner")


def _musubi_architecture(run: dict[str, Any]) -> str:
    """Resolve and normalize the shared Musubi architecture selector aliases."""
    override = _musubi_backend_override(run)
    value = override.get("architecture") or override.get("model_arch")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Musubi backend.config.architecture is required")
    return canonical_musubi_architecture(value)


def _require_paths(paths: dict[str, str], names: tuple[str, ...]) -> list[str]:
    missing = [name for name in names if not paths.get(name)]
    if missing:
        raise ValueError("Musubi Tuner model_paths missing: " + ", ".join(missing))
    return [paths[name] for name in names]
