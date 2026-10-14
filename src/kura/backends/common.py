"""Musubi-specific backend adapter helpers."""

from __future__ import annotations

from typing import Any

from kura.run_envelope import backend_config


# Every architecture the Musubi adapter compiles; backend.config.architecture names one exactly.
MUSUBI_ARCHITECTURES = frozenset({
    "flux2", "flux_kontext", "framepack", "hidream_o1", "hunyuan_video", "hunyuan_video_1_5", "ideogram4",
    "kandinsky5", "krea2", "minimax_h3", "qwen_image", "wan", "zimage",
})

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


def musubi_native_dataset_architecture(value: str) -> str:
    native = MUSUBI_NATIVE_DATASET_ARCHITECTURES.get(value)
    if native is None:
        raise ValueError(
            f"Musubi architecture {value!r} has no pinned native dataset architecture"
        )
    return native


def _musubi_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "musubi-tuner")


def _musubi_architecture(run: dict[str, Any]) -> str:
    """The Musubi architecture selector, as written; validate_backend_config admits listed values only."""
    value = _musubi_backend_override(run).get("architecture")
    if not isinstance(value, str) or not value:
        raise ValueError("Musubi backend.config.architecture is required")
    return value


def _require_paths(paths: dict[str, str], names: tuple[str, ...]) -> list[str]:
    missing = [name for name in names if not paths.get(name)]
    if missing:
        raise ValueError("Musubi Tuner model_paths missing: " + ", ".join(missing))
    return [paths[name] for name in names]
