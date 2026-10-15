"""Pinned AI-Toolkit training baseline (docs/adr/upstream-training-baseline.md).

The baseline is extracted from the pinned image by
scripts/extract_ai_toolkit_baseline.py and records the configuration the
AI-Toolkit UI builds for each architecture. Kura fills only unset values from
it; authored values always win.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


BASELINE_PATH = Path(__file__).with_name("ai_toolkit_baseline.json")

# UI entry names whose AI-Toolkit arch differs from the entry name. The pinned
# UI entry "sd15" builds a Stable Diffusion 1.5 job
# (stable-diffusion-v1-5/stable-diffusion-v1-5), which the pinned registry and
# Kura select as "sd1".
UI_ARCH_ALIASES = {"sd15": "sd1"}

# Native keys the baseline may fill, by section. Everything else stays either
# Kura-owned (paths, steps, seed, sampling) or authored.
BASELINE_KEYS = {
    "train": ("noise_scheduler", "timestep_type", "dtype", "optimizer", "optimizer_params", "lr",
              "content_or_style", "loss_type", "unload_text_encoder", "cache_text_embeddings"),
    "model": ("quantize", "qtype", "quantize_te", "qtype_te", "low_vram", "model_kwargs"),
    "dataset": ("cache_latents_to_disk",),
}


def trained_arch(model_arch: str) -> str:
    """The arch the pinned trainer trains for ``model_arch``: it drops a UI ``:tag`` and reads
    ``flex1`` as ``flux`` (toolkit/config_modules.py:768-775), so Kura writes the value as given."""
    arch = model_arch.split(":", 1)[0]
    return "flux" if arch == "flex1" else arch


def ui_entry_values(pinned_archs: frozenset[str]) -> tuple[str, ...]:
    """UI entry names that are model_arch values of their own: the trainer reads each as a
    pinned arch (``zimage:turbo``, ``flex1``), though it is not one by that name."""
    return tuple(sorted(
        name for name in load_baseline()["entries"] if name not in pinned_archs and trained_arch(name) in pinned_archs
    ))


@lru_cache(maxsize=1)
def load_baseline() -> dict[str, Any]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), dict):
        raise ValueError(f"{BASELINE_PATH.name} is not a schema-1 AI-Toolkit baseline")
    return payload


def select_baseline_entry(model_arch: str, model_base: object) -> tuple[str, dict[str, Any]] | None:
    """The UI entry the declared ``model_arch`` names, or None when there is none.

    That is the entry of that name, else the one entry whose arch it is (``sd1`` is the UI's
    ``sd15``). ``model.base`` never chooses: a base that is another entry's model path, and not
    this entry's, is refused as a conflict.
    """
    entries = load_baseline()["entries"]
    candidates = [name for name, entry in entries.items() if entry.get("arch") == model_arch]
    name = model_arch if model_arch in entries else candidates[0] if len(candidates) == 1 else None
    if name is None:
        return None
    entry = entries[name]
    if isinstance(model_base, str) and model_base and model_base != entry.get("name_or_path"):
        others = sorted(other for other, item in entries.items() if item.get("name_or_path") == model_base)
        if others:
            raise ValueError(
                f"AI-Toolkit model.base {model_base!r} is the model of UI entry {', '.join(others)}, "
                f"but model_arch {model_arch!r} selects entry {name!r} ({entry.get('name_or_path')}); "
                "set model_arch to the entry you train, or point model.base at its model"
            )
    return name, entry
