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


@lru_cache(maxsize=1)
def load_baseline() -> dict[str, Any]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), dict):
        raise ValueError(f"{BASELINE_PATH.name} is not a schema-1 AI-Toolkit baseline")
    return payload


def select_baseline_entry(model_arch: str, normalized_arch: str, model_base: object) -> tuple[str, dict[str, Any]] | None:
    """Pick the UI entry the user would have selected, or None when there is none.

    ``model_arch`` is the authored selector (it may carry a UI tag such as
    ``zimage:turbo`` or the ``flex1`` alias); ``normalized_arch`` is the arch
    AI-Toolkit trains. Several UI entries can share one arch. The entry whose
    model path equals ``model.base`` wins, then the entry named exactly like
    the selector, then the entry named after the arch, then the only entry for
    that arch. Anything else is ambiguous and returns None.
    """
    entries = {name: entry for name, entry in load_baseline()["entries"].items() if entry.get("arch") == normalized_arch}
    if not entries:
        return None
    for name, entry in sorted(entries.items()):
        if isinstance(model_base, str) and model_base and entry.get("name_or_path") == model_base:
            return name, entry
    for name in (model_arch, normalized_arch):
        if name in entries:
            return name, entries[name]
    if len(entries) == 1:
        return next(iter(entries.items()))
    return None
