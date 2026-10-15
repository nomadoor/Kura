"""Values a test run writes for the fields a backend requires, so a fixture states only what it tests."""

from __future__ import annotations

from typing import Any

from kura.backends.registry import get_backend

_VALUES: dict[str, Any] = {
    "learning_rate": 1e-4,
    "optimizer_type": "adamw8bit",
    "mixed_precision": "bf16",
    "network_dim": 32,
    "network_alpha": 32,
    "discrete_flow_shift": 3.0,
    "guidance_scale": 1.0,
    "model_prediction_type": "raw",
    "h3_guidance_loss_scale": 4.0,
    "h3_guidance_loss_sigma_min": 0.15,
}
_TIMESTEP_SAMPLING = {"flux2": "flux2_shift", "wan": "shift", "krea2": "krea2_shift", "hidream_o1": "uniform", "flux1": "flux_shift", "anima": "shift"}


def with_required(backend: str, config: dict[str, Any]) -> dict[str, Any]:
    """`config` plus a value for each field `backend` requires there and `config` leaves out."""
    surface = get_backend(backend).surface
    defaults = dict(surface.selector_defaults)
    filled = dict(config)
    for item in surface.required:
        applies = any(all(config.get(name, defaults.get(name)) in allowed for name, allowed in clause) for clause in item.when_any)
        if applies and item.field not in filled:
            filled[item.field] = _TIMESTEP_SAMPLING[config["architecture"]] if item.field == "timestep_sampling" else _VALUES[item.field]
    return filled
