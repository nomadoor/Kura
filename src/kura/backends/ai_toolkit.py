"""AI-Toolkit backend adapter."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from kura.backends.common import _datasets


def _ai_toolkit_datasets(datasets: list[dict[str, Any]], override_folder: Any, resolution: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        dataset_id = dataset.get("id", "")
        folder = override_folder if index == 0 and isinstance(override_folder, str) and override_folder else f"/workspace/datasets/{dataset_id}/images"
        entries.append({"folder_path": folder, "caption_ext": ".txt", "cache_latents_to_disk": True, "resolution": resolution})
    return entries


def _ai_toolkit_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    overrides = run.get("backend_overrides")
    if not isinstance(overrides, dict):
        return {}
    override = overrides.get("ai-toolkit")
    return override if isinstance(override, dict) else {}


def _ai_toolkit_large_model_defaults(model_base: Any) -> bool:
    if not isinstance(model_base, str):
        return False
    normalized = model_base.lower().replace("_", "-")
    large_markers = (
        "flux",
        "kontext",
        "qwen",
        "hidream",
        "hunyuan",
        "wan",
        "z-image",
        "zimage",
        "krea",
    )
    return any(marker in normalized for marker in large_markers)


def compile_ai_toolkit(run: dict[str, Any], destination: Path) -> None:
    """Write AI-Toolkit native YAML for configured training runs."""
    override = _ai_toolkit_backend_override(run)
    params = run.get("params", {})
    model = run.get("model", {})
    datasets = _datasets(run)
    native = override.get("config")
    optimize_large_model = _ai_toolkit_large_model_defaults(model.get("base"))
    config = {
        "job": "extension",
        "config": {
            "name": run["id"],
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"/workspace/runs/{run['id']}/outputs",
                "device": "cuda:0",
                "network": {"type": "lora", "linear": params.get("rank"), "linear_alpha": params.get("alpha")},
                "save": {"dtype": "bf16", "save_every": 1, "max_step_saves_to_keep": 1},
                "datasets": _ai_toolkit_datasets(datasets, override.get("dataset_folder"), params.get("resolution")),
                "train": {"batch_size": params.get("batch_size"), "steps": params.get("steps"), "gradient_accumulation_steps": 1, "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": optimize_large_model, "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": params.get("lr"), "dtype": "bf16", "disable_sampling": True},
                "model": {"name_or_path": model.get("base"), "arch": override.get("model_arch"), "quantize": optimize_large_model, "quantize_te": optimize_large_model, "low_vram": False},
            }],
        },
    }
    process = config["config"]["process"][0]
    if "config" in override and not isinstance(native, dict):
        raise ValueError("backend_overrides.ai-toolkit.config must be a mapping.")
    if isinstance(native, dict):
        for section, values in native.items():
            if section in process and isinstance(process[section], dict) and isinstance(values, dict):
                process[section].update(deepcopy(values))
            else:
                process[section] = deepcopy(values)
    destination.with_suffix(".yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def command_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Return a container-native command spec, without executing it."""
    command = _ai_toolkit_backend_override(run).get("command")
    if command is None:
        return {"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", f"/workspace/runs/{run['id']}/resolved/ai-toolkit.yaml"], "env": {}}
    if not isinstance(command, dict):
        raise ValueError(
            "AI-Toolkit command is not configured. "
            "Set backend_overrides.ai-toolkit.command."
        )
    cwd, argv, env = command.get("cwd"), command.get("argv"), command.get("env", {})
    if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("AI-Toolkit command must provide string cwd and argv values.")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("AI-Toolkit command env must be a string-to-string mapping.")
    if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
        raise ValueError("AI-Toolkit command env must not contain secrets; use the process environment instead.")
    return {"cwd": cwd, "argv": argv, "env": env}
