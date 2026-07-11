"""AI-Toolkit backend adapter."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from kura.backends.common import _datasets
from kura.fsio import atomic_write_yaml
from kura.run_envelope import backend_config, common_recipe, legacy_params


def _ai_toolkit_datasets(datasets: list[dict[str, Any]], override_folder: Any, resolution: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        dataset_id = dataset.get("id", "")
        folder = override_folder if index == 0 and isinstance(override_folder, str) and override_folder else f"/workspace/datasets/{dataset_id}/images"
        entries.append({"folder_path": folder, "caption_ext": ".txt", "cache_latents_to_disk": True, "resolution": resolution})
    return entries


def _ai_toolkit_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "ai-toolkit")


def compile_ai_toolkit(run: dict[str, Any], destination: Path) -> None:
    """Write AI-Toolkit native YAML for configured training runs."""
    override = _ai_toolkit_backend_override(run)
    params = legacy_params(run)
    recipe = common_recipe(run)
    model = run.get("model", {})
    datasets = _datasets(run)
    native = override.get("config")
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
                "train": {"batch_size": params.get("batch_size"), "steps": recipe.get("steps"), "gradient_accumulation_steps": 1, "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": False, "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": params.get("lr"), "dtype": "bf16", "disable_sampling": True, "seed": recipe.get("seed")},
                "model": {"name_or_path": model.get("base"), "arch": override.get("model_arch"), "quantize": False, "quantize_te": False, "low_vram": False},
            }],
        },
    }
    process = config["config"]["process"][0]
    if "config" in override and not isinstance(native, dict):
        location = "backend.config.config" if isinstance(run.get("backend"), dict) and run["backend"].get("config") else "backend_overrides.ai-toolkit.config"
        raise ValueError(f"{location} must be a mapping.")
    if isinstance(native, dict):
        for section, values in native.items():
            if section in process and isinstance(process[section], dict) and isinstance(values, dict):
                process[section].update(deepcopy(values))
            else:
                process[section] = deepcopy(values)
    atomic_write_yaml(destination.with_suffix(".yaml"), config)


def command_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Return a container-native command spec, without executing it."""
    command = _ai_toolkit_backend_override(run).get("command")
    if command is None:
        return {"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", f"/workspace/runs/{run['id']}/resolved/ai-toolkit.yaml"], "env": {}}
    if not isinstance(command, dict):
        raise ValueError(
            "AI-Toolkit command is not configured. "
            "Set backend.config.command."
        )
    cwd, argv, env = command.get("cwd"), command.get("argv"), command.get("env", {})
    if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("AI-Toolkit command must provide string cwd and argv values.")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("AI-Toolkit command env must be a string-to-string mapping.")
    if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
        raise ValueError("AI-Toolkit command env must not contain secrets; use the process environment instead.")
    return {"cwd": cwd, "argv": argv, "env": env}
