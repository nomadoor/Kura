"""AI-Toolkit backend adapter."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from kura.backends.common import _datasets
from kura.fsio import atomic_write_yaml
from kura.run_envelope import backend_config, validated_recipe
from kura.provenance import artifact_pinning


def _ai_toolkit_datasets(datasets: list[dict[str, Any]], override_folder: Any, resolution: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        dataset_id = dataset.get("id", "")
        folder = override_folder if index == 0 and isinstance(override_folder, str) and override_folder else f"/workspace/datasets/{dataset_id}/images"
        entry = {"folder_path": folder, "caption_ext": ".txt", "cache_latents_to_disk": True}
        if resolution is not None:
            entry["resolution"] = resolution
        entries.append(entry)
    return entries


def _ai_toolkit_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "ai-toolkit")


def _nested(mapping: Any, *path: str) -> Any:
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def display_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Project adapter-owned native values for generic display."""
    native = _ai_toolkit_backend_override(run)
    config = native.get("config") if isinstance(native.get("config"), dict) else {}
    datasets = config.get("datasets") if isinstance(config.get("datasets"), list) else []
    first_dataset = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    return {
        "architecture": native.get("model_arch") or _nested(config, "model", "arch"),
        "rank": _nested(config, "network", "linear"),
        "alpha": _nested(config, "network", "linear_alpha"),
        "learning_rate": _nested(config, "train", "lr"),
        "scheduler": _nested(config, "train", "lr_scheduler"),
        "batch_size": _nested(config, "train", "batch_size"),
        "gradient_accumulation_steps": _nested(config, "train", "gradient_accumulation_steps"),
        "resolution": first_dataset.get("resolution") or native.get("resolution"),
        "optimizer": _nested(config, "train", "optimizer"),
        "precision": _nested(config, "train", "dtype"),
        "memory": {
            "gradient_checkpointing": _nested(config, "train", "gradient_checkpointing"),
            "low_vram": _nested(config, "model", "low_vram"),
            "quantize": _nested(config, "model", "quantize"),
            "quantize_te": _nested(config, "model", "quantize_te"),
        },
        "checkpoint": {
            "save_every_n_steps": _nested(config, "save", "save_every"),
            "keep_last": _nested(config, "save", "max_step_saves_to_keep"),
        },
    }


def requirements_ai_toolkit(run: dict[str, Any], download_estimate: dict[str, Any] | None = None, *, declared: bool = False) -> list[dict[str, Any]]:
    del download_estimate, declared
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    base = model.get("base")
    if not isinstance(base, str) or not base:
        return []
    revision = model.get("revision")
    if base.startswith(("/", "./", "../", "~")):
        acquisition = "local-path"
        identity: dict[str, Any] = {"kind": "path", "path": base}
        expected_format, observable = "backend-native-path", True
    else:
        acquisition = "backend"
        identity = {"kind": "huggingface-repository", "repo_id": base}
        expected_format, observable = "backend-native-repository", False
        if isinstance(revision, str) and revision:
            identity["revision"] = revision
    return [{"role": "base_model", "acquisition": acquisition, "identity": identity, "runtime_reference": base, "expected_format": expected_format, "measurement": {"scope": "backend-runtime", "status": "not-measured-by-kura"}, "pinning": artifact_pinning(identity, observable=observable)}]


def compile_ai_toolkit(run: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Write AI-Toolkit native YAML for configured training runs."""
    override = _ai_toolkit_backend_override(run)
    recipe = validated_recipe(run, required=override.get("command") is None)
    model = run.get("model", {})
    datasets = _datasets(run)
    native = override.get("config")
    if isinstance(native, dict):
        native_train = native.get("train")
        duplicated = sorted({"steps", "seed"} & set(native_train)) if isinstance(native_train, dict) else []
        if duplicated:
            raise ValueError("AI-Toolkit backend.config.config.train duplicates common recipe field(s): " + ", ".join(duplicated))
    config = {
        "job": "extension",
        "config": {
            "name": run["id"],
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"/workspace/runs/{run['id']}/outputs",
                "device": "cuda:0",
                "network": {"type": "lora"},
                "save": {},
                "datasets": _ai_toolkit_datasets(datasets, override.get("dataset_folder"), None),
                "train": {"steps": recipe.get("steps"), "train_unet": True, "train_text_encoder": False, "disable_sampling": True, "seed": recipe.get("seed")},
                "model": {"name_or_path": model.get("base"), "arch": override.get("model_arch"), "quantize": False, "quantize_te": False, "low_vram": False},
            }],
        },
    }
    process = config["config"]["process"][0]
    if "config" in override and not isinstance(native, dict):
        raise ValueError("backend.config.config must be a mapping for AI-Toolkit.")
    if isinstance(native, dict):
        for section, values in native.items():
            if section in process and isinstance(process[section], dict) and isinstance(values, dict):
                process[section].update(deepcopy(values))
            else:
                process[section] = deepcopy(values)
    atomic_write_yaml(destination.with_suffix(".yaml"), config)
    return command_ai_toolkit(run)


def command_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Return a container-native command spec, without executing it."""
    override = _ai_toolkit_backend_override(run)
    command = override.get("command")
    validated_recipe(run, required=command is None)
    if command is None:
        compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
        cwd = "/app/ai-toolkit" if compute.get("executor") == "runpod" else "/opt/ai-toolkit"
        return {"cwd": cwd, "argv": ["python", "run.py", f"/workspace/runs/{run['id']}/resolved/ai-toolkit.yaml"], "env": {}}
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
