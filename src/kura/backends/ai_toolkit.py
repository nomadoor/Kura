"""AI-Toolkit backend adapter."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from kura.backends.shared import _datasets, _script_command
from kura.container_scripts import script_source
from kura.fsio import atomic_write_yaml
from kura.provenance import artifact_pinning
from kura.run_envelope import backend_config, resume_intent, training_state_policy, validated_recipe


def training_state_contract_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    native = backend_config(run, "ai-toolkit")
    config = native.get("native_config") if isinstance(native.get("native_config"), dict) else {}
    train = config.get("train") if isinstance(config.get("train"), dict) else {}
    accumulation = native.get("gradient_accumulation_steps", train.get("gradient_accumulation_steps", 1))
    optimizer = str(native.get("optimizer_type", train.get("optimizer", "adamw8bit"))).lower()
    limitations = []
    if accumulation != 1:
        limitations.append("AI-Toolkit Resume initially requires gradient_accumulation_steps=1")
    if optimizer not in {"adamw", "adamw8bit"}:
        limitations.append("AI-Toolkit Resume initially requires AdamW or AdamW8bit optimizer state with a verified update counter")
    if limitations:
        return {
            "native_format": "ai-toolkit-weight-optimizer-pair",
            "required_files": ("model.safetensors", "optimizer.pt", "state-info.json"),
            "native_progress": "logical",
            "native_target": "logical",
            "state_step": {
                "path": "state-info.json", "field": "logical_step", "space": "logical",
                "schema_version": 1, "backend": "ai-toolkit",
                "digests": {"weight_sha256": "model.safetensors", "optimizer_sha256": "optimizer.pt"},
            },
            "capability": "unsupported",
            "restoration_contract": {
                "level": "unsupported",
                "restored": [],
                "not_restored": ["optimizer_update_step"],
                "limitations": limitations,
                "scheduler_behavior": "not available without a verified optimizer-update counter",
            },
        }
    return {
        "native_format": "ai-toolkit-weight-optimizer-pair",
        "required_files": ("model.safetensors", "optimizer.pt", "state-info.json"),
        "native_progress": "logical",
        "native_target": "logical",
        "state_step": {
            "path": "state-info.json", "field": "logical_step", "space": "logical",
            "schema_version": 1, "backend": "ai-toolkit",
            "digests": {"weight_sha256": "model.safetensors", "optimizer_sha256": "optimizer.pt"},
        },
        "capability": "partial_resume",
        "restoration_contract": {
            "level": "partial_resume",
            "restored": ["model", "optimizer", "global_step", "epoch"],
            "not_restored": ["scheduler", "rng", "exact_dataloader_position"],
            "scheduler_behavior": "reconstructed to the saved step; initial Resume execution limited to constant scheduler",
        },
    }


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
    config = native.get("native_config") if isinstance(native.get("native_config"), dict) else {}
    datasets = config.get("datasets") if isinstance(config.get("datasets"), list) else []
    first_dataset = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    return {
        "architecture": native.get("model_arch") or _nested(config, "model", "arch"),
        "rank": native.get("network_dim") or _nested(config, "network", "linear"),
        "alpha": native.get("network_alpha") or _nested(config, "network", "linear_alpha"),
        "learning_rate": native.get("learning_rate") or _nested(config, "train", "lr"),
        "scheduler": native.get("lr_scheduler") or _nested(config, "train", "lr_scheduler"),
        "batch_size": native.get("batch_size") or _nested(config, "train", "batch_size"),
        "gradient_accumulation_steps": native.get("gradient_accumulation_steps") or _nested(config, "train", "gradient_accumulation_steps"),
        "resolution": first_dataset.get("resolution") or native.get("resolution"),
        "optimizer": native.get("optimizer_type") or _nested(config, "train", "optimizer"),
        "precision": native.get("mixed_precision") or _nested(config, "train", "dtype"),
        "memory": {
            "gradient_checkpointing": native.get("gradient_checkpointing") if "gradient_checkpointing" in native else _nested(config, "train", "gradient_checkpointing"),
            "low_vram": native.get("low_vram") if "low_vram" in native else _nested(config, "model", "low_vram"),
            "quantize": native.get("quantize") if "quantize" in native else _nested(config, "model", "quantize"),
            "quantize_te": native.get("quantize_te") if "quantize_te" in native else _nested(config, "model", "quantize_te"),
        },
        "checkpoint": {
            "save_every_n_steps": native.get("save_every_n_steps") or _nested(config, "save", "save_every"),
            "keep_last": native.get("save_last_n_steps") or _nested(config, "save", "max_step_saves_to_keep"),
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
    native = override.get("native_config")
    if isinstance(native, dict):
        native_train = native.get("train")
        duplicated = sorted({"steps", "seed"} & set(native_train)) if isinstance(native_train, dict) else []
        if duplicated:
            raise ValueError("AI-Toolkit backend.config.native_config.train duplicates common recipe field(s): " + ", ".join(duplicated))
        protected: list[str] = []
        for key in ("name", "type", "training_folder", "device", "datasets"):
            if key in native:
                protected.append(key)
        native_model = native.get("model")
        if isinstance(native_model, dict) and "name_or_path" in native_model:
            protected.append("model.name_or_path")
        if "model_arch" in override and isinstance(native_model, dict) and "arch" in native_model:
            protected.append("model.arch (duplicates backend.config.model_arch)")
        if protected:
            raise ValueError(
                "AI-Toolkit backend.config.native_config overrides Kura-owned field(s): "
                + ", ".join(protected)
            )
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
    if "native_config" in override and not isinstance(native, dict):
        raise ValueError("backend.config.native_config must be a mapping for AI-Toolkit.")
    if isinstance(native, dict):
        for section, values in native.items():
            if section in process and isinstance(process[section], dict) and isinstance(values, dict):
                process[section].update(deepcopy(values))
            else:
                process[section] = deepcopy(values)
    ordinary = {
        "network": {"linear": "network_dim", "linear_alpha": "network_alpha"},
        "train": {
            "lr": "learning_rate", "lr_scheduler": "lr_scheduler", "optimizer": "optimizer_type",
            "dtype": "mixed_precision", "batch_size": "batch_size",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "gradient_checkpointing": "gradient_checkpointing",
        },
        "model": {"low_vram": "low_vram", "quantize": "quantize", "quantize_te": "quantize_te"},
        "save": {"save_every": "save_every_n_steps", "max_step_saves_to_keep": "save_last_n_steps"},
    }
    for section, fields in ordinary.items():
        for native_key, authored_key in fields.items():
            if authored_key in override:
                raw_section = native.get(section) if isinstance(native, dict) else None
                if isinstance(raw_section, dict) and native_key in raw_section:
                    raise ValueError(
                        f"AI-Toolkit backend.config.{authored_key} duplicates "
                        f"backend.config.native_config.{section}.{native_key}"
                    )
                process.setdefault(section, {})[native_key] = deepcopy(override[authored_key])
    if override.get("resolution") is not None:
        for dataset in process.get("datasets", []):
            if isinstance(dataset, dict):
                dataset["resolution"] = deepcopy(override["resolution"])
    policy = training_state_policy(run)
    continuation = resume_intent(run)
    if policy["enabled"]:
        if process.get("type") != "sd_trainer" or _nested(process, "network", "type") != "lora":
            raise ValueError("AI-Toolkit training-state capture initially supports only the standard sd_trainer LoRA process")
        ema_config = process.get("ema_config") if isinstance(process.get("ema_config"), dict) else {}
        if any(process.get(key) for key in ("embedding", "adapter", "decorator")) or ema_config.get("use_ema") or _nested(process, "train", "merge_network_on_save"):
            raise ValueError("AI-Toolkit training-state capture does not support embedding, adapter, decorator, EMA, or merge_network_on_save")
    if continuation is not None:
        scheduler = str(_nested(process, "train", "lr_scheduler") or "constant").lower()
        if scheduler != "constant":
            raise ValueError("AI-Toolkit State Resume initially requires the constant scheduler")
        process["train"]["steps"] = continuation["target_step"]
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
        config_path = f"/workspace/runs/{run['id']}/resolved/ai-toolkit.yaml"
        continuation = resume_intent(run)
        policy = training_state_policy(run)
        state_contract = training_state_contract_ai_toolkit(run)
        if not policy["enabled"] or state_contract.get("capability") == "unsupported":
            if continuation is not None:
                limitations = state_contract.get("restoration_contract", {}).get("limitations") or []
                detail = "; ".join(limitations) if limitations else "training-state capture is disabled"
                raise ValueError(f"AI-Toolkit Resume is unavailable: {detail}")
            return {"cwd": cwd, "argv": ["python", "run.py", config_path], "env": {}}
        spec: dict[str, Any] = {
            "config_path": config_path,
            "run_id": run["id"],
            "state_root": f"/workspace/runs/{run['id']}/outputs",
            "keep_generations": policy["keep_generations"],
        }
        runner = ["python", "-c", script_source("ai_toolkit_state.py"), json.dumps(spec, ensure_ascii=False, separators=(",", ":"))]
        if continuation is None:
            return {"cwd": cwd, "argv": runner, "env": {}}
        artifact_id = continuation["source"]["artifact_id"]
        spec["resume"] = {
            "payload": f"/workspace/artifacts/training-state/{artifact_id}/payload",
            "source_step": continuation["source"]["observed_step"],
        }
        runner[-1] = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
        verifier = [
            "python", "-c", script_source("training_state_verify.py"),
            f"/workspace/runs/{run['id']}/resolved/training-state-source.lock.json", "/workspace",
        ]
        return {"cwd": cwd, "argv": _script_command([verifier, runner], step_name="ai-toolkit"), "env": {}}
    combined = sorted(set(override) - {"command"})
    if combined:
        raise ValueError("AI-Toolkit explicit command cannot be combined with: " + ", ".join(combined))
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
