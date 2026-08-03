"""sd-scripts backend compiler and Tier 1 command selectors."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from kura.backends.sd_scripts_datasets import write_sd_scripts_dataset_config
from kura.backends.sd_scripts_models import explicit_model_paths, sd_scripts_architecture, sd_scripts_download_commands, sd_scripts_mode, sd_scripts_model_lock, sd_scripts_model_paths, sd_scripts_native
from kura.backends.shared import _append_flag, _extra_args as _shared_extra_args, _script_command, _truthy
from kura.container_scripts import script_source
from kura.fsio import atomic_write_json, atomic_write_yaml
from kura.run_envelope import validated_recipe


ENTRYPOINTS = {
    ("sd15", "lora"): "train_network.py",
    ("sdxl", "lora"): "sdxl_train_network.py",
    ("flux1", "lora"): "flux_train_network.py",
    ("anima", "lora"): "anima_train_network.py",
    ("anima", "controlnet_lllite"): "anima_train_control_net_lllite.py",
}
NETWORK_MODULES = {"sd15": "networks.lora", "sdxl": "networks.lora", "flux1": "networks.lora_flux", "anima": "networks.lora_anima"}
SECRET_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
OWNED_FLAGS = {
    "--dataset_config", "--pretrained_model_name_or_path", "--clip_l", "--t5xxl", "--ae", "--qwen3", "--vae",
    "--llm_adapter_path", "--t5_tokenizer_path", "--network_module", "--network_dim", "--network_alpha",
    "--learning_rate", "--optimizer_type", "--lr_scheduler", "--mixed_precision", "--max_train_steps", "--seed",
    "--output_dir", "--output_name", "--save_model_as", "--save_every_n_steps", "--save_last_n_steps",
    "--gradient_accumulation_steps", "--network_train_unet_only",
    "--gradient_checkpointing", "--fp8_base", "--blocks_to_swap",
    "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
    "--cpu_offload_checkpointing", "--unsloth_offload_checkpointing",
    "--deepspeed", "--fused_backward_pass",
    "--timestep_sampling", "--discrete_flow_shift", "--sigmoid_scale", "--guidance_scale", "--model_prediction_type",
    "--attn_mode",
    "--qwen_image_vae_2d", "--vae_chunk_size", "--unet_lr", "--text_encoder_lr1", "--text_encoder_lr2",
    "--cond_emb_dim", "--lllite_mlp_dim", "--lllite_target_layers", "--lllite_cond_dim", "--lllite_cond_resblocks",
    "--lllite_dropout", "--lllite_multiplier", "--lllite_cond_in_channels", "--lllite_use_aspp",
}
CONFIG_KEYS = {
    "architecture", "mode", "command", "model_paths", "model_downloads", "dataset_config", "output_name", "env", "extra_args",
    "network_dim", "network_alpha", "learning_rate", "optimizer_type", "lr_scheduler", "mixed_precision",
    "gradient_checkpointing", "gradient_accumulation_steps", "network_train_unet_only", "fp8_base", "blocks_to_swap",
    "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
    "disk_cache_estimate_gb", "cpu_offload_checkpointing", "unsloth_offload_checkpointing", "deepspeed", "fused_backward_pass",
    "save_every_n_steps", "save_last_n_steps", "timestep_sampling", "discrete_flow_shift", "sigmoid_scale", "guidance_scale",
    "model_prediction_type", "attn_mode", "qwen_image_vae_2d", "vae_chunk_size", "unet_lr", "text_encoder_lr1", "text_encoder_lr2",
    "cond_emb_dim", "lllite_mlp_dim", "lllite_target_layers", "lllite_cond_dim", "lllite_cond_resblocks", "lllite_dropout",
    "lllite_multiplier", "lllite_cond_in_channels", "lllite_use_aspp",
}
TIMESTEP_SAMPLING = {"sigma", "uniform", "sigmoid", "shift", "flux_shift"}
FLUX_PREDICTION_TYPES = {"raw", "additive", "sigma_scaled"}
ANIMA_ATTENTION_MODES = {"torch", "sdpa", "flash"}
BOOLEAN_CONFIG_KEYS = {
    "gradient_checkpointing", "network_train_unet_only", "fp8_base", "cache_latents", "cache_latents_to_disk",
    "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk", "qwen_image_vae_2d", "cpu_offload_checkpointing",
    "unsloth_offload_checkpointing", "deepspeed", "fused_backward_pass", "lllite_use_aspp",
}


def _extra_args(native: dict[str, Any]) -> list[str]:
    values = _shared_extra_args(native, backend_label="sd-scripts")
    duplicated = sorted({item.split("=", 1)[0] for item in values if item.startswith("--")} & OWNED_FLAGS)
    if duplicated:
        raise ValueError("sd-scripts extra_args duplicates adapter-owned flag(s): " + ", ".join(duplicated))
    return values


def _safe_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"sd-scripts {field} must contain only letters, digits, dot, underscore, or hyphen")
    return value


def _number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"sd-scripts {field} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"sd-scripts {field} must be positive")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"sd-scripts {field} must be a positive integer")
    return value


def _validate_config(native: dict[str, Any], architecture: str, mode: str) -> None:
    unknown = sorted(set(native) - CONFIG_KEYS)
    if unknown:
        raise ValueError("sd-scripts backend.config contains unsupported key(s): " + ", ".join(unknown))
    invalid_booleans = sorted(key for key in BOOLEAN_CONFIG_KEYS if key in native and not isinstance(native[key], bool))
    if invalid_booleans:
        raise ValueError("sd-scripts boolean config key(s) must be true or false: " + ", ".join(invalid_booleans))
    for key in ("learning_rate", "network_alpha", "unet_lr", "text_encoder_lr1", "text_encoder_lr2", "discrete_flow_shift", "sigmoid_scale", "guidance_scale"):
        if native.get(key) is not None:
            _number(native[key], field=key, positive=key != "guidance_scale")
    if native.get("network_dim") is not None:
        _positive_int(native["network_dim"], field="network_dim")
    if native.get("vae_chunk_size") is not None:
        _positive_int(native["vae_chunk_size"], field="vae_chunk_size")
    if native.get("timestep_sampling") is not None and (not isinstance(native["timestep_sampling"], str) or native["timestep_sampling"] not in TIMESTEP_SAMPLING):
        raise ValueError("sd-scripts timestep_sampling must be one of: " + ", ".join(sorted(TIMESTEP_SAMPLING)))
    if native.get("model_prediction_type") is not None and (not isinstance(native["model_prediction_type"], str) or native["model_prediction_type"] not in FLUX_PREDICTION_TYPES):
        raise ValueError("sd-scripts model_prediction_type must be one of: " + ", ".join(sorted(FLUX_PREDICTION_TYPES)))
    if native.get("attn_mode") is not None and (not isinstance(native["attn_mode"], str) or native["attn_mode"] not in ANIMA_ATTENTION_MODES):
        raise ValueError("sd-scripts attn_mode must be one of: " + ", ".join(sorted(ANIMA_ATTENTION_MODES)))
    flow_only = ("timestep_sampling", "discrete_flow_shift", "sigmoid_scale")
    if architecture not in {"flux1", "anima"} and any(native.get(key) is not None for key in flow_only):
        raise ValueError("sd-scripts flow-matching parameters are only valid for FLUX.1 or Anima")
    if architecture != "flux1" and any(native.get(key) is not None for key in ("guidance_scale", "model_prediction_type")):
        raise ValueError("sd-scripts guidance_scale/model_prediction_type are only valid for FLUX.1")
    if architecture != "anima" and any(native.get(key) is not None for key in ("qwen_image_vae_2d", "vae_chunk_size")):
        raise ValueError("sd-scripts qwen_image_vae_2d/vae_chunk_size are only valid for Anima")
    if architecture != "anima" and native.get("attn_mode") is not None:
        raise ValueError("sd-scripts attn_mode is only valid for Anima")
    if architecture != "sdxl" and any(native.get(key) is not None for key in ("unet_lr", "text_encoder_lr1", "text_encoder_lr2")):
        raise ValueError("sd-scripts SDXL component learning rates are only valid for SDXL")
    if mode == "controlnet_lllite" and native.get("network_dim") is not None:
        raise ValueError("sd-scripts Anima LLLite uses lllite_mlp_dim, not network_dim")
    for key in ("cond_emb_dim", "lllite_mlp_dim", "lllite_cond_dim", "lllite_cond_in_channels"):
        if native.get(key) is not None:
            _positive_int(native[key], field=key)
    if native.get("lllite_cond_resblocks") is not None:
        value = native["lllite_cond_resblocks"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("sd-scripts lllite_cond_resblocks must be a non-negative integer")
    if native.get("lllite_dropout") is not None:
        dropout = _number(native["lllite_dropout"], field="lllite_dropout")
        if not 0 <= dropout < 1:
            raise ValueError("sd-scripts lllite_dropout must be at least 0 and less than 1")
    if native.get("lllite_multiplier") is not None:
        _number(native["lllite_multiplier"], field="lllite_multiplier", positive=True)


def _explicit_command(run: dict[str, Any], command: Any) -> dict[str, Any]:
    validated_recipe(run, required=False)
    if not isinstance(command, dict):
        raise ValueError("sd-scripts backend.config.command must be a mapping")
    cwd, argv, env = command.get("cwd"), command.get("argv"), command.get("env", {})
    if not isinstance(cwd, str) or not cwd or not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("sd-scripts command must provide non-empty string cwd and argv values")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("sd-scripts command env must be a string-to-string mapping")
    if any(any(fragment in key.upper() for fragment in SECRET_FRAGMENTS) for key in env):
        raise ValueError("sd-scripts command env must not contain secrets; use the process environment instead")
    return {"cwd": cwd, "argv": list(argv), "env": dict(env)}


def _validate_native_env(native: dict[str, Any]) -> dict[str, str]:
    env = native.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("sd-scripts env must be a string-to-string mapping")
    if any(any(fragment in key.upper() for fragment in SECRET_FRAGMENTS) for key in env):
        raise ValueError("sd-scripts env must not contain secrets; use the process environment instead")
    return dict(env)


def _validate_selector(native: dict[str, Any], architecture: str, mode: str) -> None:
    if (architecture, mode) not in ENTRYPOINTS:
        raise ValueError(
            f"unsupported Kura built-in sd-scripts adapter: architecture={architecture}, mode={mode}. "
            "sd-scripts may support this path upstream, but Kura does not generate it yet; use backend.config.command for a complete reviewed command."
        )
    _validate_config(native, architecture, mode)
    if mode == "controlnet_lllite":
        if native.get("fp8_base") is True:
            raise ValueError("sd-scripts Anima LLLite does not support fp8_base")
        incompatible = [name for name in ("blocks_to_swap", "cpu_offload_checkpointing", "unsloth_offload_checkpointing", "deepspeed", "fused_backward_pass") if native.get(name) not in (None, False, 0, "0", "false")]
        if incompatible:
            raise ValueError("sd-scripts Anima LLLite does not support: " + ", ".join(incompatible))
        channels = native.get("lllite_cond_in_channels", 3)
        if isinstance(channels, bool) or not isinstance(channels, int) or channels != 3:
            raise ValueError("sd-scripts Tier 1 Anima LLLite is image-only and requires lllite_cond_in_channels=3")
        target = native.get("lllite_target_layers", "self_attn_q")
        allowed = {"self_attn_q", "self_attn_qkv", "self_attn_qkv_cross_q", "self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre", "mlp_fc1_pre"}
        if not isinstance(target, str) or any(part not in allowed for part in target.split(",")):
            raise ValueError("sd-scripts Anima LLLite lllite_target_layers contains an unsupported selector")
    if architecture == "anima" and mode == "lora":
        if _truthy(native.get("fp8_base")):
            raise ValueError("sd-scripts Anima LoRA does not support fp8_base")
        if (_truthy(native.get("cache_text_encoder_outputs")) or _truthy(native.get("cache_text_encoder_outputs_to_disk"))) and not _truthy(native.get("network_train_unet_only")):
            raise ValueError("sd-scripts Anima LoRA text-encoder caching requires network_train_unet_only=true")
        incompatible_pairs = (("blocks_to_swap", "cpu_offload_checkpointing"), ("blocks_to_swap", "unsloth_offload_checkpointing"), ("cpu_offload_checkpointing", "unsloth_offload_checkpointing"))
        for left, right in incompatible_pairs:
            if native.get(left) not in (None, False, 0) and _truthy(native.get(right)):
                raise ValueError(f"sd-scripts Anima LoRA cannot combine {left} with {right}")
    if architecture == "flux1":
        blocks = native.get("blocks_to_swap")
        if blocks not in (None, False, 0):
            if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
                raise ValueError("sd-scripts FLUX.1 blocks_to_swap must be a positive integer")
            if _truthy(native.get("cpu_offload_checkpointing")):
                raise ValueError("sd-scripts FLUX.1 cannot combine blocks_to_swap with cpu_offload_checkpointing")
    if mode != "controlnet_lllite" and any(_truthy(native.get(key)) for key in ("deepspeed", "fused_backward_pass")):
        raise ValueError("sd-scripts built-in LoRA selectors do not own deepspeed/fused_backward_pass; use a reviewed explicit command")
    if architecture in {"sdxl", "flux1"} and (_truthy(native.get("cache_text_encoder_outputs")) or _truthy(native.get("cache_text_encoder_outputs_to_disk"))) and not _truthy(native.get("network_train_unet_only")):
        raise ValueError(f"sd-scripts {architecture} text-encoder caching requires network_train_unet_only=true")


def _base_training_args(run: dict[str, Any], native: dict[str, Any], paths: dict[str, str], output_dir: str, output_name: str) -> list[str]:
    architecture, mode = sd_scripts_architecture(run), sd_scripts_mode(run)
    recipe = validated_recipe(run, required=True)
    args = [
        ENTRYPOINTS[(architecture, mode)],
        "--dataset_config", f"/workspace/runs/{run['id']}/resolved/sd-scripts/dataset.toml",
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--save_model_as", "safetensors",
        "--max_train_steps", str(recipe["steps"]),
        "--seed", str(recipe["seed"]),
        "--learning_rate", str(native.get("learning_rate", 1e-4)),
        "--optimizer_type", str(native.get("optimizer_type", "AdamW8bit")),
        "--lr_scheduler", str(native.get("lr_scheduler", "constant")),
        "--mixed_precision", str(native.get("mixed_precision", "bf16")),
    ]
    if architecture in ("sd15", "sdxl"):
        args.extend(["--pretrained_model_name_or_path", paths["base"], "--network_module", NETWORK_MODULES[architecture]])
        if paths.get("vae"):
            args.extend(["--vae", paths["vae"]])
    elif architecture == "flux1":
        args.extend(["--pretrained_model_name_or_path", paths["dit"], "--clip_l", paths["clip_l"], "--t5xxl", paths["t5xxl"], "--ae", paths["ae"], "--network_module", NETWORK_MODULES[architecture]])
    elif architecture == "anima":
        args.extend(["--pretrained_model_name_or_path", paths["dit"], "--qwen3", paths["qwen3"], "--vae", paths["vae"]])
        for role, flag in (("llm_adapter", "--llm_adapter_path"), ("t5_tokenizer", "--t5_tokenizer_path")):
            if paths.get(role):
                args.extend([flag, paths[role]])
        if mode == "lora":
            args.extend(["--network_module", NETWORK_MODULES[architecture]])
    if mode == "lora":
        args.extend(["--network_dim", str(native.get("network_dim", 16)), "--network_alpha", str(native.get("network_alpha", 1.0))])
    if architecture == "sdxl":
        for key in ("unet_lr", "text_encoder_lr1", "text_encoder_lr2"):
            if native.get(key) is not None:
                args.extend([f"--{key}", str(native[key])])
    if _truthy(native.get("gradient_checkpointing")):
        args.append("--gradient_checkpointing")
    accumulation = native.get("gradient_accumulation_steps", 1)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("sd-scripts gradient_accumulation_steps must be a positive integer")
    args.extend(["--gradient_accumulation_steps", str(accumulation)])
    if mode == "lora" and _truthy(native.get("network_train_unet_only")):
        args.append("--network_train_unet_only")
    if architecture != "anima" and _truthy(native.get("fp8_base")):
        args.append("--fp8_base")
    for key in ("save_every_n_steps", "save_last_n_steps"):
        value = native.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sd-scripts {key} must be a positive integer")
            args.extend([f"--{key}", str(value)])
    if _truthy(native.get("cache_latents")) or _truthy(native.get("cache_latents_to_disk")):
        args.append("--cache_latents")
    if _truthy(native.get("cache_latents_to_disk")):
        args.append("--cache_latents_to_disk")
    if _truthy(native.get("cache_text_encoder_outputs")) or _truthy(native.get("cache_text_encoder_outputs_to_disk")):
        args.append("--cache_text_encoder_outputs")
    if _truthy(native.get("cache_text_encoder_outputs_to_disk")):
        args.append("--cache_text_encoder_outputs_to_disk")
    if architecture in ("flux1", "anima") and native.get("blocks_to_swap") not in (None, False, 0):
        args.extend(["--blocks_to_swap", str(native["blocks_to_swap"])])
    if architecture == "flux1":
        args.extend([
            "--timestep_sampling", str(native.get("timestep_sampling", "flux_shift")),
            "--guidance_scale", str(native.get("guidance_scale", 1.0)),
            "--model_prediction_type", str(native.get("model_prediction_type", "raw")),
        ])
    elif architecture == "anima":
        default_sampling = "shift" if mode == "controlnet_lllite" else "sigmoid"
        default_shift = 3.0 if mode == "controlnet_lllite" else 1.0
        args.extend([
            "--timestep_sampling", str(native.get("timestep_sampling", default_sampling)),
            "--discrete_flow_shift", str(native.get("discrete_flow_shift", default_shift)),
        ])
        if mode == "controlnet_lllite" or native.get("attn_mode") is not None:
            args.extend(["--attn_mode", str(native.get("attn_mode", "sdpa"))])
    if architecture in {"flux1", "anima"} and native.get("sigmoid_scale") is not None:
        args.extend(["--sigmoid_scale", str(native["sigmoid_scale"])])
    if architecture == "flux1" and native.get("discrete_flow_shift") is not None:
        args.extend(["--discrete_flow_shift", str(native["discrete_flow_shift"])])
    if architecture == "anima" and _truthy(native.get("qwen_image_vae_2d")):
        args.append("--qwen_image_vae_2d")
    if architecture == "anima" and native.get("vae_chunk_size") is not None:
        args.extend(["--vae_chunk_size", str(native["vae_chunk_size"])])
    for key in ("cpu_offload_checkpointing", "unsloth_offload_checkpointing"):
        _append_flag(args, native, key)
    if mode == "controlnet_lllite":
        scalar_flags = {
            "cond_emb_dim": 32, "lllite_mlp_dim": 64, "lllite_target_layers": "self_attn_q",
            "lllite_cond_dim": 64, "lllite_cond_resblocks": 1, "lllite_dropout": None,
            "lllite_multiplier": 1.0, "lllite_cond_in_channels": 3,
        }
        for key, default in scalar_flags.items():
            value = native.get(key, default)
            if value is not None:
                args.extend([f"--{key}", str(value)])
        _append_flag(args, native, "lllite_use_aspp")
    args.extend(_extra_args(native))
    return args


def _validation_command(*, models: list[dict[str, str]] | None = None, pattern: str | None = None, kind: str | None = None) -> list[str]:
    spec: dict[str, Any] = {"models": models or []}
    if pattern and kind:
        spec["output"] = {"pattern": pattern, "kind": kind}
    return ["python", "-c", script_source("sd_scripts_validate.py"), json.dumps(spec, ensure_ascii=False)]


def _anima_conversion_command(run_id: str, output_name: str, training_argv: list[str]) -> list[str]:
    run_root = f"/workspace/runs/{run_id}"
    spec = {
        "native_dir": f"{run_root}/cache/sd-scripts/native-output",
        "staging_dir": f"{run_root}/cache/sd-scripts/converted-output",
        "output_dir": f"{run_root}/outputs",
        "recovery_dir": f"{run_root}/recovery/sd-scripts/anima-native",
        "output_name": output_name,
        "converter": "networks/convert_anima_lora_to_comfy.py",
        "train_argv": training_argv,
    }
    return ["python", "-c", script_source("sd_scripts_publish_anima.py"), json.dumps(spec, ensure_ascii=False)]


def command_sd_scripts(run: dict[str, Any]) -> dict[str, Any]:
    native = sd_scripts_native(run)
    if native.get("command") is not None:
        unknown = sorted(set(native) - {"command"})
        if unknown:
            raise ValueError("sd-scripts explicit command cannot be combined with: " + ", ".join(unknown))
        return _explicit_command(run, native["command"])
    architecture, mode = sd_scripts_architecture(run), sd_scripts_mode(run)
    _validate_selector(native, architecture, mode)
    paths = sd_scripts_model_paths(run)
    explicit = explicit_model_paths(run)
    download_commands, downloaded = sd_scripts_download_commands(run, explicit)
    paths.update(downloaded)
    output_name = _safe_name(native.get("output_name", run["id"]), field="output_name")
    final_output = f"/workspace/runs/{run['id']}/outputs"
    train_output = f"/workspace/runs/{run['id']}/cache/sd-scripts/native-output" if architecture == "anima" and mode == "lora" else final_output
    model_items = [{"role": role, "path": path} for role, path in sorted(paths.items())]
    training_argv = ["accelerate", "launch", "--num_cpu_threads_per_process", "1", *_base_training_args(run, native, paths, train_output, output_name)]
    commands: list[list[str]] = [
        ["python", "-c", script_source("sd_scripts_dataset_stage.py"), f"/workspace/runs/{run['id']}/resolved/sd-scripts/dataset-stage.lock.json", "/workspace"],
        *download_commands,
        _validation_command(models=model_items),
    ]
    if architecture == "anima" and mode == "lora":
        commands.append(_anima_conversion_command(run["id"], output_name, training_argv))
    else:
        commands.append(training_argv)
        kind = "anima-lllite" if mode == "controlnet_lllite" else "lora"
        commands.append(_validation_command(pattern=f"{final_output}/{output_name}.safetensors", kind=kind))
    return {"cwd": "/opt/sd-scripts", "argv": _script_command(commands, step_name="sd-scripts"), "env": _validate_native_env(native)}


def compile_sd_scripts(run: dict[str, Any], destination: Path, *, workspace: Path | None = None, strict: bool = False) -> dict[str, Any]:
    native = sd_scripts_native(run)
    destination.mkdir(parents=True, exist_ok=True)
    if native.get("command") is not None:
        command = command_sd_scripts(run)
        atomic_write_yaml(destination / "model-bundle.lock.yaml", {"schema_version": 1, "backend": "sd-scripts", "ownership": "explicit-command"})
        atomic_write_json(destination / "command.json", command)
        return command
    write_sd_scripts_dataset_config(run, destination / "dataset.toml", workspace=workspace, strict=strict)
    atomic_write_yaml(destination / "model-bundle.lock.yaml", sd_scripts_model_lock(run))
    command = command_sd_scripts(run)
    atomic_write_json(destination / "command.json", command)
    return command


def display_sd_scripts(run: dict[str, Any]) -> dict[str, Any]:
    native = sd_scripts_native(run)
    if native.get("command") is not None:
        unknown = sorted(set(native) - {"command"})
        if unknown:
            raise ValueError("sd-scripts explicit command cannot be combined with: " + ", ".join(unknown))
        return {"architecture": "explicit-command", "mode": "explicit-command"}
    if not isinstance(native.get("architecture"), str) or not native["architecture"].strip():
        # Monitoring may encounter a hand-authored or historical run before it
        # was compiled. Compilation remains the strict validation boundary.
        return {"architecture": native.get("architecture"), "mode": native.get("mode", "lora")}
    architecture, mode = sd_scripts_architecture(run), sd_scripts_mode(run)
    config = native.get("dataset_config") if isinstance(native.get("dataset_config"), dict) else {}
    general = config.get("general") if isinstance(config.get("general"), dict) else {}
    datasets = config.get("datasets") if isinstance(config.get("datasets"), list) else []
    first = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    flow: dict[str, Any] = {}
    if architecture == "flux1":
        flow = {
            "timestep_sampling": native.get("timestep_sampling", "flux_shift"),
            "discrete_flow_shift": native.get("discrete_flow_shift"),
            "sigmoid_scale": native.get("sigmoid_scale"),
            "guidance_scale": native.get("guidance_scale", 1.0),
            "model_prediction_type": native.get("model_prediction_type", "raw"),
        }
    elif architecture == "anima":
        flow = {
            "timestep_sampling": native.get("timestep_sampling", "shift" if mode == "controlnet_lllite" else "sigmoid"),
            "discrete_flow_shift": native.get("discrete_flow_shift", 3.0 if mode == "controlnet_lllite" else 1.0),
            "sigmoid_scale": native.get("sigmoid_scale"),
        }
    return {
        "architecture": architecture, "mode": mode,
        "rank": native.get("network_dim") if mode == "lora" else None, "alpha": native.get("network_alpha") if mode == "lora" else None,
        "learning_rate": native.get("learning_rate"), "scheduler": native.get("lr_scheduler"),
        "batch_size": first.get("batch_size"), "gradient_accumulation_steps": native.get("gradient_accumulation_steps", 1), "resolution": first.get("resolution") or general.get("resolution"),
        "optimizer": native.get("optimizer_type"), "precision": native.get("mixed_precision"),
        "flow_matching": flow,
        "component_learning_rates": {key: native.get(key) for key in ("unet_lr", "text_encoder_lr1", "text_encoder_lr2")},
        "memory": {
            **{key: native.get(key) for key in ("gradient_checkpointing", "fp8_base", "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk", "blocks_to_swap", "qwen_image_vae_2d", "vae_chunk_size")},
            "attn_mode": native.get("attn_mode", "sdpa") if architecture == "anima" and mode == "controlnet_lllite" else native.get("attn_mode"),
        },
        "checkpoint": {"save_every_n_steps": native.get("save_every_n_steps"), "retention_window_steps": native.get("save_last_n_steps")},
    }
