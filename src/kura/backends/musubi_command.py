"""Musubi command assembly and compile entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from kura.container_scripts import script_source
from kura.backends.common import _append_flag, _extra_args, _int_or_none, _musubi_backend_override, _require_paths, _script_command, _truthy
from kura.backends.musubi_datasets import _write_musubi_dataset_config
from kura.backends.musubi_models import _musubi_explicit_model_paths, _musubi_flux2_model_version, _musubi_lora_validation_command, _musubi_model_downloads, _musubi_model_lock, _musubi_model_paths, _musubi_model_validation_command, _musubi_output_compatibility, _unsupported_musubi_adapter_error
from kura.fsio import atomic_write_json, atomic_write_yaml


def compile_musubi_tuner(run: dict[str, Any], destination: Path, *, workspace: Path | None = None, strict: bool = False) -> None:
    """Write Musubi Tuner native dataset TOML and a readable command manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    _write_musubi_dataset_config(run, destination / "dataset.toml", workspace=workspace, strict=strict)
    command = command_musubi_tuner(run)
    atomic_write_json(destination / "command.json", command)
    atomic_write_yaml(destination / "model-bundle.lock.yaml", _musubi_model_lock(run))


def _musubi_prune_checkpoints_command(output_dir: str, output_name: str, before_step: Any) -> list[str] | None:
    if before_step in (None, False):
        return None
    try:
        threshold = int(before_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("Musubi Tuner prune_checkpoints_before_step must be an integer") from exc
    if threshold <= 0:
        return None
    return ["python", "-c", script_source("prune_checkpoints.py"), output_dir, output_name, str(threshold)]


def _musubi_start_commands(dataset_config: str, download_commands: list[list[str]]) -> list[list[str]]:
    return [["python", "-c", script_source("musubi_dataset_assert.py"), dataset_config], *download_commands]


def _musubi_save_precision(override: dict[str, Any]) -> str:
    value = override.get("save_precision", "bf16")
    if not isinstance(value, str) or value not in {"float", "fp32", "fp16", "bf16"}:
        raise ValueError("Musubi Tuner save_precision must be one of: float, fp32, fp16, bf16")
    return value


def _musubi_micro_batch(run: dict[str, Any], override: dict[str, Any]) -> int | None:
    dataset_config = override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general = dataset_config.get("general")
        if isinstance(general, dict):
            batch_size = _int_or_none(general.get("batch_size"))
            if batch_size is not None:
                return batch_size
        datasets = dataset_config.get("datasets")
        if isinstance(datasets, list):
            for item in datasets:
                if not isinstance(item, dict):
                    continue
                batch_size = _int_or_none(item.get("batch_size"))
                if batch_size is not None:
                    return batch_size
    return _int_or_none(run.get("params", {}).get("batch_size"))


def _musubi_max_resolution(run: dict[str, Any], override: dict[str, Any]) -> int | None:
    values: list[int] = []
    dataset_config = override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general = dataset_config.get("general")
        if isinstance(general, dict):
            resolution = general.get("resolution")
            if isinstance(resolution, list):
                values.extend(value for item in resolution if (value := _int_or_none(item)) is not None)
        datasets = dataset_config.get("datasets")
        if isinstance(datasets, list):
            for item in datasets:
                if not isinstance(item, dict):
                    continue
                for key in ("resolution", "control_resolution"):
                    resolution = item.get(key)
                    if isinstance(resolution, list):
                        values.extend(value for part in resolution if (value := _int_or_none(part)) is not None)
    params_resolution = run.get("params", {}).get("resolution")
    if isinstance(params_resolution, list):
        values.extend(value for item in params_resolution if (value := _int_or_none(item)) is not None)
    return max(values) if values else None


def _validate_musubi_resource_flags(run: dict[str, Any], override: dict[str, Any], architecture: str) -> None:
    extra_args = _extra_args(override)
    if "--block_swap_h2d_only" in extra_args and not (_truthy(override.get("gradient_checkpointing")) or "--gradient_checkpointing" in extra_args):
        raise ValueError("Musubi H2D-only block swap requires explicit gradient_checkpointing")
    if architecture not in ("flux2", "flux_2"):
        return
    model_version = str(override.get("model_version") or "").lower()
    if "9b" not in model_version:
        return
    gpu = str(run.get("compute", {}).get("gpu") or "")
    if "A40" not in gpu.upper():
        return
    micro_batch = _musubi_micro_batch(run, override)
    if micro_batch is None or micro_batch <= 1:
        max_resolution = _musubi_max_resolution(run, override)
        rank = _int_or_none(run.get("params", {}).get("rank")) or _int_or_none(override.get("network_dim"))
        checkpointing = _truthy(override.get("gradient_checkpointing")) or "--gradient_checkpointing" in extra_args
        if max_resolution is not None and max_resolution >= 1024 and (rank is None or rank >= 32) and not checkpointing:
            if _truthy(override.get("allow_a40_uncheckpointed_9b")):
                return
            raise ValueError(
                "Musubi FLUX.2 9B on NVIDIA A40 at 1024-class resolution/rank32 has been observed to OOM "
                "even with batch_size=1. Set backend_overrides.musubi-tuner.gradient_checkpointing: true, "
                "or set allow_a40_uncheckpointed_9b: true to accept the risk."
            )
        return
    if not _truthy(override.get("allow_a40_large_micro_batch")):
        raise ValueError(
            "Musubi FLUX.2 9B on NVIDIA A40 treats batch_size as GPU micro-batch; "
            f"batch_size={micro_batch} has been observed to OOM before step 1. "
            "Use batch_size: 1 and keep the intended effective batch with explicit "
            "--gradient_accumulation_steps, or set backend_overrides.musubi-tuner."
            "allow_a40_large_micro_batch: true to accept the risk."
        )


def _musubi_uses_sample_prompts(override: dict[str, Any], extra_args: list[str]) -> bool:
    if override.get("sample_prompts") or override.get("sample_every_n_steps") or override.get("sample_every_n_epochs"):
        return True
    return any(arg.startswith("--sample_") for arg in extra_args)


def _musubi_common_train_args(run: dict[str, Any], override: dict[str, Any], output_dir: str, output_name: str, *, default_lr: str = "1e-4") -> list[str]:
    params = run.get("params", {})
    args = [
        "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
        "--learning_rate", str(params.get("lr") or override.get("learning_rate") or default_lr),
        "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
        "--persistent_data_loader_workers",
        "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
    ]
    if params.get("alpha") is not None:
        args.extend(["--network_alpha", str(params["alpha"])])
    args.extend([
        "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
        "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
        "--save_precision", _musubi_save_precision(override),
        "--seed", str(params.get("seed") or override.get("seed") or 42),
        "--output_dir", output_dir,
        "--output_name", output_name,
    ])
    return args


def _validate_krea2_dataset_shape(override: dict[str, Any]) -> None:
    dataset_config = override.get("dataset_config")
    if not isinstance(dataset_config, dict):
        return
    datasets = dataset_config.get("datasets")
    if not isinstance(datasets, list):
        return
    forbidden = {"paired_jsonl", "control_directory", "control_resolution", "conditioning_data_dir"}
    for item in datasets:
        if isinstance(item, dict) and any(key in item for key in forbidden):
            raise ValueError("Musubi Krea2 supports plain image/caption datasets only; remove paired/control dataset fields")


def _backend_env(backend_name: str, override: dict[str, Any]) -> dict[str, str]:
    env = override.get("env", {})
    if env is None:
        return {}
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError(f"{backend_name} command env must be a string-to-string mapping")
    if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
        raise ValueError(f"{backend_name} command env must not contain secrets; use the process environment instead")
    return dict(env)


def command_musubi_tuner(run: dict[str, Any]) -> dict[str, Any]:
    """Return a Musubi Tuner command spec without executing it."""
    override = _musubi_backend_override(run)
    explicit = override.get("command")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("Musubi Tuner command must be a mapping")
        cwd, argv, env = explicit.get("cwd"), explicit.get("argv"), explicit.get("env", {})
        if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Musubi Tuner command must provide string cwd and argv values")
        if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
            raise ValueError("Musubi Tuner command env must be a string-to-string mapping")
        if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
            raise ValueError("Musubi Tuner command env must not contain secrets; use the process environment instead")
        return {"cwd": cwd, "argv": argv, "env": env}

    architecture = str(override.get("architecture") or override.get("model_arch") or "flux2").lower().replace("-", "_")
    _validate_musubi_resource_flags(run, override, architecture)
    params = run.get("params", {})
    explicit_paths = _musubi_explicit_model_paths(override)
    paths = _musubi_model_paths(run)
    download_commands, _ = _musubi_model_downloads(run, existing_paths=explicit_paths)
    dataset_config = f"/workspace/runs/{run['id']}/resolved/musubi/dataset.toml"
    output_dir = f"/workspace/runs/{run['id']}/outputs"
    output_name = run["id"]
    common = [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
    ]
    precache = bool(override.get("precache", True))
    if architecture in ("flux2", "flux_2"):
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        model_version = _musubi_flux2_model_version(run)
        if model_version == "dev" and _truthy(override.get("fp8_text_encoder")):
            raise ValueError("Musubi FLUX.2 dev uses Mistral 3 and does not support fp8_text_encoder")
        train_argv = [
            *common, "src/musubi_tuner/flux_2_train_network.py",
            "--model_version", model_version,
            "--dit", dit, "--vae", vae, "--text_encoder", text_encoder,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--timestep_sampling", str(override.get("timestep_sampling") or "flux2_shift"),
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(params.get("lr") or override.get("learning_rate") or "1e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_flux_2",
            "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
            "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
            "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(params.get("seed") or override.get("seed") or 42),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        if override.get("vae_dtype"):
            train_argv.extend(["--vae_dtype", str(override["vae_dtype"])])
        if params.get("alpha") is not None:
            train_argv.extend(["--network_alpha", str(params["alpha"])])
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/flux_2_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--model_version", model_version,
                "--skip_existing",
            ]
            if override.get("vae_dtype"):
                latent_argv.extend(["--vae_dtype", str(override["vae_dtype"])])
            text_argv = [
                "python", "src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder", text_encoder,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--model_version", model_version,
                "--skip_existing",
            ]
            if override.get("fp8_text_encoder"):
                text_argv.append("--fp8_text_encoder")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture == "wan":
        dit, vae, t5 = _require_paths(paths, ("dit", "vae", "t5"))
        task = str(override.get("task") or "t2v-1.3B")
        clip = paths.get("clip")
        one_frame = _truthy(override.get("one_frame"))
        wan21_i2v_tasks = {"i2v-14B", "i2v-14B-FC", "flf2v-14B"}
        if task in wan21_i2v_tasks and not clip:
            raise ValueError(f"Musubi Wan task {task} requires model_paths.clip or model_downloads.clip")
        if one_frame and task not in {"i2v-14B", "flf2v-14B"}:
            raise ValueError("Musubi Wan one_frame requires task i2v-14B or flf2v-14B")
        dit_high_noise = paths.get("dit_high_noise")
        if dit_high_noise and task not in {"t2v-A14B", "i2v-A14B"}:
            raise ValueError("Musubi Wan dit_high_noise is supported only for Wan 2.2 task t2v-A14B or i2v-A14B")
        if "timestep_boundary" in override and not dit_high_noise:
            raise ValueError("Musubi Wan timestep_boundary requires model_paths.dit_high_noise or model_downloads.dit_high_noise")
        train_argv = [
            *common, "src/musubi_tuner/wan_train_network.py",
            "--task", task,
            "--dit", dit, "--vae", vae, "--t5", t5,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(params.get("lr") or override.get("learning_rate") or "2e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_wan",
            "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
            "--timestep_sampling", str(override.get("timestep_sampling") or "shift"),
            "--discrete_flow_shift", str(override.get("discrete_flow_shift") or "3.0"),
            "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
            "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(params.get("seed") or override.get("seed") or 42),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("fp8_base")):
            train_argv.append("--fp8_base")
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if dit_high_noise:
            train_argv.extend(["--dit_high_noise", dit_high_noise])
            if "timestep_boundary" in override:
                train_argv.extend(["--timestep_boundary", str(override["timestep_boundary"])])
        if one_frame:
            train_argv.append("--one_frame")
        if params.get("alpha") is not None:
            train_argv.extend(["--network_alpha", str(params["alpha"])])
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/wan_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--skip_existing",
            ]
            text_argv = [
                "python", "src/musubi_tuner/wan_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--t5", t5,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--skip_existing",
            ]
            if "i2v" in task:
                latent_argv.append("--i2v")
            if clip:
                latent_argv.extend(["--clip", clip])
            if one_frame:
                latent_argv.append("--one_frame")
            if override.get("fp8_t5"):
                text_argv.append("--fp8_t5")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("krea2", "krea_2"):
        _validate_krea2_dataset_shape(override)
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        extra_args = _extra_args(override)
        train_argv = [
            *common, "src/musubi_tuner/krea2_train_network.py",
            "--dit", dit, "--vae", vae,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--timestep_sampling", str(override.get("timestep_sampling") or "krea2_shift"),
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(params.get("lr") or override.get("learning_rate") or "1e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_krea2",
            "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
            "--network_alpha", str(params.get("alpha") or override.get("network_alpha") or params.get("rank") or override.get("network_dim") or 32),
            "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
            "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(params.get("seed") or override.get("seed") or 42),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if _truthy(override.get("fp8_base")):
            train_argv.extend(["--fp8_base", "--fp8_scaled"])
        elif _truthy(override.get("fp8_scaled")):
            train_argv.append("--fp8_scaled")
        if _musubi_uses_sample_prompts(override, extra_args):
            train_argv.extend(["--text_encoder", text_encoder])
            if paths.get("turbo_dit"):
                train_argv.extend(["--turbo_dit", paths["turbo_dit"]])
        train_argv.extend(extra_args)
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            commands.extend([
                [
                    "python", "src/musubi_tuner/krea2_cache_latents.py",
                    "--dataset_config", dataset_config,
                    "--vae", vae,
                    "--skip_existing",
                ],
                [
                    "python", "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
                    "--dataset_config", dataset_config,
                    "--text_encoder", text_encoder,
                    "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                    "--skip_existing",
                ],
            ])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("qwen_image", "qwen"):
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        model_version = str(override.get("model_version") or "original")
        train_argv = [
            *common, "src/musubi_tuner/qwen_image_train_network.py",
            "--dit", dit, "--vae", vae, "--text_encoder", text_encoder,
            "--model_version", model_version,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--network_module", "networks.lora_qwen_image",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        _append_flag(train_argv, override, "gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        _append_flag(train_argv, override, "fp8_vl")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/qwen_image_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--model_version", model_version,
                "--skip_existing",
            ]
            text_argv = [
                "python", "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder", text_encoder,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--model_version", model_version,
                "--skip_existing",
            ]
            if _truthy(override.get("fp8_vl")):
                text_argv.append("--fp8_vl")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("zimage", "z_image"):
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        train_argv = [
            *common, "src/musubi_tuner/zimage_train_network.py",
            "--dit", dit, "--vae", vae, "--text_encoder", text_encoder,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora_zimage",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        _append_flag(train_argv, override, "gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        _append_flag(train_argv, override, "fp8_llm")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/zimage_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--skip_existing",
            ]
            text_argv = [
                "python", "src/musubi_tuner/zimage_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder", text_encoder,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--skip_existing",
            ]
            if _truthy(override.get("fp8_llm")):
                text_argv.append("--fp8_llm")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("flux_kontext", "flux1_kontext"):
        dit, vae, text_encoder1, text_encoder2 = _require_paths(paths, ("dit", "vae", "text_encoder1", "text_encoder2"))
        train_argv = [
            *common, "src/musubi_tuner/flux_kontext_train_network.py",
            "--dit", dit, "--vae", vae,
            "--text_encoder1", text_encoder1, "--text_encoder2", text_encoder2,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora_flux",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        _append_flag(train_argv, override, "gradient_checkpointing")
        if _truthy(override.get("fp8_base")) or _truthy(override.get("fp8")):
            train_argv.append("--fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        _append_flag(train_argv, override, "fp8_t5")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/flux_kontext_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--skip_existing",
            ]
            text_argv = [
                "python", "src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder1", text_encoder1,
                "--text_encoder2", text_encoder2,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--skip_existing",
            ]
            if _truthy(override.get("fp8_t5")):
                text_argv.append("--fp8_t5")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("ideogram4", "ideogram_4"):
        extra_args = _extra_args(override)
        uses_sampling = _musubi_uses_sample_prompts(override, extra_args)
        if precache or uses_sampling:
            dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        else:
            dit = _require_paths(paths, ("dit",))[0]
            vae = paths.get("vae")
            text_encoder = paths.get("text_encoder")
        train_argv = [
            *common, "src/musubi_tuner/ideogram4_train_network.py",
            "--dataset_config", dataset_config,
            "--dit", dit,
            "--network_module", "networks.lora_ideogram4",
            "--mixed_precision", "bf16",
            "--sdpa",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        if uses_sampling:
            train_argv.extend(["--vae", vae, "--text_encoder", text_encoder])
        if override.get("dit_dtype"):
            train_argv.extend(["--dit_dtype", str(override["dit_dtype"])])
        _append_flag(train_argv, override, "gradient_checkpointing")
        train_argv.extend(extra_args)
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/ideogram4_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--skip_existing",
            ]
            if override.get("vae_dtype"):
                latent_argv.extend(["--vae_dtype", str(override["vae_dtype"])])
            text_argv = [
                "python", "src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder", text_encoder,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--skip_existing",
            ]
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("hidream_o1", "hidream"):
        dit = _require_paths(paths, ("dit",))[0]
        model_type = str(override.get("model_type") or "full")
        task = str(override.get("task") or "t2i")
        train_argv = [
            *common, "src/musubi_tuner/hidream_o1_train_network.py",
            "--dit", dit,
            "--dataset_config", dataset_config,
            "--model_type", model_type,
            "--task", task,
            "--mixed_precision", "bf16",
            "--sdpa",
            "--timestep_sampling", str(override.get("timestep_sampling") or "uniform"),
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--network_module", "networks.lora_hidream_o1",
            *_musubi_common_train_args(run, override, output_dir, output_name, default_lr="4e-5"),
        ]
        if "noise_scale_start" in override:
            train_argv.extend(["--noise_scale_start", str(override["noise_scale_start"])])
        if "noise_scale_end" in override:
            train_argv.extend(["--noise_scale_end", str(override["noise_scale_end"])])
        if "noise_clip_std" in override:
            train_argv.extend(["--noise_clip_std", str(override["noise_clip_std"])])
        _append_flag(train_argv, override, "gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        _append_flag(train_argv, override, "flash_attn")
        _append_flag(train_argv, override, "skip_t2i_visual_dummy")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            pixel_argv = [
                "python", "src/musubi_tuner/hidream_o1_cache_pixel.py",
                "--dataset_config", dataset_config,
                "--batch_size", str(override.get("pixel_cache_batch_size") or 1),
            ]
            text_argv = [
                "python", "src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--model_type", model_type,
                "--batch_size", str(override.get("text_encoder_batch_size") or 16),
            ]
            if _truthy(override.get("fp8_te")):
                text_argv.extend(["--dit", dit, "--fp8_te"])
            commands.extend([pixel_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("hunyuan_video", "hunyuanvideo"):
        dit, vae, text_encoder1, text_encoder2 = _require_paths(paths, ("dit", "vae", "text_encoder1", "text_encoder2"))
        train_argv = [
            *common, "src/musubi_tuner/hv_train_network.py",
            "--dit", dit,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        _append_flag(train_argv, override, "gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--skip_existing",
            ]
            if override.get("vae_chunk_size"):
                latent_argv.extend(["--vae_chunk_size", str(override["vae_chunk_size"])])
            if _truthy(override.get("vae_tiling")):
                latent_argv.append("--vae_tiling")
            text_argv = [
                "python", "src/musubi_tuner/cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder1", text_encoder1,
                "--text_encoder2", text_encoder2,
                "--batch_size", str(override.get("text_encoder_batch_size") or 16),
                "--skip_existing",
            ]
            if _truthy(override.get("fp8_llm")):
                text_argv.append("--fp8_llm")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture == "hunyuan_video_1_5":
        task = str(override.get("task") or "t2v")
        required = ("dit", "vae", "text_encoder", "byt5", "image_encoder") if task == "i2v" else ("dit", "vae", "text_encoder", "byt5")
        required_paths = dict(zip(required, _require_paths(paths, required)))
        train_argv = [
            *common, "src/musubi_tuner/hv_1_5_train_network.py",
            "--dit", required_paths["dit"],
            "--vae", required_paths["vae"],
            "--text_encoder", required_paths["text_encoder"],
            "--byt5", required_paths["byt5"],
            "--dataset_config", dataset_config,
            "--task", task,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora_hv_1_5",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        if task == "i2v":
            train_argv.extend(["--image_encoder", required_paths["image_encoder"]])
        _append_flag(train_argv, override, "gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        _append_flag(train_argv, override, "fp8_vl")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/hv_1_5_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", required_paths["vae"],
                "--skip_existing",
            ]
            if task == "i2v":
                latent_argv.extend(["--i2v", "--image_encoder", required_paths["image_encoder"]])
            text_argv = [
                "python", "src/musubi_tuner/hv_1_5_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder", required_paths["text_encoder"],
                "--byt5", required_paths["byt5"],
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--skip_existing",
            ]
            if _truthy(override.get("fp8_vl")):
                text_argv.append("--fp8_vl")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("framepack", "frame_pack"):
        dit, vae, text_encoder1, text_encoder2, image_encoder = _require_paths(paths, ("dit", "vae", "text_encoder1", "text_encoder2", "image_encoder"))
        train_argv = [
            *common, "src/musubi_tuner/fpack_train_network.py",
            "--dit", dit, "--vae", vae,
            "--text_encoder1", text_encoder1, "--text_encoder2", text_encoder2,
            "--image_encoder", image_encoder,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora_framepack",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        if _truthy(override.get("f1")):
            train_argv.append("--f1")
        if _truthy(override.get("one_frame")):
            train_argv.append("--one_frame")
        _append_flag(train_argv, override, "gradient_checkpointing")
        if _truthy(override.get("fp8_base")) or _truthy(override.get("fp8")):
            train_argv.append("--fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        _append_flag(train_argv, override, "fp8_llm")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/fpack_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--image_encoder", image_encoder,
                "--skip_existing",
            ]
            if _truthy(override.get("f1")):
                latent_argv.append("--f1")
            if _truthy(override.get("one_frame")):
                latent_argv.append("--one_frame")
                if _truthy(override.get("one_frame_no_2x")):
                    latent_argv.append("--one_frame_no_2x")
                if _truthy(override.get("one_frame_no_4x")):
                    latent_argv.append("--one_frame_no_4x")
            if override.get("vae_chunk_size"):
                latent_argv.extend(["--vae_chunk_size", str(override["vae_chunk_size"])])
            text_argv = [
                "python", "src/musubi_tuner/fpack_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder1", text_encoder1,
                "--text_encoder2", text_encoder2,
                "--batch_size", str(override.get("text_encoder_batch_size") or 16),
                "--skip_existing",
            ]
            if _truthy(override.get("fp8_llm")):
                text_argv.append("--fp8_llm")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("kandinsky5", "kandinsky_5"):
        dit, vae, text_encoder_qwen, text_encoder_clip = _require_paths(paths, ("dit", "vae", "text_encoder_qwen", "text_encoder_clip"))
        task = str(override.get("task") or "k5-pro-t2v-5s-sd")
        train_argv = [
            *common, "src/musubi_tuner/kandinsky5_train_network.py",
            "--mixed_precision", "bf16",
            "--dataset_config", dataset_config,
            "--task", task,
            "--dit", dit,
            "--text_encoder_qwen", text_encoder_qwen,
            "--text_encoder_clip", text_encoder_clip,
            "--vae", vae,
            "--sdpa",
            "--network_module", "networks.lora_kandinsky",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        _append_flag(train_argv, override, "gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            text_argv = [
                    "python", "src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py",
                    "--dataset_config", dataset_config,
                    "--text_encoder_qwen", text_encoder_qwen,
                    "--text_encoder_clip", text_encoder_clip,
                    "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                    "--skip_existing",
            ]
            if _truthy(override.get("quantized_qwen")):
                text_argv.append("--quantized_qwen")
            commands.extend([
                text_argv,
                [
                    "python", "src/musubi_tuner/kandinsky5_cache_latents.py",
                    "--dataset_config", dataset_config,
                    "--vae", vae,
                    "--skip_existing",
                ],
            ])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    else:
        raise _unsupported_musubi_adapter_error(architecture)

    return {"cwd": "/opt/musubi-tuner", "argv": argv, "env": _backend_env("Musubi Tuner", override)}
