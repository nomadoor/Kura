"""Musubi command assembly and compile entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from kura.container_scripts import script_source
from kura.dataset_handoff import load_frozen_dataset_projection
from kura.backends.common import _musubi_architecture, _musubi_backend_override, _require_paths, musubi_native_dataset_architecture
from kura.backends.shared import _append_flag, _extra_args as _shared_extra_args, _int_or_none, _reject_owned_extra_args, _script_command as _shared_script_command, _truthy, state_runner_argv, write_state_runner
from kura.backends.musubi_datasets import (
    MUSUBI_AUDIO_SUFFIXES,
    FRAMEPACK_LATENT_WINDOW_SIZE,
    MUSUBI_IMAGE_SUFFIXES,
    MUSUBI_VIDEO_SUFFIXES,
    _musubi_h3_effective_task,
    _write_musubi_dataset_config,
)
from kura.backends.musubi_models import _musubi_flux2_model_version, _musubi_lora_validation_command, _musubi_model_downloads, _musubi_model_lock, _musubi_model_paths, _musubi_model_validation_command, _musubi_model_version, _musubi_output_compatibility, _unsupported_musubi_adapter_error
from kura.backends.musubi_native_selectors import musubi_native_task, wan_native_selector
from kura.fsio import atomic_write_yaml
from kura.media_types import frozen_suffixes
from kura.secrets import is_secret_name
from kura.training_artifacts import checkpoint_save_cadence, managed_state_save_args, resume_steps, run_output_name, training_state_managed, training_state_payload
from kura.run_envelope import resume_intent, validated_recipe


def training_state_contract_musubi(run: dict[str, Any]) -> dict[str, Any]:
    del run
    return {
        "native_format": "accelerate-state-directory",
        # Kura's state runner writes kura-state-info.json after each complete save. An artifact
        # published before the marker is still read by its own inventory, though a Resume from it
        # is refused when its adapter or image identity no longer matches the current ones.
        "required_files": ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "kura-state-info.json"),
        # The pinned image's patch 0001 continues the logical step on Resume, so the trainer's
        # step counter, progress, and target are logical; a lock compiled before it froze
        # `process_local` and is read that way.
        "native_progress": "logical",
        "native_target": "logical",
        # A run compiled before Kura launched Musubi through the state runner has no runner here
        # and no marker; `compiled_training_state_contract` reads its states as before.
        "state_runner": "musubi/state-runner.py",
        "state_step": {
            "path": "kura-state-info.json", "field": "logical_step", "space": "logical",
            "schema_version": 1, "backend": "musubi-tuner",
            "digests": {"optimizer_sha256": "optimizer.bin", "scheduler_sha256": "scheduler.bin"},
        },
        # Every Musubi command names the recipe's steps as its cadence when the run sets none.
        "unset_save_cadence": "recipe_steps",
        "capability": "best_effort_resume",
        "restoration_contract": {
            "level": "best_effort_resume",
            "restored": ["model", "optimizer", "scheduler", "rng", "scaler_when_present", "supported_sampler_state", "application_global_step"],
            "not_restored": ["epoch", "exact_dataloader_position"],
            "scheduler_behavior": "restored; Resume execution limited to constant scheduler",
        },
    }


# Flags Kura owns for every architecture. Flags Kura emits only for some
# architectures (for example --timestep_sampling) are refused by
# _reject_emitted_duplicates only where that architecture's command carries them.
MUSUBI_OWNED_FLAGS = frozenset({
    "--audio_vae", "--base_weights", "--batch_size", "--block_swap_h2d_only",
    "--block_swap_ring_size", "--blocks_to_swap", "--byt5", "--cache_seed",
    "--clip", "--convrot_int8", "--convrot_int8_bwd", "--dataset_config", "--dit",
    "--dit_dtype", "--dit_high_noise", "--f1", "--fp8", "--fp8_base", "--fp8_llm",
    "--fp8_scaled", "--fp8_t5", "--fp8_te", "--fp8_text_encoder", "--fp8_vl",
    "--gradient_accumulation_steps", "--gradient_checkpointing",
    "--gradient_checkpointing_cpu_offload", "--h3_guidance_loss_scale",
    "--h3_guidance_loss_sigma_min", "--h3_guidance_loss_uncond_cache",
    "--h3_teacher_conditions", "--h3_teacher_matching", "--i2v", "--image_encoder",
    "--latent_window_size", "--learning_rate", "--lr_scheduler",
    "--max_data_loader_n_workers", "--max_train_steps", "--mixed_precision",
    "--model_type", "--model_version", "--network_alpha", "--network_dim",
    "--network_module", "--noise_clip_std", "--noise_scale_end",
    "--noise_scale_start", "--one_frame", "--one_frame_no_2x", "--one_frame_no_4x",
    "--optimizer_type", "--output_dir", "--output_name",
    "--persistent_data_loader_workers", "--quantized_qwen", "--resume",
    "--save_every_n_steps", "--save_last_n_steps", "--save_last_n_steps_state",
    "--save_precision", "--save_state", "--save_state_on_train_end", "--sdpa",
    "--seed", "--skip_existing", "--t5", "--task", "--teacher_conditions",
    "--text_cache_dtype", "--text_encoder", "--text_encoder1", "--text_encoder2",
    "--text_encoder_blocks_to_swap", "--text_encoder_clip", "--text_encoder_qwen",
    "--turbo_dit", "--uncond_output", "--use_pinned_memory_for_block_swap", "--vae",
    "--vae_chunk_size", "--vae_dtype", "--vae_tiling", "--video_only",
    "--video_vae",
})


def _extra_args(override: dict[str, Any]) -> list[str]:
    values = _shared_extra_args(override, backend_label="Musubi Tuner")
    _reject_owned_extra_args(
        values, owned_flags=MUSUBI_OWNED_FLAGS, backend_label="Musubi Tuner",
    )
    return values


def _extra_arg_value(arguments: list[str], flag: str) -> str | None:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"Musubi backend.config.extra_args {flag} requires a value")
            values.append(arguments[index + 1])
        elif argument.startswith(flag + "="):
            value = argument.split("=", 1)[1]
            if not value:
                raise ValueError(f"Musubi backend.config.extra_args {flag} requires a value")
            values.append(value)
    if len(values) > 1:
        raise ValueError(f"Musubi backend.config.extra_args duplicates {flag}")
    return values[0] if values else None


def _reject_emitted_duplicates(train: list[str], extra_args: list[str]) -> None:
    """Refuse an extra argument that repeats or abbreviates a flag Kura emitted for this architecture."""
    extra_flags = [argument.split("=", 1)[0] for argument in extra_args if argument.startswith("--")]
    emitted = [argument.split("=", 1)[0] for argument in train if argument.startswith("--")]
    for flag in extra_flags:
        if flag in emitted:
            emitted.remove(flag)
    duplicates = sorted({
        candidate for candidate in extra_flags
        if any(flag.startswith(candidate) for flag in emitted)
    })
    if duplicates:
        raise ValueError("Musubi Tuner extra_args duplicates adapter-owned flag(s): " + ", ".join(duplicates))


def _script_command(commands: list[list[str]], override: dict[str, Any], run: dict[str, Any] | None = None) -> list[str]:
    training_commands = [command for command in commands if "--max_train_steps" in command]
    if len(training_commands) != 1:
        raise ValueError(
            "Musubi built-in adapter must produce exactly one training command with --max_train_steps; "
            f"found {len(training_commands)}"
        )
    train = training_commands[0]
    if run is None:
        raise ValueError("Musubi training command assembly requires the frozen run envelope")
    continuation = resume_intent(run)
    steps = resume_steps(run, contract=training_state_contract_musubi(run))
    if steps is not None:
        target_index = train.index("--max_train_steps") + 1
        train[target_index] = str(steps["native_end"])
    if training_state_managed(run, training_state_contract_musubi(run)):
        configured_cadence = override.get("save_every_n_steps")
        if configured_cadence is not None and (
            isinstance(configured_cadence, bool)
            or not isinstance(configured_cadence, int)
            or configured_cadence <= 0
        ):
            raise ValueError("Musubi backend.config.save_every_n_steps must be a positive integer")
        save_args = managed_state_save_args(run, configured_cadence, _extra_args(override), contract=training_state_contract_musubi(run))
        if save_args[0] == "--save_every_n_steps":
            # Every Musubi command already names its default cadence; one Kura sets takes its place.
            cadence = save_args[1]
            save_args = save_args[2:]
            try:
                cadence_index = train.index("--save_every_n_steps") + 1
            except ValueError:
                train.extend(["--save_every_n_steps", cadence])
            else:
                train[cadence_index] = cadence
        train.extend(save_args)
    if continuation is not None:
        extra_scheduler = _extra_arg_value(_extra_args(override), "--lr_scheduler")
        if extra_scheduler is not None and override.get("lr_scheduler") is not None:
            raise ValueError("Musubi backend.config.lr_scheduler duplicates backend.config.extra_args --lr_scheduler")
        scheduler = str(override.get("lr_scheduler") or extra_scheduler or "constant").lower()
        if scheduler != "constant":
            raise ValueError("Musubi State Resume initially requires the constant scheduler")
        artifact_id = continuation["source"]["artifact_id"]
        train.extend(["--resume", training_state_payload(artifact_id)])
        commands.insert(
            commands.index(train),
            [
                "python",
                "-c",
                script_source("training_state_verify.py"),
                f"/workspace/runs/{run['id']}/resolved/training-state-source.lock.json",
                "/workspace",
            ],
        )
    for key, flag in (
        ("lr_scheduler", "--lr_scheduler"),
        ("gradient_accumulation_steps", "--gradient_accumulation_steps"),
        ("blocks_to_swap", "--blocks_to_swap"),
        ("block_swap_ring_size", "--block_swap_ring_size"),
    ):
        if override.get(key) is None:
            continue
        if any(arg == flag or arg.startswith(flag + "=") for arg in train):
            raise ValueError(f"Musubi backend.config.{key} duplicates backend.config.extra_args {flag}")
        train.extend([flag, str(override[key])])
    for key, flag in (
        ("block_swap_h2d_only", "--block_swap_h2d_only"),
        ("use_pinned_memory_for_block_swap", "--use_pinned_memory_for_block_swap"),
    ):
        if not _truthy(override.get(key)):
            continue
        if any(arg == flag or arg.startswith(flag + "=") for arg in train):
            raise ValueError(f"Musubi backend.config.{key} duplicates backend.config.extra_args {flag}")
        train.append(flag)
    _reject_emitted_duplicates(train, _extra_args(override))
    if training_state_managed(run, training_state_contract_musubi(run)):
        entrypoint = next(index for index, argument in enumerate(train) if argument.endswith(".py"))
        train[entrypoint:] = state_runner_argv(
            "musubi-tuner", f"/workspace/runs/{run['id']}/resolved/musubi", train[entrypoint], train[entrypoint + 1:],
        )
    return _shared_script_command(commands, step_name="musubi")


def compile_musubi_tuner(run: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Write Musubi Tuner native dataset TOML and a readable command manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    explicit_command = _musubi_backend_override(run).get("command") is not None
    command = command_musubi_tuner(run)
    if not explicit_command:
        projection = load_frozen_dataset_projection(
            destination.parent,
            backend="musubi-tuner",
            dataset_ids=[str(item.get("id")) for item in run.get("datasets", [])],
        )
        assert projection is not None
        command["env"].update(_musubi_video_preflight_env(run, projection))
        _write_musubi_dataset_config(
            run, destination / "dataset.toml",
            projection=projection,
        )
    if not explicit_command:
        atomic_write_yaml(destination / "model-bundle.lock.yaml", _musubi_model_lock(run))
        if training_state_managed(run, training_state_contract_musubi(run)):
            write_state_runner(destination)
    return command


def _musubi_video_preflight_env(run: dict[str, Any], projection: dict[str, Any]) -> dict[str, str]:
    """Resolve video preflight semantics from the frozen projection profiles."""
    datasets = projection.get("datasets") if isinstance(projection, dict) else None
    if not isinstance(datasets, list):
        raise ValueError("Musubi frozen projection has no dataset list for video preflight")
    policies: list[dict[str, Any]] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        native = dataset.get("native")
        if not isinstance(native, dict):
            continue
        native_blocks = native.get("datasets")
        if not isinstance(native_blocks, list):
            raise ValueError("Musubi video projection native handoff must contain datasets[]")
        if not any(
            isinstance(block, dict) and isinstance(block.get("video_jsonl_file"), str)
            for block in native_blocks
        ):
            continue
        policy = dataset.get("policy")
        if not isinstance(policy, dict):
            raise ValueError("Musubi video projection has no verified profile policy")
        policies.append(policy)
    if not policies:
        return {}
    target_fps_values = {policy.get("target_fps") for policy in policies}
    resample_modes = {policy.get("fps_resample_mode") for policy in policies}
    profiles = {policy.get("profile") for policy in policies}
    if len(target_fps_values) != 1 or not all(isinstance(value, (int, float)) for value in target_fps_values):
        raise ValueError(f"Musubi video projection profiles disagree on target_fps: {sorted(map(str, target_fps_values))}")
    if len(resample_modes) != 1 or not all(isinstance(value, str) and value for value in resample_modes):
        raise ValueError(f"Musubi video projection profiles disagree on fps_resample_mode: {sorted(map(str, resample_modes))}")
    if not all(isinstance(value, str) and value for value in profiles):
        raise ValueError("Musubi video projection has an invalid profile name")
    architecture = _musubi_architecture(run)
    return {
        "KURA_MUSUBI_ARCHITECTURE": architecture,
        "KURA_MUSUBI_NATIVE_DATASET_ARCHITECTURE": musubi_native_dataset_architecture(architecture),
        "KURA_MUSUBI_TARGET_FPS": str(next(iter(target_fps_values))),
        "KURA_MUSUBI_FPS_RESAMPLE_MODE": str(next(iter(resample_modes))),
        "KURA_MUSUBI_PROFILES": ",".join(sorted(profiles)),
    }


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
    del run
    direct = _int_or_none(override.get("batch_size"))
    return direct


def _musubi_max_resolution(run: dict[str, Any], override: dict[str, Any]) -> int | None:
    del run
    values: list[int] = []
    direct = override.get("resolution")
    if isinstance(direct, list):
        values.extend(value for item in direct if (value := _int_or_none(item)) is not None)
    elif (value := _int_or_none(direct)) is not None:
        values.append(value)
    dataset_options = override.get("dataset_options")
    if isinstance(dataset_options, dict):
        for options in dataset_options.values():
            if not isinstance(options, dict):
                continue
            blocks = options.get("blocks")
            sources = [options, *blocks] if isinstance(blocks, list) else [options]
            for source in sources:
                if not isinstance(source, dict):
                    continue
                for key in ("resolution", "control_resolution"):
                    configured = source.get(key)
                    if isinstance(configured, list):
                        values.extend(
                            value for part in configured
                            if (value := _int_or_none(part)) is not None
                        )
    return max(values) if values else None


def _validate_musubi_resource_flags(run: dict[str, Any], override: dict[str, Any], architecture: str) -> None:
    extra_args = _extra_args(override)
    h2d_extra = any(arg == "--block_swap_h2d_only" or arg.startswith("--block_swap_h2d_only=") for arg in extra_args)
    h2d_typed = _truthy(override.get("block_swap_h2d_only"))
    if h2d_typed and h2d_extra:
        raise ValueError(
            "Musubi backend.config.block_swap_h2d_only duplicates backend.config.extra_args --block_swap_h2d_only"
        )
    h2d_only = h2d_typed or h2d_extra
    checkpointing = _truthy(override.get("gradient_checkpointing")) or "--gradient_checkpointing" in extra_args
    if h2d_only and not checkpointing:
        raise ValueError("Musubi H2D-only block swap requires explicit gradient_checkpointing")
    blocks = _int_or_none(override.get("blocks_to_swap"))
    if blocks is None:
        blocks = _int_or_none(_extra_arg_value(extra_args, "--blocks_to_swap"))
    if h2d_only and (blocks is None or blocks <= 0):
        raise ValueError("Musubi H2D-only block swap requires blocks_to_swap > 0")
    ring_size = override.get("block_swap_ring_size")
    if ring_size is not None:
        if not h2d_only:
            raise ValueError("Musubi block_swap_ring_size requires block_swap_h2d_only=true")
        if isinstance(ring_size, bool) or not isinstance(ring_size, int) or ring_size <= 0:
            raise ValueError("Musubi block_swap_ring_size must be a positive integer")
    pinned = _truthy(override.get("use_pinned_memory_for_block_swap"))
    if pinned and (blocks is None or blocks <= 0):
        raise ValueError("Musubi use_pinned_memory_for_block_swap requires blocks_to_swap > 0")
    if architecture != "flux2":
        return
    model_version = _musubi_flux2_model_version(run)
    if "9b" not in model_version:
        return
    gpu = str(run.get("compute", {}).get("gpu") or "")
    if "A40" not in gpu.upper():
        return
    micro_batch = _musubi_micro_batch(run, override)
    if micro_batch is None or micro_batch <= 1:
        max_resolution = _musubi_max_resolution(run, override)
        rank = _int_or_none(override.get("network_dim"))
        checkpointing = _truthy(override.get("gradient_checkpointing")) or "--gradient_checkpointing" in extra_args
        if max_resolution is not None and max_resolution >= 1024 and (rank is None or rank >= 32) and not checkpointing:
            if _truthy(override.get("allow_a40_uncheckpointed_9b")):
                return
            raise ValueError(
                "Musubi FLUX.2 9B on NVIDIA A40 at 1024-class resolution/rank32 has been observed to OOM "
                "even with batch_size=1. Set backend.config.gradient_checkpointing: true, "
                "or set allow_a40_uncheckpointed_9b: true to accept the risk."
            )
        return
    if not _truthy(override.get("allow_a40_large_micro_batch")):
        raise ValueError(
            "Musubi FLUX.2 9B on NVIDIA A40 treats batch_size as GPU micro-batch; "
            f"batch_size={micro_batch} has been observed to OOM before step 1. "
            "Use batch_size: 1 and keep the intended effective batch with explicit "
            "--gradient_accumulation_steps, or set backend.config."
            "allow_a40_large_micro_batch: true to accept the risk."
        )


def _musubi_uses_sample_prompts(override: dict[str, Any], extra_args: list[str]) -> bool:
    if override.get("sample_prompts") or override.get("sample_every_n_steps") or override.get("sample_every_n_epochs"):
        return True
    return any(arg.startswith("--sample_") for arg in extra_args)


def display_musubi_tuner(run: dict[str, Any]) -> dict[str, Any]:
    """Project adapter-owned native values for generic display and safety."""
    native = _musubi_backend_override(run)
    extra_args = _extra_args(native)
    gradient_accumulation = native.get("gradient_accumulation_steps") or _extra_arg_value(extra_args, "--gradient_accumulation_steps") or 1
    memory = {
        key: native.get(key)
        for key in (
            "fp8_base",
            "fp8_scaled",
            "fp8_t5",
            "fp8_llm",
            "fp8_vl",
            "gradient_checkpointing",
            "block_swap_h2d_only",
            "block_swap_ring_size",
            "use_pinned_memory_for_block_swap",
        )
    }
    memory["blocks_to_swap"] = native.get("blocks_to_swap") or _extra_arg_value(extra_args, "--blocks_to_swap")
    return {
        "architecture": native.get("architecture") or native.get("model_arch"),
        "selector": native.get("task"),
        "rank": native.get("network_dim"),
        "alpha": native.get("network_alpha"),
        "learning_rate": native.get("learning_rate"),
        "scheduler": native.get("lr_scheduler"),
        "batch_size": native.get("batch_size"),
        "gradient_accumulation_steps": gradient_accumulation,
        "resolution": native.get("resolution"),
        "optimizer": native.get("optimizer_type"),
        "precision": native.get("save_precision"),
        "memory": memory,
        "checkpoint": {
            "save_every_n_steps": native.get("save_every_n_steps"),
            "prune_before_step": native.get("prune_checkpoints_before_step"),
        },
    }


def _musubi_common_train_args(run: dict[str, Any], override: dict[str, Any], output_dir: str, output_name: str, *, default_lr: str = "1e-4") -> list[str]:
    recipe = validated_recipe(run, required=True)
    args = [
        "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
        "--learning_rate", str(override.get("learning_rate") or default_lr),
        "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
        "--persistent_data_loader_workers",
        "--network_dim", str(override.get("network_dim") or 32),
    ]
    alpha = override.get("network_alpha")
    if alpha is not None:
        args.extend(["--network_alpha", str(alpha)])
    args.extend([
        "--max_train_steps", str(recipe["steps"]),
        "--save_every_n_steps", str(checkpoint_save_cadence(run, override.get("save_every_n_steps"), contract=training_state_contract_musubi(run))),
        "--save_precision", _musubi_save_precision(override),
        "--seed", str(recipe["seed"]),
        "--output_dir", output_dir,
        "--output_name", output_name,
    ])
    return args


def _backend_env(backend_name: str, override: dict[str, Any]) -> dict[str, str]:
    env = override.get("env", {})
    if env is None:
        return {}
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError(f"{backend_name} command env must be a string-to-string mapping")
    if any(is_secret_name(key) for key in env):
        raise ValueError(f"{backend_name} command env must not contain secrets; use the process environment instead")
    return dict(env)


def command_musubi_tuner(run: dict[str, Any]) -> dict[str, Any]:
    """Return a Musubi Tuner command spec without executing it."""
    override = _musubi_backend_override(run)
    explicit = override.get("command")
    if explicit is not None:
        combined = sorted(set(override) - {"command"})
        if combined:
            raise ValueError("Musubi explicit command cannot be combined with: " + ", ".join(combined))
        validated_recipe(run, required=False)
        if not isinstance(explicit, dict):
            raise ValueError("Musubi Tuner command must be a mapping")
        cwd, argv, env = explicit.get("cwd"), explicit.get("argv"), explicit.get("env", {})
        if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Musubi Tuner command must provide string cwd and argv values")
        if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
            raise ValueError("Musubi Tuner command env must be a string-to-string mapping")
        if any(is_secret_name(key) for key in env):
            raise ValueError("Musubi Tuner command env must not contain secrets; use the process environment instead")
        return {"cwd": cwd, "argv": argv, "env": env}

    architecture = _musubi_architecture(run)
    _validate_musubi_resource_flags(run, override, architecture)
    recipe = validated_recipe(run, required=True)
    paths = _musubi_model_paths(run)
    download_commands, _ = _musubi_model_downloads(run)
    dataset_config = f"/workspace/runs/{run['id']}/resolved/musubi/dataset.toml"
    output_dir = f"/workspace/runs/{run['id']}/outputs"
    output_name = run_output_name(run)
    common = [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
    ]
    backend_cache_path: str | None = None
    precache = bool(override.get("precache", True))
    if architecture == "flux2":
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
            "--learning_rate", str(override.get("learning_rate") or "1e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_flux_2",
            "--network_dim", str(override.get("network_dim") or 32),
            "--max_train_steps", str(recipe["steps"]),
            "--save_every_n_steps", str(checkpoint_save_cadence(run, override.get("save_every_n_steps"), contract=training_state_contract_musubi(run))),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(recipe["seed"]),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        _append_flag(train_argv, override, "fp8_base")
        _append_flag(train_argv, override, "fp8_scaled")
        if override.get("vae_dtype"):
            train_argv.extend(["--vae_dtype", str(override["vae_dtype"])])
        alpha = override.get("network_alpha")
        if alpha is not None:
            train_argv.extend(["--network_alpha", str(alpha)])
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
        argv = _script_command(commands, override, run)
    elif architecture == "wan":
        dit, vae, t5 = _require_paths(paths, ("dit", "vae", "t5"))
        task = musubi_native_task(architecture, override.get("task"))
        native_selector = wan_native_selector(task)
        clip = paths.get("clip")
        one_frame = _truthy(override.get("one_frame"))
        if native_selector.clip_required and not clip:
            raise ValueError(f"Musubi Wan task {task} requires model_paths.clip or model_downloads.clip")
        if one_frame and not native_selector.one_frame_allowed:
            raise ValueError("Musubi Wan one_frame requires task i2v-14B or flf2v-14B")
        dit_high_noise = paths.get("dit_high_noise")
        if dit_high_noise and not native_selector.dual_dit_allowed:
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
            "--learning_rate", str(override.get("learning_rate") or "2e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_wan",
            "--network_dim", str(override.get("network_dim") or 32),
            "--timestep_sampling", str(override.get("timestep_sampling") or "shift"),
            "--discrete_flow_shift", str(override.get("discrete_flow_shift") or "3.0"),
            "--max_train_steps", str(recipe["steps"]),
            "--save_every_n_steps", str(checkpoint_save_cadence(run, override.get("save_every_n_steps"), contract=training_state_contract_musubi(run))),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(recipe["seed"]),
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
        alpha = override.get("network_alpha")
        if alpha is not None:
            train_argv.extend(["--network_alpha", str(alpha)])
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
            if native_selector.i2v_cache:
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
        argv = _script_command(commands, override, run)
    elif architecture == "minimax_h3":
        dit, video_vae, audio_vae, text_encoder = _require_paths(
            paths, ("dit", "video_vae", "audio_vae", "text_encoder")
        )
        task = musubi_native_task(architecture, override.get("task"))
        if task not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("Musubi MiniMax-H3 task must be t2va, fl2va, or ref2va")
        loss_method = str(override.get("h3_loss_method") or "guidance")
        if loss_method not in {"guidance", "training_adapter", "teacher_matching"}:
            raise ValueError(
                "Musubi MiniMax-H3 h3_loss_method must be guidance, training_adapter, or teacher_matching"
            )
        one_frame = _truthy(override.get("one_frame"))
        if one_frame and not _truthy(override.get("video_only")):
            raise ValueError("Musubi MiniMax-H3 one_frame requires video_only=true")
        latent_task = task
        text_task = task
        train_task = task
        teacher_conditions = None
        if loss_method == "training_adapter":
            if not paths.get("base_weights"):
                raise ValueError(
                    "Musubi MiniMax-H3 training_adapter requires model_paths.base_weights "
                    "or model_downloads.base_weights"
                )
            if "int8_convrot" in Path(dit).name.lower():
                raise ValueError(
                    "Musubi MiniMax-H3 training_adapter cannot merge base_weights into a "
                    "pre-quantized ConvRot INT8 DiT; provide a BF16 DiT and optionally request "
                    "upstream dynamic ConvRot quantization through a reviewed execution setting"
                )
        elif loss_method == "teacher_matching":
            if task != "t2va":
                raise ValueError("Musubi MiniMax-H3 teacher_matching requires task t2va")
            teacher_conditions = str(override.get("h3_teacher_conditions") or "")
            if teacher_conditions not in {"first,last", "ref", "subject_ref"}:
                raise ValueError(
                    "Musubi MiniMax-H3 teacher_matching h3_teacher_conditions must be "
                    "first,last, ref, or subject_ref"
                )
            if one_frame and teacher_conditions != "subject_ref":
                raise ValueError(
                    "Musubi MiniMax-H3 teacher_matching one_frame requires h3_teacher_conditions=subject_ref"
                )
            latent_task = _musubi_h3_effective_task(override)
            text_task = "t2va"
            train_task = "t2va"
        text_encoder_blocks_to_swap = _int_or_none(override.get("text_encoder_blocks_to_swap"))
        if text_encoder_blocks_to_swap is not None and not 0 <= text_encoder_blocks_to_swap <= 50:
            raise ValueError("Musubi MiniMax-H3 text_encoder_blocks_to_swap must be in [0, 50]")
        blocks_to_swap = _int_or_none(override.get("blocks_to_swap"))
        if blocks_to_swap is not None and not 0 <= blocks_to_swap <= 48:
            raise ValueError("Musubi MiniMax-H3 blocks_to_swap must be in [0, 48]")
        backend_cache_path = f"/workspace/runs/{run['id']}/cache/musubi"
        uncond_cache = f"{backend_cache_path}/minimax-h3-uncond.safetensors"
        train_argv = [
            *common, "src/musubi_tuner/minimax_h3_train_network.py",
            "--dataset_config", dataset_config,
            "--task", train_task,
            "--dit", dit,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora_minimax_h3",
            *_musubi_common_train_args(run, override, output_dir, output_name),
        ]
        if loss_method == "guidance":
            if not precache:
                raise ValueError("Musubi MiniMax-H3 guidance requires precache=true to produce the unconditional cache")
            guidance_scale = float(override.get("h3_guidance_loss_scale", 4.0))
            guidance_sigma_min = float(override.get("h3_guidance_loss_sigma_min", 0.15))
            if guidance_scale <= 0:
                raise ValueError("Musubi MiniMax-H3 requires h3_guidance_loss_scale > 0")
            if not 0 <= guidance_sigma_min <= 1:
                raise ValueError("Musubi MiniMax-H3 h3_guidance_loss_sigma_min must be in [0, 1]")
            train_argv.extend([
                "--h3_guidance_loss_scale", str(guidance_scale),
                "--h3_guidance_loss_sigma_min", str(guidance_sigma_min),
                "--h3_guidance_loss_uncond_cache", uncond_cache,
            ])
        elif loss_method == "training_adapter":
            train_argv.extend(["--base_weights", str(paths["base_weights"])])
        else:
            train_argv.extend([
                "--h3_teacher_matching",
                "--h3_teacher_conditions", str(teacher_conditions),
            ])
            for key in (
                "h3_teacher_condition_sigma_min",
                "h3_teacher_condition_sigma_max",
                "h3_teacher_loss_dc_weight",
                "h3_teacher_loss_mag_weight",
                "h3_teacher_preservation_weight",
                "h3_timestep_focus_min",
                "h3_timestep_focus_max",
                "h3_timestep_focus_prob",
            ):
                if override.get(key) is not None:
                    train_argv.extend(["--" + key, str(override[key])])
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if one_frame:
            train_argv.append("--one_frame")
        if _truthy(override.get("video_only")):
            train_argv.append("--video_only")
        train_argv.extend(_extra_args(override))
        commands = _musubi_start_commands(dataset_config, download_commands)
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/minimax_h3_cache_latents.py",
                "--dataset_config", dataset_config,
                "--task", latent_task,
                "--video_vae", video_vae,
                "--audio_vae", audio_vae,
                "--cache_seed", str(recipe["seed"]),
                "--skip_existing",
            ]
            text_argv = [
                "python", "src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--task", text_task,
                "--text_encoder", text_encoder,
                "--text_cache_dtype", "bf16",
                "--skip_existing",
            ]
            if loss_method == "guidance":
                text_argv.extend(["--uncond_output", uncond_cache])
            elif loss_method == "teacher_matching":
                text_argv.extend(["--teacher_conditions", str(teacher_conditions)])
            if one_frame:
                latent_argv.append("--one_frame")
                text_argv.append("--one_frame")
            commands.extend([latent_argv, text_argv])
            if text_encoder_blocks_to_swap is not None:
                commands[-1].extend(["--text_encoder_blocks_to_swap", str(text_encoder_blocks_to_swap)])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(
            output_dir, output_name, override.get("prune_checkpoints_before_step")
        )
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands, override, run)
    elif architecture == "krea2":
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        extra_args = _extra_args(override)
        convrot_int8 = _truthy(override.get("convrot_int8"))
        convrot_int8_bwd = override.get("convrot_int8_bwd")
        checkpoint_cpu_offload = _truthy(override.get("gradient_checkpointing_cpu_offload"))
        if convrot_int8 and (_truthy(override.get("fp8_base")) or _truthy(override.get("fp8_scaled"))):
            raise ValueError("Musubi Krea 2 convrot_int8 cannot be combined with fp8_base or fp8_scaled")
        if convrot_int8 and _truthy(override.get("include_turbo_dit")):
            raise ValueError("Musubi Krea 2 convrot_int8 cannot be combined with include_turbo_dit")
        if convrot_int8_bwd is not None:
            if not isinstance(convrot_int8_bwd, str) or convrot_int8_bwd not in ("bf16", "int8"):
                raise ValueError("Musubi Krea 2 convrot_int8_bwd must be one of: bf16, int8")
            if not convrot_int8:
                raise ValueError("Musubi Krea 2 convrot_int8_bwd requires convrot_int8=true")
        if checkpoint_cpu_offload and not _truthy(override.get("gradient_checkpointing")):
            raise ValueError(
                "Musubi Krea 2 gradient_checkpointing_cpu_offload requires gradient_checkpointing=true"
            )
        train_argv = [
            *common, "src/musubi_tuner/krea2_train_network.py",
            "--dit", dit, "--vae", vae,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--timestep_sampling", str(override.get("timestep_sampling") or "krea2_shift"),
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(override.get("learning_rate") or "1e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_krea2",
            "--network_dim", str(override.get("network_dim") or 32),
            "--network_alpha", str(override.get("network_alpha") or override.get("network_dim") or 32),
            "--max_train_steps", str(recipe["steps"]),
            "--save_every_n_steps", str(checkpoint_save_cadence(run, override.get("save_every_n_steps"), contract=training_state_contract_musubi(run))),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(recipe["seed"]),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if checkpoint_cpu_offload:
            train_argv.append("--gradient_checkpointing_cpu_offload")
        if convrot_int8:
            train_argv.append("--convrot_int8")
        if convrot_int8_bwd is not None:
            train_argv.extend(["--convrot_int8_bwd", convrot_int8_bwd])
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
        argv = _script_command(commands, override, run)
    elif architecture == "qwen_image":
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        model_version = _musubi_model_version(run)
        if _truthy(override.get("remove_first_image_from_target")) and model_version != "layered":
            raise ValueError(
                "Musubi remove_first_image_from_target requires model_version=layered"
            )
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
        _append_flag(train_argv, override, "remove_first_image_from_target")
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
        argv = _script_command(commands, override, run)
    elif architecture == "zimage":
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
        argv = _script_command(commands, override, run)
    elif architecture == "flux_kontext":
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
        argv = _script_command(commands, override, run)
    elif architecture == "ideogram4":
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
        argv = _script_command(commands, override, run)
    elif architecture == "hidream_o1":
        dit = _require_paths(paths, ("dit",))[0]
        model_type = str(override.get("model_type") or "full")
        task = musubi_native_task(architecture, override.get("task"))
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
        argv = _script_command(commands, override, run)
    elif architecture == "hunyuan_video":
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
        argv = _script_command(commands, override, run)
    elif architecture == "hunyuan_video_1_5":
        task = musubi_native_task(architecture, override.get("task"))
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
        argv = _script_command(commands, override, run)
    elif architecture == "framepack":
        dit, vae, text_encoder1, text_encoder2, image_encoder = _require_paths(paths, ("dit", "vae", "text_encoder1", "text_encoder2", "image_encoder"))
        train_argv = [
            *common, "src/musubi_tuner/fpack_train_network.py",
            "--dit", dit, "--vae", vae,
            "--text_encoder1", text_encoder1, "--text_encoder2", text_encoder2,
            "--image_encoder", image_encoder,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--network_module", "networks.lora_framepack",
            "--latent_window_size", str(FRAMEPACK_LATENT_WINDOW_SIZE),
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
        argv = _script_command(commands, override, run)
    elif architecture == "kandinsky5":
        dit, vae, text_encoder_qwen, text_encoder_clip = _require_paths(paths, ("dit", "vae", "text_encoder_qwen", "text_encoder_clip"))
        task = musubi_native_task(architecture, override.get("task"))
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
        argv = _script_command(commands, override, run)
    else:
        raise _unsupported_musubi_adapter_error(architecture)

    env = _backend_env("Musubi Tuner", override)
    env.update({
        "KURA_MUSUBI_IMAGE_SUFFIXES": frozen_suffixes(MUSUBI_IMAGE_SUFFIXES),
        "KURA_MUSUBI_VIDEO_SUFFIXES": frozen_suffixes(MUSUBI_VIDEO_SUFFIXES),
        "KURA_MUSUBI_AUDIO_SUFFIXES": frozen_suffixes(MUSUBI_AUDIO_SUFFIXES),
    })
    result = {
        "cwd": "/opt/musubi-tuner", "argv": argv, "env": env,
        "output_contract": {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]},
    }
    if backend_cache_path is not None:
        env["KURA_MUSUBI_CACHE"] = backend_cache_path
        result["write_roots"] = [{
            "role": "backend-cache",
            "path": backend_cache_path,
            "env": "KURA_MUSUBI_CACHE",
        }]
    return result
