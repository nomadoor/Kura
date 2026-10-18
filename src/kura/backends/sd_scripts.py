"""sd-scripts backend compiler and Tier 1 command selectors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kura.backends.sd_scripts_datasets import (
    _text_cache_enabled,
    display_sd_scripts_dataset_config,
    validate_sd_scripts_dataset_config,
    write_sd_scripts_dataset_config,
)
from kura.backends.sd_scripts_models import OPTIONAL_ROLES, sd_scripts_architecture, sd_scripts_download_commands, sd_scripts_mode, sd_scripts_model_lock, sd_scripts_model_paths, sd_scripts_native
from kura.backends.shared import _extra_args as _shared_extra_args, append_config_args, argv_value, _reject_owned_extra_args, _script_command, _truthy, state_runner_argv, write_state_runner
from kura.container_scripts import script_source
from kura.dataset_handoff import load_frozen_dataset_projection
from kura.fsio import atomic_write_json, atomic_write_yaml
from kura.secrets import is_secret_name
from kura.training_artifacts import EPOCH_SAVE_FLAGS, managed_state_save_args, require_constant_scheduler_for_resume, resume_steps, run_output_name, training_state_managed, training_state_payload
from kura.run_envelope import SettingsRefused, backend_config, resume_intent, validated_recipe


# Spellings a run compiled before selectors had one spelling may hold; compile refuses them now,
# but the training-state contract of such a run still in flight reads them as these.
_RECORDED_ARCHITECTURES = {"sd_1_5": "sd15", "stable_diffusion_1_5": "sd15", "stable_diffusion15": "sd15", "sdxl_1_0": "sdxl", "flux": "flux1", "flux_1": "flux1"}
_RECORDED_MODES = {"lllite": "controlnet_lllite"}


def _recorded_selector(value: str, aliases: dict[str, str]) -> str:
    normalized = value.lower().replace("-", "_").replace(".", "_")
    return aliases.get(normalized, normalized)


def training_state_contract_sd_scripts(run: dict[str, Any]) -> dict[str, Any]:
    native = backend_config(run, "sd-scripts")
    authored_architecture = native.get("architecture")
    architecture = (
        ""
        if authored_architecture is None or (isinstance(authored_architecture, str) and not authored_architecture.strip())
        else _recorded_selector(sd_scripts_architecture(run), _RECORDED_ARCHITECTURES)
    )
    mode = _recorded_selector(sd_scripts_mode(run), _RECORDED_MODES)
    common = ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl")
    if (architecture, mode) in {("sd15", "lora"), ("sdxl", "lora"), ("flux1", "lora")}:
        return {
            "native_format": "accelerate-state-directory",
            "required_files": (*common, "train_state.json", "kura-state-info.json"),
            # The pinned image's patch 0002 keeps sd-scripts' step counter and progress bar at
            # the logical step on Resume; a lock compiled before it froze `process_local`.
            "native_progress": "logical",
            "native_target": "logical",
            "state_step": {
                "path": "kura-state-info.json", "field": "logical_step", "space": "logical",
                "schema_version": 1, "backend": "sd-scripts",
                "digests": {
                    "train_state_sha256": "train_state.json",
                    "optimizer_sha256": "optimizer.bin",
                    "scheduler_sha256": "scheduler.bin",
                },
            },
            "capability": "best_effort_resume",
            "restoration_contract": {
                "level": "best_effort_resume",
                "restored": ["model", "optimizer", "scheduler", "rng", "scaler_when_present", "application_global_step"],
                "not_restored": ["application_epoch_counter", "exact_dataloader_position"],
                "limitations": ["Kura normalizes sd-scripts application step metadata from the persisted scheduler after each complete save"],
                "scheduler_behavior": "restored; initial Resume execution limited to constant scheduler",
            },
        }
    return {
        "native_format": "accelerate-state-directory",
        "required_files": common,
        "native_progress": "logical",
        "native_target": "logical",
        "capability": "unsupported",
        "restoration_contract": {
            "level": "unsupported",
            "restored": [],
            "not_restored": ["model", "optimizer", "scheduler", "rng", "application_global_step", "epoch", "exact_dataloader_position"],
            "limitations": [f"{architecture}/{mode} execution is not yet supported by Kura"],
            "scheduler_behavior": "backend state only; no supported Kura Resume execution envelope",
        },
    }


ENTRYPOINTS = {
    ("sd15", "lora"): "train_network.py",
    ("sdxl", "lora"): "sdxl_train_network.py",
    ("flux1", "lora"): "flux_train_network.py",
    ("anima", "lora"): "anima_train_network.py",
    ("anima", "controlnet_lllite"): "anima_train_control_net_lllite.py",
}
_OPTIONAL_ROLE_FLAGS = {"vae": "--vae", "llm_adapter": "--llm_adapter_path", "t5_tokenizer": "--t5_tokenizer_path"}
NETWORK_MODULES = {"sd15": "networks.lora", "sdxl": "networks.lora", "flux1": "networks.lora_flux", "anima": "networks.lora_anima"}
# --lr_scheduler_type replaces the named schedule and --max_train_epochs overrides the steps
# (library/optimizer.py:509 and train_network.py:1206 at v0.12.0), so they are owned too.
# --save_state_on_train_end is no longer emitted (the trainer saves the end state under
# --save_state alone) but stays owned: written without --save_state, where Kura manages no state,
# it would save a state Kura neither publishes nor counts in the disk estimate.
_COMMAND_FLAGS = EPOCH_SAVE_FLAGS | {"--lr_scheduler_type", "--max_train_epochs"} | {
    "--dataset_config", "--pretrained_model_name_or_path", "--clip_l", "--t5xxl", "--ae", "--qwen3", "--vae",
    "--llm_adapter_path", "--t5_tokenizer_path", "--network_module", "--network_dim", "--network_alpha",
    "--learning_rate", "--optimizer_type", "--lr_scheduler", "--mixed_precision", "--max_train_steps", "--seed",
    "--output_dir", "--output_name", "--save_model_as", "--save_every_n_steps", "--save_last_n_steps",
    "--save_state", "--save_state_on_train_end", "--save_last_n_steps_state", "--resume", "--skip_until_initial_step",
    "--gradient_accumulation_steps", "--network_train_unet_only",
    "--gradient_checkpointing", "--fp8_base", "--blocks_to_swap",
    "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
    "--cpu_offload_checkpointing", "--unsloth_offload_checkpointing",
    "--deepspeed", "--fused_backward_pass",
    "--timestep_sampling", "--discrete_flow_shift", "--sigmoid_scale", "--guidance_scale", "--model_prediction_type",
    "--attn_mode",
    "--qwen_image_vae_2d", "--vae_chunk_size", "--unet_lr", "--text_encoder_lr",
    "--cond_emb_dim", "--lllite_mlp_dim", "--lllite_target_layers", "--lllite_cond_dim", "--lllite_cond_resblocks",
    "--lllite_dropout", "--lllite_multiplier", "--lllite_cond_in_channels", "--lllite_use_aspp",
    # The generated dataset TOML owns these values even when a pinned trainer
    # parser also exposes a command-line fallback for them.
    "--train_data_dir", "--conditioning_data_dir", "--dataset_class", "--cache_info",
    "--batch_size", "--resolution", "--enable_bucket", "--bucket_no_upscale",
    "--min_bucket_reso", "--max_bucket_reso", "--bucket_reso_steps",
    "--caption_extension", "--caption_extention", "--shuffle_caption",
    "--caption_separator", "--keep_tokens", "--keep_tokens_separator",
    "--secondary_separator", "--enable_wildcard", "--caption_prefix",
    "--caption_suffix", "--caption_dropout_rate", "--caption_dropout_every_n_epochs",
    "--caption_tag_dropout_rate", "--token_warmup_min", "--token_warmup_step",
    "--color_aug", "--flip_aug", "--random_crop", "--network_multiplier",
}
CONFIG_KEYS = {
    "architecture", "mode", "command", "model_paths", "model_downloads", "dataset_config", "flatten_groups", "output_name", "env", "extra_args",
    "network_dim", "network_alpha", "learning_rate", "optimizer_type", "lr_scheduler", "mixed_precision",
    "gradient_checkpointing", "gradient_accumulation_steps", "network_train_unet_only", "fp8_base", "blocks_to_swap",
    "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
    "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
    "save_every_n_steps", "save_last_n_steps", "timestep_sampling", "discrete_flow_shift", "sigmoid_scale", "guidance_scale",
    "model_prediction_type", "attn_mode", "qwen_image_vae_2d", "vae_chunk_size", "unet_lr", "text_encoder_lr",
    "cond_emb_dim", "lllite_mlp_dim", "lllite_target_layers", "lllite_cond_dim", "lllite_cond_resblocks", "lllite_dropout",
    "lllite_multiplier", "lllite_cond_in_channels", "lllite_use_aspp",
}
# Flags Kura owns that no declared field names; the flag of a declared field is refused in
# extra_args by validate_backend_config wherever the field applies.
OWNED_FLAGS = _COMMAND_FLAGS - {f"--{key}" for key in CONFIG_KEYS}
BOOLEAN_CONFIG_KEYS = {
    "gradient_checkpointing", "network_train_unet_only", "fp8_base", "cache_latents", "cache_latents_to_disk",
    "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk", "qwen_image_vae_2d", "cpu_offload_checkpointing",
    "unsloth_offload_checkpointing", "lllite_use_aspp", "flatten_groups",
}


def _extra_args(native: dict[str, Any]) -> list[str]:
    values = _shared_extra_args(native, backend_label="sd-scripts")
    _reject_owned_extra_args(
        values, owned_flags=OWNED_FLAGS, backend_label="sd-scripts",
    )
    return values


def _safe_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"sd-scripts {field} must contain only letters, digits, dot, underscore, or hyphen")
    return value


def _explicit_command(run: dict[str, Any], command: Any) -> dict[str, Any]:
    validated_recipe(run, required=False)
    if not isinstance(command, dict):
        raise ValueError("sd-scripts backend.config.command must be a mapping")
    cwd, argv, env = command.get("cwd"), command.get("argv"), command.get("env", {})
    if not isinstance(cwd, str) or not cwd or not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("sd-scripts command must provide non-empty string cwd and argv values")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("sd-scripts command env must be a string-to-string mapping")
    if any(is_secret_name(key) for key in env):
        raise ValueError("sd-scripts command env must not contain secrets; use the process environment instead")
    return {"cwd": cwd, "argv": list(argv), "env": dict(env)}


def _validate_native_env(native: dict[str, Any]) -> dict[str, str]:
    env = native.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("sd-scripts env must be a string-to-string mapping")
    if any(is_secret_name(key) for key in env):
        raise ValueError("sd-scripts env must not contain secrets; use the process environment instead")
    return dict(env)


def _validate_selector(native: dict[str, Any], architecture: str, mode: str) -> None:
    """Refuse a selector pair or combination the trainer cannot run; registry conditions own
    which fields apply where, and every forwarded value reaches the trainer as written."""
    if (architecture, mode) not in ENTRYPOINTS:
        raise ValueError(
            f"unsupported Kura built-in sd-scripts adapter: architecture={architecture}, mode={mode}. "
            "sd-scripts may support this path upstream, but Kura does not generate it yet; use backend.config.command for a complete reviewed command."
        )
    if mode == "controlnet_lllite" and native.get("lllite_cond_in_channels", 3) != 3:
        # Kura projects image conditioning only.
        raise ValueError(f"sd-scripts Anima LLLite is image-only and requires lllite_cond_in_channels: 3, not {native['lllite_cond_in_channels']!r}")
    if architecture == "anima" and mode == "lora":
        if _text_cache_enabled(native) and not _truthy(native.get("network_train_unet_only")):
            raise ValueError("sd-scripts Anima LoRA text-encoder caching requires network_train_unet_only=true")
        incompatible_pairs = (("blocks_to_swap", "cpu_offload_checkpointing"), ("blocks_to_swap", "unsloth_offload_checkpointing"), ("cpu_offload_checkpointing", "unsloth_offload_checkpointing"))
        for left, right in incompatible_pairs:
            if native.get(left) not in (None, False, 0) and _truthy(native.get(right)):
                raise ValueError(f"sd-scripts Anima LoRA cannot combine {left} with {right}")
    if architecture == "flux1" and native.get("blocks_to_swap") not in (None, 0) and _truthy(native.get("cpu_offload_checkpointing")):
        raise ValueError("sd-scripts FLUX.1 cannot combine blocks_to_swap with cpu_offload_checkpointing")
    if architecture in {"sdxl", "flux1"} and _text_cache_enabled(native) and not _truthy(native.get("network_train_unet_only")):
        raise ValueError(f"sd-scripts {architecture} text-encoder caching requires network_train_unet_only=true")


def validate_sd_scripts_authored(run: dict[str, Any]) -> None:
    """Refuse every independent problem of a built-in run's settings at once: its dataset_config,
    its selector combination, and its model roles. An architecture or mode that does not name a
    built-in entrypoint stops first, since the other checks read it. command_sd_scripts calls the
    same checks again at compile."""
    native = sd_scripts_native(run)
    if native.get("command") is not None:
        return
    architecture, mode = sd_scripts_architecture(run), sd_scripts_mode(run)
    if (architecture, mode) not in ENTRYPOINTS:
        _validate_selector(native, architecture, mode)
    problems: list[str] = []
    for check in (
        lambda: validate_sd_scripts_dataset_config(run),
        lambda: _validate_selector(native, architecture, mode),
        lambda: sd_scripts_model_paths(run),
    ):
        try:
            check()
        except ValueError as exc:
            problems.append(str(exc))
    if len(problems) == 1:
        raise SettingsRefused(problems[0])
    if problems:
        raise SettingsRefused(f"{len(problems)} problems: " + " ".join(f"({index}) {problem.rstrip('.')}." for index, problem in enumerate(problems, 1)))


def _base_training_args(run: dict[str, Any], native: dict[str, Any], paths: dict[str, str], output_dir: str, output_name: str) -> list[str]:
    architecture, mode = sd_scripts_architecture(run), sd_scripts_mode(run)
    recipe = validated_recipe(run, required=True)
    continuation = resume_intent(run)
    native_steps = recipe["steps"]
    if continuation is not None:
        if (architecture, mode) not in {("sd15", "lora"), ("sdxl", "lora"), ("flux1", "lora")}:
            raise ValueError(f"sd-scripts {architecture}/{mode} State Resume is not yet supported")
        require_constant_scheduler_for_resume("sd-scripts", native.get("lr_scheduler"))
        if native.get("gradient_accumulation_steps", 1) != 1:
            raise ValueError("sd-scripts State Resume initially requires gradient_accumulation_steps=1")
        native_steps = resume_steps(run, contract=training_state_contract_sd_scripts(run))["native_end"]
    args = [
        ENTRYPOINTS[(architecture, mode)],
        "--dataset_config", f"/workspace/runs/{run['id']}/resolved/sd-scripts/dataset.toml",
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--save_model_as", "safetensors",
        *argv_value("--max_train_steps", native_steps),
        *argv_value("--seed", recipe["seed"]),
    ]
    # Registry conditions decide which of these apply to the selected architecture and mode.
    append_config_args(args, native, "learning_rate", "optimizer_type", "lr_scheduler", "mixed_precision")
    if architecture in ("sd15", "sdxl"):
        args.extend(["--pretrained_model_name_or_path", paths["base"], "--network_module", NETWORK_MODULES[architecture]])
    elif architecture == "flux1":
        args.extend(["--pretrained_model_name_or_path", paths["dit"], "--clip_l", paths["clip_l"], "--t5xxl", paths["t5xxl"], "--ae", paths["ae"], "--network_module", NETWORK_MODULES[architecture]])
    elif architecture == "anima":
        args.extend(["--pretrained_model_name_or_path", paths["dit"], "--qwen3", paths["qwen3"], "--vae", paths["vae"]])
    for role in OPTIONAL_ROLES[(architecture, mode)]:
        if paths.get(role):
            args.extend([_OPTIONAL_ROLE_FLAGS[role], paths[role]])
    if architecture == "anima" and mode == "lora":
        args.extend(["--network_module", NETWORK_MODULES[architecture]])
    append_config_args(
        args, native,
        "network_dim", "network_alpha", "unet_lr", "text_encoder_lr",
        "gradient_checkpointing", "gradient_accumulation_steps", "network_train_unet_only", "fp8_base",
    )
    save_args: list[str] = []
    if training_state_managed(run, training_state_contract_sd_scripts(run)):
        save_args = managed_state_save_args(
            run, native.get("save_every_n_steps"), contract=training_state_contract_sd_scripts(run),
        )
    for key in ("save_every_n_steps", "save_last_n_steps"):
        if key == "save_every_n_steps" and save_args[:1] == ["--save_every_n_steps"]:
            # Kura's cadence takes the configured one's place, so a fresh run's argv is unchanged.
            args.extend(save_args[:2])
            save_args = save_args[2:]
        else:
            append_config_args(args, native, key)
    args.extend(save_args)
    if continuation is not None:
        artifact_id = continuation["source"]["artifact_id"]
        args.extend(["--resume", training_state_payload(artifact_id), "--skip_until_initial_step"])
    append_config_args(
        args, native,
        "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
        "blocks_to_swap", "timestep_sampling", "guidance_scale", "model_prediction_type", "discrete_flow_shift",
        "sigmoid_scale", "attn_mode", "qwen_image_vae_2d", "vae_chunk_size",
        "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
        "cond_emb_dim", "lllite_mlp_dim", "lllite_target_layers", "lllite_cond_dim", "lllite_cond_resblocks",
        "lllite_dropout", "lllite_multiplier", "lllite_cond_in_channels", "lllite_use_aspp",
    )
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
    download_commands, downloaded = sd_scripts_download_commands(run)
    paths.update(downloaded)
    # A Resume run owns a new output namespace.  The source output_name is part
    # of the frozen training recipe, but reusing it would make the derived run
    # report (and, with a shared workspace, potentially overwrite) source-run
    # weights.
    if "output_name" in native:
        _safe_name(native["output_name"], field="output_name")
    output_name = _safe_name(run_output_name(run), field="output_name")
    final_output = f"/workspace/runs/{run['id']}/outputs"
    train_output = f"/workspace/runs/{run['id']}/cache/sd-scripts/native-output" if architecture == "anima" and mode == "lora" else final_output
    model_items = [{"role": role, "path": path} for role, path in sorted(paths.items())]
    native_training_argv = _base_training_args(run, native, paths, train_output, output_name)
    training_argv = ["accelerate", "launch", "--num_cpu_threads_per_process", "1"]
    state_contract = training_state_contract_sd_scripts(run)
    if training_state_managed(run, state_contract):
        training_argv.extend(
            state_runner_argv(
                "sd-scripts", f"/workspace/runs/{run['id']}/resolved/sd-scripts", native_training_argv[0], native_training_argv[1:],
            )
        )
    else:
        training_argv.extend(native_training_argv)
    commands: list[list[str]] = [
        *download_commands,
        _validation_command(models=model_items),
    ]
    if resume_intent(run) is not None:
        commands.append(
            [
                "python",
                "-c",
                script_source("training_state_verify.py"),
                f"/workspace/runs/{run['id']}/resolved/training-state-source.lock.json",
                "/workspace",
            ]
        )
    if architecture == "anima" and mode == "lora":
        commands.append(_anima_conversion_command(run["id"], output_name, training_argv))
    else:
        commands.append(training_argv)
        kind = "anima-lllite" if mode == "controlnet_lllite" else "lora"
        commands.append(_validation_command(pattern=f"{final_output}/{output_name}.safetensors", kind=kind))
    return {
        "cwd": "/opt/sd-scripts", "argv": _script_command(commands, step_name="sd-scripts"), "env": _validate_native_env(native),
        "output_contract": {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]},
    }


def sd_scripts_disk_cache(run: dict[str, Any]) -> str | None:
    """What run-scoped disk cache the trainer writes, or None: latents or text-encoder outputs
    cached to disk when the run sets `cache_latents_to_disk` or `cache_text_encoder_outputs_to_disk`."""
    native = backend_config(run, "sd-scripts")
    if native.get("command") is not None:
        return None
    written = [key for key in ("cache_latents_to_disk", "cache_text_encoder_outputs_to_disk") if native.get(key) is True]
    return f"sd-scripts writes an on-disk cache ({', '.join(written)})" if written else None


def compile_sd_scripts(run: dict[str, Any], destination: Path) -> dict[str, Any]:
    native = sd_scripts_native(run)
    destination.mkdir(parents=True, exist_ok=True)
    if native.get("command") is not None:
        command = command_sd_scripts(run)
        atomic_write_yaml(destination / "model-bundle.lock.yaml", {"schema_version": 1, "backend": "sd-scripts", "ownership": "explicit-command"})
        atomic_write_json(destination / "command.json", command)
        return command
    projection = load_frozen_dataset_projection(
        destination.parent,
        backend="sd-scripts",
        dataset_ids=[str(item.get("id")) for item in run.get("datasets", [])],
    )
    assert projection is not None
    write_sd_scripts_dataset_config(
        run, destination / "dataset.toml",
        projection=projection,
    )
    atomic_write_yaml(destination / "model-bundle.lock.yaml", sd_scripts_model_lock(run))
    if training_state_managed(run, training_state_contract_sd_scripts(run)):
        write_state_runner(destination)
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
    dataset_display = display_sd_scripts_dataset_config(native)
    displayed_datasets = dataset_display["datasets"]

    def uniform_dataset_value(key: str, fallback: Any = None) -> Any:
        values = [item.get(key) for item in displayed_datasets]
        if not values:
            return fallback
        return values[0] if all(value == values[0] for value in values[1:]) else None
    # Unset values are the trainer's own defaults, which Kura does not restate.
    flow: dict[str, Any] = {}
    if architecture == "flux1":
        flow = {key: native.get(key) for key in ("timestep_sampling", "discrete_flow_shift", "sigmoid_scale", "guidance_scale", "model_prediction_type")}
    elif architecture == "anima":
        flow = {key: native.get(key) for key in ("timestep_sampling", "discrete_flow_shift", "sigmoid_scale")}
    return {
        "architecture": architecture, "mode": mode,
        "rank": native.get("network_dim") if mode == "lora" else None, "alpha": native.get("network_alpha") if mode == "lora" else None,
        "learning_rate": native.get("learning_rate"), "scheduler": native.get("lr_scheduler"),
        "batch_size": uniform_dataset_value("batch_size", general.get("batch_size")), "gradient_accumulation_steps": native.get("gradient_accumulation_steps", 1), "resolution": uniform_dataset_value("resolution", general.get("resolution")),
        "optimizer": native.get("optimizer_type"), "precision": native.get("mixed_precision"),
        "flow_matching": flow,
        "component_learning_rates": {key: native.get(key) for key in ("unet_lr", "text_encoder_lr")},
        "memory": {
            **{key: native.get(key) for key in ("gradient_checkpointing", "fp8_base", "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk", "blocks_to_swap", "qwen_image_vae_2d", "vae_chunk_size")},
            "attn_mode": native.get("attn_mode", "sdpa") if architecture == "anima" and mode == "controlnet_lllite" else native.get("attn_mode"),
        },
        "checkpoint": {"save_every_n_steps": native.get("save_every_n_steps"), "retention_window_steps": native.get("save_last_n_steps")},
        "dataset_config": dataset_display,
    }
