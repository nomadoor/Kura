"""Single registry for backend adapter ownership and dispatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kura.backends.ai_toolkit import AI_TOOLKIT_DATASET_FIELD_SPECS, AI_TOOLKIT_DATASET_OPTION_CAPABILITIES, AI_TOOLKIT_PINNED_MODEL_ARCHS, command_ai_toolkit, compile_ai_toolkit, display_ai_toolkit, project_ai_toolkit_dataset, requirements_ai_toolkit, runtime_checks_ai_toolkit, training_state_contract_ai_toolkit, validate_ai_toolkit_config
from kura.backends.common import MUSUBI_ARCHITECTURE_ALIASES, canonical_musubi_architecture
from kura.backends.musubi_command import command_musubi_tuner, compile_musubi_tuner, display_musubi_tuner, training_state_contract_musubi
from kura.backends.musubi_models import requirements_musubi
from kura.backends.musubi_models import musubi_model_download_specs
from kura.backends.musubi_datasets import MUSUBI_DATASET_OPTION_CAPABILITIES, musubi_general_resolution, project_musubi_dataset, runtime_checks_musubi, validate_musubi_authored_config
from kura.backends.sd_scripts import BOOLEAN_CONFIG_KEYS, CONFIG_KEYS, command_sd_scripts, compile_sd_scripts, display_sd_scripts, sd_scripts_disk_cache_estimate, training_state_contract_sd_scripts
from kura.backends.sd_scripts_datasets import SD_SCRIPTS_DATASET_CAPABILITIES, project_sd_scripts_dataset, validate_sd_scripts_dataset_config
from kura.backends.sd_scripts_models import requirements_sd_scripts, sd_scripts_model_download_specs
from kura.run_envelope import COMMON_RECIPE_FIELDS, backend_config


Compile = Callable[[dict[str, Any], Path], dict[str, Any]]
ProjectDataset = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class FieldCondition:
    """A field is valid when at least one selector clause matches."""

    field: str
    when_any: tuple[tuple[tuple[str, tuple[Any, ...]], ...], ...]


def _when(field: str, **selectors: tuple[Any, ...]) -> FieldCondition:
    return FieldCondition(field, (tuple((name, values) for name, values in selectors.items()),))


def _when_any(field: str, *clauses: dict[str, tuple[Any, ...]]) -> FieldCondition:
    return FieldCondition(field, tuple(tuple((name, values) for name, values in clause.items()) for clause in clauses))


@dataclass(frozen=True)
class SelectorNormalization:
    """Resolve authored selector names and values before surface conditions."""

    field: str
    aliases: tuple[str, ...]
    normalize: Callable[[Any], Any]
    rule: str
    value_aliases: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BackendSurface:
    """The adapter-owned authoring vocabulary accepted by Kura."""

    fields: frozenset[str]
    escape_hatches: frozenset[str] = frozenset()
    # Fields that take YAML true or false only; validate_backend_config refuses anything else.
    boolean_fields: frozenset[str] = frozenset()
    conditions: tuple[FieldCondition, ...] = ()
    selector_defaults: tuple[tuple[str, Any], ...] = ()
    unavailable: tuple[tuple[str, str], ...] = ()
    nested_config_fields: dict[str, dict[str, dict[str, Any]]] | None = None
    config_value_choices: tuple[tuple[str, tuple[str, ...]], ...] = ()
    selector_normalizations: tuple[SelectorNormalization, ...] = ()

    def __post_init__(self) -> None:
        if self.boolean_fields - self.fields:
            raise ValueError("boolean backend fields are not declared: " + ", ".join(sorted(self.boolean_fields - self.fields)))
        overlap = self.fields & self.escape_hatches
        if overlap:
            raise ValueError("backend surface fields and escape hatches overlap: " + ", ".join(sorted(overlap)))
        conditional = [item.field for item in self.conditions]
        unknown = set(conditional) - self.fields
        if unknown:
            raise ValueError("conditional backend fields are not declared: " + ", ".join(sorted(unknown)))
        if len(conditional) != len(set(conditional)):
            raise ValueError("conditional backend fields must be declared exactly once")
        choice_fields = [field for field, _ in self.config_value_choices]
        if len(choice_fields) != len(set(choice_fields)) or set(choice_fields) - self.fields:
            raise ValueError("backend config value choices must name distinct declared fields")
        normalized_fields = [item.field for item in self.selector_normalizations]
        if len(normalized_fields) != len(set(normalized_fields)):
            raise ValueError("backend selector normalizations must name distinct fields")
        for item in self.selector_normalizations:
            if item.field not in self.fields:
                raise ValueError(f"backend selector normalization names undeclared field: {item.field}")
            if item.field in item.aliases or len(item.aliases) != len(set(item.aliases)):
                raise ValueError(f"backend selector normalization aliases are invalid for {item.field}")


@dataclass(frozen=True)
class BackendAdapter:
    name: str
    image_name: str
    compile: Compile
    command: Callable[[dict[str, Any]], dict[str, Any]]
    display: Callable[[dict[str, Any]], dict[str, Any]]
    requirements: Callable[..., list[dict[str, Any]]]
    surface: BackendSurface
    project_dataset: ProjectDataset | None = None
    validate_authored: Callable[[dict[str, Any]], None] | None = None
    download_specs: Callable[..., tuple[list[dict[str, Any]], dict[str, str]]] | None = None
    validate_dataset: Callable[[dict[str, Any], Path], None] | None = None
    training_state: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # Planning facts the adapter owns; plan renders them without backend tables.
    runtime_checks: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None
    disk_cache_estimate: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    general_resolution: Callable[[dict[str, Any]], Any] | None = None
    runpod_template_compatible: bool = False
    default_ports: tuple[str, ...] = ("22/tcp",)


def _compile_ai(run: dict[str, Any], resolved: Path) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_ai_toolkit(run, resolved / "ai-toolkit")


def _compile_musubi(run: dict[str, Any], resolved: Path) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_musubi_tuner(run, resolved / "musubi")


def _compile_sd_scripts(run: dict[str, Any], resolved: Path) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_sd_scripts(run, resolved / "sd-scripts")


AI_TOOLKIT_SURFACE = BackendSurface(
    fields=frozenset({
        "batch_size", "dataset_config", "dataset_options", "flatten_groups", "gradient_accumulation_steps", "gradient_checkpointing",
        "bypass_guidance_embedding", "extras_name_or_path", "learning_rate", "low_vram", "lr_scheduler", "mixed_precision", "model_arch", "model_edit",
        "network_alpha", "network_dim", "optimizer_type", "quantize", "quantize_te", "resolution",
        "save_every_n_steps", "save_last_n_steps",
    }),
    conditions=(
        _when("bypass_guidance_embedding", model_arch=("flex2",)),
        _when("extras_name_or_path", model_arch=("zimage_l2p",)),
        _when("model_edit", model_arch=("krea2",)),
    ),
    escape_hatches=frozenset({"command", "native_config"}),
    boolean_fields=frozenset({
        "bypass_guidance_embedding", "flatten_groups", "gradient_checkpointing", "low_vram", "model_edit", "quantize", "quantize_te",
    }),
    unavailable=((
        "dataset_folder",
        "AI-Toolkit backend.config.dataset_folder was replaced by the dataset manifest; "
        "list selected files in items.jsonl with role 'target', remove dataset_folder, "
        "and recompile",
    ),),
    nested_config_fields={
        "dataset_config": AI_TOOLKIT_DATASET_FIELD_SPECS,
        **AI_TOOLKIT_DATASET_OPTION_CAPABILITIES,
    },
    config_value_choices=(("model_arch", tuple(sorted(AI_TOOLKIT_PINNED_MODEL_ARCHS))),),
)

MUSUBI_SURFACE = BackendSurface(
    fields=frozenset({
        "allow_a40_large_micro_batch", "allow_a40_uncheckpointed_9b", "architecture", "batch_size",
        "block_swap_h2d_only", "block_swap_ring_size", "blocks_to_swap", "dataset_options", "discrete_flow_shift",
        "convrot_int8", "convrot_int8_bwd", "dit_dtype", "env", "f1", "flatten_groups", "fp8", "fp8_base", "fp8_llm", "fp8_scaled", "fp8_t5", "fp8_te",
        "fp8_text_encoder", "fp8_vl", "gradient_accumulation_steps", "gradient_checkpointing", "gradient_checkpointing_cpu_offload",
        "h3_guidance_loss_scale", "h3_guidance_loss_sigma_min", "h3_loss_method",
        "h3_teacher_condition_sigma_max", "h3_teacher_condition_sigma_min", "h3_teacher_conditions",
        "h3_teacher_loss_dc_weight", "h3_teacher_loss_mag_weight", "h3_teacher_preservation_weight",
        "h3_timestep_focus_max", "h3_timestep_focus_min", "h3_timestep_focus_prob",
        "include_turbo_dit", "learning_rate", "lr_scheduler", "max_data_loader_n_workers",
        "model_bundle", "model_downloads", "model_expectations", "model_paths", "model_type", "model_version",
        "network_alpha", "network_dim", "noise_clip_std", "noise_scale_end", "noise_scale_start", "one_frame",
        "one_frame_no_2x", "one_frame_no_4x", "optimizer_type", "output_compatibility",
        "pixel_cache_batch_size", "precache", "prune_checkpoints_before_step", "quantized_qwen", "resolution",
        "remove_first_image_from_target",
        "save_every_n_steps", "save_precision",
        "task", "text_encoder_batch_size", "text_encoder_blocks_to_swap", "timestep_boundary", "timestep_sampling", "vae_chunk_size", "vae_dtype",
        "use_pinned_memory_for_block_swap", "vae_tiling", "validate_models", "video_only", "weighting_scheme",
    }),
    escape_hatches=frozenset({"command", "extra_args"}),
    boolean_fields=frozenset({
        "allow_a40_large_micro_batch", "allow_a40_uncheckpointed_9b", "block_swap_h2d_only", "convrot_int8", "f1",
        "flatten_groups", "fp8", "fp8_base", "fp8_llm", "fp8_scaled", "fp8_t5", "fp8_te", "fp8_text_encoder", "fp8_vl",
        "gradient_checkpointing", "gradient_checkpointing_cpu_offload", "include_turbo_dit", "one_frame",
        "one_frame_no_2x", "one_frame_no_4x", "precache", "quantized_qwen", "remove_first_image_from_target",
        "use_pinned_memory_for_block_swap", "vae_tiling", "validate_models", "video_only",
    }),
    selector_normalizations=(SelectorNormalization(
        field="architecture",
        aliases=("model_arch",),
        normalize=canonical_musubi_architecture,
        rule="case-insensitive; hyphens become underscores; named aliases become canonical values",
        value_aliases=tuple(sorted(MUSUBI_ARCHITECTURE_ALIASES.items())),
    ),),
    selector_defaults=(("precache", True), ("one_frame", False), ("h3_loss_method", "guidance")),
    unavailable=(("mixed_precision", "Musubi training precision is fixed to bf16; save_precision controls only the saved checkpoint dtype"),),
    conditions=(
        _when("allow_a40_large_micro_batch", architecture=("flux2",)),
        _when("allow_a40_uncheckpointed_9b", architecture=("flux2",)),
        _when("convrot_int8_bwd", architecture=("krea2",), convrot_int8=(True,)),
        _when("convrot_int8", architecture=("krea2",)),
        _when("discrete_flow_shift", architecture=("wan",)),
        _when("dit_dtype", architecture=("ideogram4",)),
        _when("f1", architecture=("framepack",)),
        _when("fp8", architecture=("flux_kontext", "framepack")),
        _when("fp8_base", architecture=("flux2", "wan", "krea2", "qwen_image", "zimage", "flux_kontext", "hidream_o1", "hunyuan_video", "hunyuan_video_1_5", "framepack", "kandinsky5")),
        _when("fp8_scaled", architecture=("flux2", "krea2", "qwen_image", "zimage", "flux_kontext", "hidream_o1", "hunyuan_video_1_5", "framepack", "kandinsky5")),
        _when_any("fp8_llm", {"architecture": ("zimage", "framepack")}, {"architecture": ("hunyuan_video",), "precache": (True,)}),
        _when_any("fp8_t5", {"architecture": ("flux_kontext",)}, {"architecture": ("wan",), "precache": (True,)}),
        _when("fp8_te", architecture=("hidream_o1",), precache=(True,)),
        _when("fp8_text_encoder", architecture=("flux2",), precache=(True,)),
        _when("fp8_vl", architecture=("qwen_image", "hunyuan_video_1_5")),
        _when("gradient_checkpointing_cpu_offload", architecture=("krea2",), gradient_checkpointing=(True,)),
        _when("include_turbo_dit", architecture=("krea2",)),
        _when("model_bundle", architecture=("flux2", "krea2", "minimax_h3")),
        _when("model_type", architecture=("hidream_o1",)),
        _when("model_version", architecture=("flux2", "qwen_image")),
        _when("noise_clip_std", architecture=("hidream_o1",)),
        _when("noise_scale_end", architecture=("hidream_o1",)),
        _when("noise_scale_start", architecture=("hidream_o1",)),
        _when_any(
            "one_frame",
            {"architecture": ("wan",)},
            {"architecture": ("framepack",)},
            {"architecture": ("minimax_h3",)},
        ),
        _when("one_frame_no_2x", architecture=("framepack",), one_frame=(True,), precache=(True,)),
        _when("one_frame_no_4x", architecture=("framepack",), one_frame=(True,), precache=(True,)),
        _when("pixel_cache_batch_size", architecture=("hidream_o1",), precache=(True,)),
        _when("quantized_qwen", architecture=("kandinsky5",), precache=(True,)),
        _when("remove_first_image_from_target", architecture=("qwen_image",)),
        _when(
            "h3_guidance_loss_scale",
            architecture=("minimax_h3",),
            h3_loss_method=("guidance",),
        ),
        _when(
            "h3_guidance_loss_sigma_min",
            architecture=("minimax_h3",),
            h3_loss_method=("guidance",),
        ),
        *(
            _when(
                field,
                architecture=("minimax_h3",),
                h3_loss_method=("teacher_matching",),
            )
            for field in (
                "h3_teacher_conditions",
                "h3_teacher_condition_sigma_min",
                "h3_teacher_condition_sigma_max",
                "h3_teacher_loss_dc_weight",
                "h3_teacher_loss_mag_weight",
                "h3_teacher_preservation_weight",
                "h3_timestep_focus_min",
                "h3_timestep_focus_max",
                "h3_timestep_focus_prob",
            )
        ),
        _when("h3_loss_method", architecture=("minimax_h3",)),
        _when("task", architecture=("wan", "hidream_o1", "hunyuan_video_1_5", "kandinsky5", "minimax_h3")),
        _when("text_encoder_batch_size", precache=(True,)),
        _when("text_encoder_blocks_to_swap", architecture=("minimax_h3",), precache=(True,)),
        _when("timestep_boundary", architecture=("wan",)),
        _when("timestep_sampling", architecture=("flux2", "wan", "krea2", "hidream_o1")),
        _when("vae_chunk_size", architecture=("hunyuan_video", "framepack"), precache=(True,)),
        _when_any("vae_dtype", {"architecture": ("flux2",)}, {"architecture": ("ideogram4",), "precache": (True,)}),
        _when("vae_tiling", architecture=("hunyuan_video",), precache=(True,)),
        _when("video_only", architecture=("minimax_h3",)),
        _when("weighting_scheme", architecture=("flux2", "krea2", "qwen_image", "hidream_o1")),
    ),
    nested_config_fields=MUSUBI_DATASET_OPTION_CAPABILITIES,
)

SD_SCRIPTS_SURFACE = BackendSurface(
    fields=frozenset(CONFIG_KEYS - {"command", "extra_args"}),
    escape_hatches=frozenset({"command", "extra_args"}),
    boolean_fields=frozenset(BOOLEAN_CONFIG_KEYS),
    selector_defaults=(("mode", "lora"),),
    unavailable=(
        ("batch", "sd-scripts batch size is configured at backend.config.dataset_config.general.batch_size or backend.config.dataset_config.datasets[].batch_size"),
        ("deepspeed", "Kura's sd-scripts built-in selectors do not own deepspeed; use a reviewed backend.config.command"),
        ("fused_backward_pass", "Kura's sd-scripts built-in selectors do not own fused_backward_pass; use a reviewed backend.config.command"),
    ),
    conditions=(
        _when("attn_mode", architecture=("anima",)),
        _when("blocks_to_swap", architecture=("flux1", "anima"), mode=("lora",)),
        _when("cond_emb_dim", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("cpu_offload_checkpointing", mode=("lora",)),
        _when("discrete_flow_shift", architecture=("flux1", "anima")),
        _when("fp8_base", architecture=("sd15", "sdxl", "flux1"), mode=("lora",)),
        _when("guidance_scale", architecture=("flux1",)),
        _when("lllite_cond_dim", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_cond_in_channels", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_cond_resblocks", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_dropout", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_mlp_dim", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_multiplier", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_target_layers", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("lllite_use_aspp", architecture=("anima",), mode=("controlnet_lllite",)),
        _when("model_prediction_type", architecture=("flux1",)),
        _when("network_alpha", mode=("lora",)),
        _when("network_dim", mode=("lora",)),
        _when("network_train_unet_only", mode=("lora",)),
        _when("qwen_image_vae_2d", architecture=("anima",)),
        _when("sigmoid_scale", architecture=("flux1", "anima")),
        _when("text_encoder_lr1", architecture=("sdxl",)),
        _when("text_encoder_lr2", architecture=("sdxl",)),
        _when("timestep_sampling", architecture=("flux1", "anima")),
        _when("unet_lr", architecture=("sdxl",)),
        _when("unsloth_offload_checkpointing", mode=("lora",)),
        _when("vae_chunk_size", architecture=("anima",)),
    ),
    nested_config_fields=SD_SCRIPTS_DATASET_CAPABILITIES,
)


BACKENDS: dict[str, BackendAdapter] = {
    "ai-toolkit": BackendAdapter(
        name="ai-toolkit", image_name="ai-toolkit", compile=_compile_ai, command=command_ai_toolkit,
        display=display_ai_toolkit, requirements=requirements_ai_toolkit, surface=AI_TOOLKIT_SURFACE,
        validate_authored=validate_ai_toolkit_config,
        project_dataset=project_ai_toolkit_dataset,
        training_state=training_state_contract_ai_toolkit,
        runtime_checks=runtime_checks_ai_toolkit,
        default_ports=("8675/http", "22/tcp"),
    ),
    "musubi-tuner": BackendAdapter(
        name="musubi-tuner", image_name="musubi-tuner", compile=_compile_musubi, command=command_musubi_tuner,
        display=display_musubi_tuner, requirements=requirements_musubi, surface=MUSUBI_SURFACE,
        validate_authored=validate_musubi_authored_config,
        project_dataset=project_musubi_dataset,
        download_specs=musubi_model_download_specs,
        training_state=training_state_contract_musubi,
        runtime_checks=runtime_checks_musubi,
        general_resolution=musubi_general_resolution,
    ),
    "sd-scripts": BackendAdapter(
        name="sd-scripts", image_name="sd-scripts", compile=_compile_sd_scripts, command=command_sd_scripts,
        display=display_sd_scripts, requirements=requirements_sd_scripts, surface=SD_SCRIPTS_SURFACE,
        project_dataset=project_sd_scripts_dataset,
        validate_authored=validate_sd_scripts_dataset_config,
        download_specs=sd_scripts_model_download_specs,
        training_state=training_state_contract_sd_scripts,
        disk_cache_estimate=sd_scripts_disk_cache_estimate,
    ),
}


def backend_names() -> tuple[str, ...]:
    return tuple(BACKENDS)


def get_backend(name: Any) -> BackendAdapter:
    if not isinstance(name, str) or name not in BACKENDS:
        raise ValueError(f"unsupported backend: {name}")
    return BACKENDS[name]


_GENERAL_ML_ALIASES = {
    "batch": "batch_size",
    "lr": "learning_rate",
    "model_arch": "architecture",
    "optimizer": "optimizer_type",
    "output_format": "output_compatibility",
    "rank": "network_dim",
    "scheduler": "lr_scheduler",
}

_GENERAL_UNAVAILABLE = {
    "epochs": "Kura training recipes are step-based; use recipe.steps",
}


def validate_backend_config(run: dict[str, Any]) -> None:
    """Reject authored values that have no declared home on the selected adapter."""

    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    adapter = get_backend(backend.get("name"))
    native = backend_config(run, adapter.name)
    selector_aliases = {
        alias
        for normalization in adapter.surface.selector_normalizations
        for alias in normalization.aliases
    }
    accepted = adapter.surface.fields | adapter.surface.escape_hatches | selector_aliases
    unknown = sorted(set(native) - accepted)
    details: list[str] = []
    unavailable = {**_GENERAL_UNAVAILABLE, **dict(adapter.surface.unavailable)}
    for key in unknown:
        if key in unavailable:
            details.append(f"{key!r}: {unavailable[key]}")
            continue
        suggestion = _GENERAL_ML_ALIASES.get(key)
        if suggestion not in accepted:
            suggestion = None
        details.append(f"{key!r}; use {suggestion!r}" if suggestion else repr(key))
    if details:
        raise ValueError(
            f"{adapter.name} backend.config contains unsupported key(s): " + ", ".join(details)
            + f". Run `kura run capabilities {adapter.name}` for accepted fields."
        )
    not_boolean = sorted(key for key in adapter.surface.boolean_fields if key in native and not isinstance(native[key], bool))
    if not_boolean:
        raise ValueError(
            f"{adapter.name} backend.config field(s) must be true or false: "
            + ", ".join(f"{key}={native[key]!r}" for key in not_boolean)
        )
    resolved = dict(native)
    for normalization in adapter.surface.selector_normalizations:
        authored = [
            (field, native[field])
            for field in (normalization.field, *normalization.aliases)
            if field in native
        ]
        if len(authored) > 1:
            raise ValueError(
                f"{adapter.name} backend.config selector {normalization.field!r} "
                "must use exactly one authored field; found "
                + ", ".join(repr(field) for field, _ in authored)
            )
        if authored:
            authored_field, authored_value = authored[0]
            if not isinstance(authored_value, str):
                raise ValueError(
                    f"{adapter.name} backend.config.{authored_field} must be a string"
                )
            resolved[normalization.field] = normalization.normalize(authored_value)
    defaults = dict(adapter.surface.selector_defaults)
    for condition in adapter.surface.conditions:
        if condition.field not in native:
            continue
        matched = False
        resolved_clauses: list[str] = []
        selector_missing = False
        for clause in condition.when_any:
            clause_matches = True
            labels: list[str] = []
            for selector, allowed in clause:
                value = resolved.get(selector, defaults.get(selector))
                if value is None:
                    selector_missing = True
                    clause_matches = False
                elif value not in allowed:
                    clause_matches = False
                labels.append(f"{selector}=" + "|".join(repr(item) for item in allowed))
            matched = matched or clause_matches
            resolved_clauses.append(" and ".join(labels))
        if not matched and not selector_missing:
            selected = ", ".join(
                f"{selector}={resolved.get(selector, defaults.get(selector))!r}"
                for selector in sorted({name for clause in condition.when_any for name, _ in clause})
            )
            raise ValueError(
                f"{adapter.name} backend.config.{condition.field} is not applicable for {selected}; "
                f"it requires " + " or ".join(resolved_clauses)
                + f". Run `kura run capabilities {adapter.name}` for field applicability."
            )
    if adapter.validate_authored is not None:
        adapter.validate_authored(run)


def backend_capabilities(name: Any) -> dict[str, Any]:
    adapter = get_backend(name)
    conditional = {item.field for item in adapter.surface.conditions}
    return {
        "backend": adapter.name,
        "common_recipe_fields": sorted(COMMON_RECIPE_FIELDS),
        "config_fields": sorted(adapter.surface.fields - conditional),
        "conditional_fields": {
            item.field: {
                "when_any": [
                    {selector: list(allowed) for selector, allowed in clause}
                    for clause in item.when_any
                ]
            }
            for item in adapter.surface.conditions
        },
        "unsupported_fields": {**_GENERAL_UNAVAILABLE, **dict(adapter.surface.unavailable)},
        "escape_hatches": {
            key: {"validation": "unverified", "recorded": True}
            for key in sorted(adapter.surface.escape_hatches)
        },
        "nested_config_fields": deepcopy(adapter.surface.nested_config_fields or {}),
        "config_value_choices": {field: list(values) for field, values in adapter.surface.config_value_choices},
        "boolean_fields": sorted(adapter.surface.boolean_fields),
        "selector_aliases": {
            item.field: {
                "fields": list(item.aliases),
                "values": dict(item.value_aliases),
                "normalization": item.rule,
            }
            for item in adapter.surface.selector_normalizations
        },
    }
