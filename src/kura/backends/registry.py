"""Single registry for backend adapter ownership and dispatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kura.backends.ai_toolkit import command_ai_toolkit, compile_ai_toolkit, display_ai_toolkit, requirements_ai_toolkit
from kura.backends.musubi_command import command_musubi_tuner, compile_musubi_tuner, display_musubi_tuner
from kura.backends.musubi_models import requirements_musubi
from kura.backends.musubi_models import musubi_model_download_specs
from kura.backends.musubi_datasets import validate_musubi_dataset_layout
from kura.backends.sd_scripts import CONFIG_KEYS, command_sd_scripts, compile_sd_scripts, display_sd_scripts
from kura.backends.sd_scripts_datasets import SD_SCRIPTS_DATASET_CAPABILITIES, validate_sd_scripts_dataset_config
from kura.backends.sd_scripts_models import requirements_sd_scripts, sd_scripts_model_download_specs
from kura.run_envelope import COMMON_RECIPE_FIELDS, backend_config


Compile = Callable[[dict[str, Any], Path, Path | None, bool], dict[str, Any]]


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
class BackendSurface:
    """The adapter-owned authoring vocabulary accepted by Kura."""

    fields: frozenset[str]
    escape_hatches: frozenset[str] = frozenset()
    conditions: tuple[FieldCondition, ...] = ()
    selector_defaults: tuple[tuple[str, Any], ...] = ()
    unavailable: tuple[tuple[str, str], ...] = ()
    nested_config_fields: dict[str, dict[str, dict[str, Any]]] | None = None

    def __post_init__(self) -> None:
        overlap = self.fields & self.escape_hatches
        if overlap:
            raise ValueError("backend surface fields and escape hatches overlap: " + ", ".join(sorted(overlap)))
        conditional = [item.field for item in self.conditions]
        unknown = set(conditional) - self.fields
        if unknown:
            raise ValueError("conditional backend fields are not declared: " + ", ".join(sorted(unknown)))
        if len(conditional) != len(set(conditional)):
            raise ValueError("conditional backend fields must be declared exactly once")


@dataclass(frozen=True)
class BackendAdapter:
    name: str
    image_name: str
    compile: Compile
    command: Callable[[dict[str, Any]], dict[str, Any]]
    display: Callable[[dict[str, Any]], dict[str, Any]]
    requirements: Callable[..., list[dict[str, Any]]]
    surface: BackendSurface
    validate_authored: Callable[[dict[str, Any]], None] | None = None
    download_specs: Callable[..., tuple[list[dict[str, Any]], dict[str, str]]] | None = None
    validate_dataset: Callable[[dict[str, Any], Path], None] | None = None
    runpod_template_compatible: bool = False
    default_ports: tuple[str, ...] = ("22/tcp",)


def _compile_ai(run: dict[str, Any], resolved: Path, workspace: Path | None, strict: bool) -> dict[str, Any]:
    del workspace, strict
    validate_backend_config(run)
    return compile_ai_toolkit(run, resolved / "ai-toolkit")


def _compile_musubi(run: dict[str, Any], resolved: Path, workspace: Path | None, strict: bool) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_musubi_tuner(run, resolved / "musubi", workspace=workspace, strict=strict)


def _compile_sd_scripts(run: dict[str, Any], resolved: Path, workspace: Path | None, strict: bool) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_sd_scripts(run, resolved / "sd-scripts", workspace=workspace, strict=strict)


AI_TOOLKIT_SURFACE = BackendSurface(
    fields=frozenset({
        "batch_size", "dataset_folder", "gradient_accumulation_steps", "gradient_checkpointing",
        "learning_rate", "low_vram", "lr_scheduler", "mixed_precision", "model_arch",
        "network_alpha", "network_dim", "optimizer_type", "quantize", "quantize_te", "resolution",
        "save_every_n_steps", "save_last_n_steps",
    }),
    escape_hatches=frozenset({"command", "native_config"}),
)

MUSUBI_SURFACE = BackendSurface(
    fields=frozenset({
        "allow_a40_large_micro_batch", "allow_a40_uncheckpointed_9b", "architecture", "batch_size", "blocks_to_swap", "discrete_flow_shift",
        "dit_dtype", "env", "f1", "fp8", "fp8_base", "fp8_llm", "fp8_scaled", "fp8_t5", "fp8_te",
        "fp8_text_encoder", "fp8_vl", "gradient_accumulation_steps", "gradient_checkpointing",
        "include_turbo_dit", "learning_rate", "lr_scheduler", "max_data_loader_n_workers",
        "model_bundle", "model_downloads", "model_expectations", "model_paths", "model_type", "model_version",
        "network_alpha", "network_dim", "noise_clip_std", "noise_scale_end", "noise_scale_start", "one_frame",
        "one_frame_no_2x", "one_frame_no_4x", "optimizer_type", "output_compatibility",
        "pixel_cache_batch_size", "precache", "prune_checkpoints_before_step", "quantized_qwen", "resolution",
        "save_every_n_steps", "save_precision",
        "task", "text_encoder_batch_size", "timestep_boundary", "timestep_sampling", "vae_chunk_size", "vae_dtype",
        "vae_tiling", "validate_models", "weighting_scheme",
    }),
    escape_hatches=frozenset({"command", "dataset_config", "extra_args"}),
    selector_defaults=(("precache", True), ("one_frame", False)),
    unavailable=(("mixed_precision", "Musubi training precision is fixed to bf16; save_precision controls only the saved checkpoint dtype"),),
    conditions=(
        _when("allow_a40_large_micro_batch", architecture=("flux2", "flux_2")),
        _when("allow_a40_uncheckpointed_9b", architecture=("flux2", "flux_2")),
        _when("discrete_flow_shift", architecture=("wan",)),
        _when("dit_dtype", architecture=("ideogram4", "ideogram_4")),
        _when("f1", architecture=("framepack", "frame_pack")),
        _when("fp8", architecture=("flux_kontext", "flux1_kontext", "framepack", "frame_pack")),
        _when("fp8_base", architecture=("flux2", "flux_2", "wan", "krea2", "krea_2", "qwen_image", "qwen", "zimage", "z_image", "flux_kontext", "flux1_kontext", "hidream_o1", "hidream", "hunyuan_video", "hunyuanvideo", "hunyuan_video_1_5", "framepack", "frame_pack", "kandinsky5", "kandinsky_5")),
        _when("fp8_scaled", architecture=("flux2", "flux_2", "krea2", "krea_2", "qwen_image", "qwen", "zimage", "z_image", "flux_kontext", "flux1_kontext", "hidream_o1", "hidream", "hunyuan_video_1_5", "framepack", "frame_pack", "kandinsky5", "kandinsky_5")),
        _when_any("fp8_llm", {"architecture": ("zimage", "z_image", "framepack", "frame_pack")}, {"architecture": ("hunyuan_video", "hunyuanvideo"), "precache": (True,)}),
        _when_any("fp8_t5", {"architecture": ("flux_kontext", "flux1_kontext")}, {"architecture": ("wan",), "precache": (True,)}),
        _when("fp8_te", architecture=("hidream_o1", "hidream"), precache=(True,)),
        _when("fp8_text_encoder", architecture=("flux2", "flux_2"), precache=(True,)),
        _when("fp8_vl", architecture=("qwen_image", "qwen", "hunyuan_video_1_5")),
        _when("include_turbo_dit", architecture=("krea2", "krea_2")),
        _when("model_bundle", architecture=("flux2", "flux_2", "krea2", "krea_2")),
        _when("model_type", architecture=("hidream_o1", "hidream")),
        _when("model_version", architecture=("flux2", "flux_2", "qwen_image", "qwen")),
        _when("noise_clip_std", architecture=("hidream_o1", "hidream")),
        _when("noise_scale_end", architecture=("hidream_o1", "hidream")),
        _when("noise_scale_start", architecture=("hidream_o1", "hidream")),
        _when_any("one_frame", {"architecture": ("wan",)}, {"architecture": ("framepack", "frame_pack")}),
        _when("one_frame_no_2x", architecture=("framepack", "frame_pack"), one_frame=(True,), precache=(True,)),
        _when("one_frame_no_4x", architecture=("framepack", "frame_pack"), one_frame=(True,), precache=(True,)),
        _when("pixel_cache_batch_size", architecture=("hidream_o1", "hidream"), precache=(True,)),
        _when("quantized_qwen", architecture=("kandinsky5", "kandinsky_5"), precache=(True,)),
        _when("task", architecture=("wan", "hidream_o1", "hidream", "hunyuan_video_1_5", "kandinsky5", "kandinsky_5")),
        _when("text_encoder_batch_size", precache=(True,)),
        _when("timestep_boundary", architecture=("wan",)),
        _when("timestep_sampling", architecture=("flux2", "flux_2", "wan", "krea2", "krea_2", "hidream_o1", "hidream")),
        _when("vae_chunk_size", architecture=("hunyuan_video", "hunyuanvideo", "framepack", "frame_pack"), precache=(True,)),
        _when_any("vae_dtype", {"architecture": ("flux2", "flux_2")}, {"architecture": ("ideogram4", "ideogram_4"), "precache": (True,)}),
        _when("vae_tiling", architecture=("hunyuan_video", "hunyuanvideo"), precache=(True,)),
        _when("weighting_scheme", architecture=("flux2", "flux_2", "krea2", "krea_2", "qwen_image", "qwen", "hidream_o1", "hidream")),
    ),
)

SD_SCRIPTS_SURFACE = BackendSurface(
    fields=frozenset(CONFIG_KEYS - {"command", "extra_args", "deepspeed", "fused_backward_pass"}),
    escape_hatches=frozenset({"command", "extra_args"}),
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
        default_ports=("8675/http", "22/tcp"),
    ),
    "musubi-tuner": BackendAdapter(
        name="musubi-tuner", image_name="musubi-tuner", compile=_compile_musubi, command=command_musubi_tuner,
        display=display_musubi_tuner, requirements=requirements_musubi, surface=MUSUBI_SURFACE,
        download_specs=musubi_model_download_specs, validate_dataset=validate_musubi_dataset_layout,
    ),
    "sd-scripts": BackendAdapter(
        name="sd-scripts", image_name="sd-scripts", compile=_compile_sd_scripts, command=command_sd_scripts,
        display=display_sd_scripts, requirements=requirements_sd_scripts, surface=SD_SCRIPTS_SURFACE,
        validate_authored=validate_sd_scripts_dataset_config,
        download_specs=sd_scripts_model_download_specs,
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
    accepted = adapter.surface.fields | adapter.surface.escape_hatches
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
                value = native.get(selector, defaults.get(selector))
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
                f"{selector}={native.get(selector, defaults.get(selector))!r}"
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
    }
