"""Single registry for backend adapter ownership and dispatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kura.backends.ai_toolkit import AI_TOOLKIT_ARCHITECTURE_REQUIREMENTS, AI_TOOLKIT_DATASET_FIELD_SPECS, AI_TOOLKIT_DATASET_OPTION_CAPABILITIES, AI_TOOLKIT_MODEL_ARCH_VALUES, AI_TOOLKIT_OPTIMIZER_DECIDED, AI_TOOLKIT_PINNED_MODEL_ARCHS, AI_TOOLKIT_PROFILE_REQUIREMENTS, ai_toolkit_model_facts, command_ai_toolkit, ai_toolkit_disk_cache, compile_ai_toolkit, display_ai_toolkit, project_ai_toolkit_dataset, requirements_ai_toolkit, runtime_checks_ai_toolkit, training_state_contract_ai_toolkit, training_state_contract_ai_toolkit_before_frozen_decision, validate_ai_toolkit_config
from kura.backends.ai_toolkit_baseline import trained_arch, ui_entry_values
from kura.backends.common import MUSUBI_ARCHITECTURES
from kura.backends.musubi_command import MUSUBI_RESOLVED_DIR, command_musubi_tuner, musubi_disk_cache, compile_musubi_tuner, display_musubi_tuner, training_state_contract_musubi
from kura.backends.musubi_models import requirements_musubi
from kura.backends.musubi_models import MODEL_VERSIONS, MUSUBI_BUNDLES, musubi_model_download_specs, musubi_model_facts
from kura.backends.musubi_datasets import MUSUBI_DATASET_OPTION_CAPABILITIES, musubi_general_resolution, project_musubi_dataset, runtime_checks_musubi, validate_musubi_authored_config
from kura.backends.sd_scripts import BOOLEAN_CONFIG_KEYS, CONFIG_KEYS, ENTRYPOINTS, command_sd_scripts, compile_sd_scripts, display_sd_scripts, sd_scripts_disk_cache, training_state_contract_sd_scripts, validate_sd_scripts_authored
from kura.backends.sd_scripts_datasets import SD_SCRIPTS_DATASET_CAPABILITIES, project_sd_scripts_dataset
from kura.backends.sd_scripts_models import requirements_sd_scripts, sd_scripts_model_download_specs, sd_scripts_model_facts
from kura.images import BUILD_SOURCES, PINNED_IMAGES
from kura.run_envelope import COMMON_RECIPE_FIELDS, RETIRED_TRAIN_RUN_KEYS, SettingsRefused, backend_config
from kura.training_artifacts import LR_SCHEDULER_DECIDED


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
class RequiredField:
    """A field an author must write when a selector clause matches: it changes training quality,
    and the trainer would otherwise fall back to `otherwise`, a generic default no one chose for
    this model (read in the pinned trainer source, cited there)."""

    field: str
    when_any: tuple[tuple[tuple[str, tuple[Any, ...]], ...], ...]
    otherwise: str


def _required(field: str, otherwise: str, *clauses: dict[str, tuple[Any, ...]]) -> RequiredField:
    """With no clause, the field is required wherever it applies: always, or where its condition matches."""
    return RequiredField(field, tuple(tuple(clause.items()) for clause in clauses), otherwise)


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
    # Selector fields Kura decides from, with their one accepted spelling each; checked first. The
    # first is the architecture selector `kura run capabilities --arch` filters by.
    config_value_choices: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Integer fields Kura decides from and the least value each accepts; anything else is refused.
    integer_minimums: tuple[tuple[str, int], ...] = ()
    # Other fields Kura decides from, besides the selectors above, each with what Kura does with it
    # (a constant its owner shares); escape hatches may not set them.
    decided: tuple[tuple[str, str], ...] = ()
    # Fields with no upstream baseline that an author must write; Kura adds no training default.
    required: tuple[RequiredField, ...] = ()
    # Whether the backend hands fields to its trainer as argv, where true and false can only mean a
    # bare flag or none (AI-Toolkit writes them into its YAML config instead).
    renders_argv: bool = False

    @property
    def decided_fields(self) -> frozenset[str]:
        return {field for field, _ in self.decided} | {field for field, _ in self.config_value_choices} | {field for field, _ in self.integer_minimums}

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
        if self.decided_fields - self.fields:
            raise ValueError("decided backend fields are not declared: " + ", ".join(sorted(self.decided_fields - self.fields)))
        required = [item.field for item in self.required]
        if len(required) != len(set(required)) or set(required) - self.fields:
            raise ValueError("required backend fields must name distinct declared fields")
        applies = {item.field: item.when_any for item in self.conditions}
        object.__setattr__(self, "required", tuple(
            item if item.when_any else RequiredField(item.field, applies.get(item.field, ((),)), item.otherwise)
            for item in self.required
        ))
        choice_fields = [field for field, _ in self.config_value_choices]
        if len(choice_fields) != len(set(choice_fields)) or set(choice_fields) - self.fields:
            raise ValueError("backend config value choices must name distinct declared fields")


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
    # The contract a run whose command lock predates the frozen `training_state_managed` was
    # compiled under, where it differs from `training_state`.
    legacy_training_state: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # Planning facts the adapter owns; plan renders them without backend tables.
    runtime_checks: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None
    # What run-scoped disk cache the trainer writes (a phrase), or None; plan shows it for every
    # backend and does not count it (the disk watch covers it).
    disk_cache: Callable[[dict[str, Any]], str | None] | None = None
    general_resolution: Callable[[dict[str, Any]], Any] | None = None
    # Model facts the adapter already owns (baselines, model roles, entry shapes), for capabilities:
    # given an architecture selector value, only that architecture's.
    model_facts: Callable[[str | None], dict[str, Any]] | None = None
    default_ports: tuple[str, ...] = ("22/tcp",)


def _compile_ai(run: dict[str, Any], resolved: Path) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_ai_toolkit(run, resolved / "ai-toolkit")


def _compile_musubi(run: dict[str, Any], resolved: Path) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_musubi_tuner(run, resolved / MUSUBI_RESOLVED_DIR)


def _compile_sd_scripts(run: dict[str, Any], resolved: Path) -> dict[str, Any]:
    validate_backend_config(run)
    return compile_sd_scripts(run, resolved / "sd-scripts")


def _ai_toolkit_arch_values(arch: str) -> tuple[str, ...]:
    """Every model_arch value that trains `arch`: the arch and its tagged UI entries."""
    return (arch, *(name for name in ui_entry_values(AI_TOOLKIT_PINNED_MODEL_ARCHS) if trained_arch(name) == arch))


def _ai_toolkit_required_field_conditions() -> tuple[FieldCondition, ...]:
    """A field an architecture's requirements name applies to that architecture only: every
    model_arch value that trains an arch in AI_TOOLKIT_ARCHITECTURE_REQUIREMENTS, or in a profile
    of AI_TOOLKIT_PROFILE_REQUIREMENTS, that names the field."""
    archs: dict[str, list[str]] = {}
    requirements = [*AI_TOOLKIT_ARCHITECTURE_REQUIREMENTS.items(), *((arch, fields) for (arch, _profile), fields in AI_TOOLKIT_PROFILE_REQUIREMENTS.items())]
    for arch, fields in requirements:
        for field in fields:
            if arch not in archs.setdefault(field, []):
                archs[field].append(arch)
    return tuple(_when(field, model_arch=tuple(value for arch in names for value in _ai_toolkit_arch_values(arch))) for field, names in archs.items())


AI_TOOLKIT_SURFACE = BackendSurface(
    fields=frozenset({
        "batch_size", "dataset_config", "dataset_options", "flatten_groups", "gradient_accumulation_steps", "gradient_checkpointing",
        "bypass_guidance_embedding", "extras_name_or_path", "learning_rate", "low_vram", "lr_scheduler", "mixed_precision", "model_arch", "model_edit",
        "network_alpha", "network_dim", "optimizer_type", "quantize", "quantize_te", "resolution",
        "save_every_n_steps", "save_last_n_steps",
    }),
    conditions=_ai_toolkit_required_field_conditions(),
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
    config_value_choices=(("model_arch", AI_TOOLKIT_MODEL_ARCH_VALUES),),
    # A max_step_saves_to_keep of 0 keeps every save in the pinned trainer. Its -1 accumulation
    # (a whole epoch) is refused: Kura counts and resumes by optimizer steps.
    integer_minimums=(("save_every_n_steps", 1), ("save_last_n_steps", 0), ("gradient_accumulation_steps", 1)),
    decided=(("lr_scheduler", LR_SCHEDULER_DECIDED), ("optimizer_type", AI_TOOLKIT_OPTIMIZER_DECIDED)),
)

MUSUBI_SURFACE = BackendSurface(
    fields=frozenset({
        "architecture", "batch_size",
        "block_swap_h2d_only", "block_swap_ring_size", "blocks_to_swap", "dataset_options", "discrete_flow_shift",
        "convrot_int8", "convrot_int8_bwd", "dit_dtype", "env", "f1", "flatten_groups", "fp8_base", "fp8_llm", "fp8_scaled", "fp8_t5", "fp8_te",
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
    renders_argv=True,
    boolean_fields=frozenset({
        "block_swap_h2d_only", "convrot_int8", "f1",
        "flatten_groups", "fp8_base", "fp8_llm", "fp8_scaled", "fp8_t5", "fp8_te", "fp8_text_encoder", "fp8_vl",
        "gradient_checkpointing", "gradient_checkpointing_cpu_offload", "include_turbo_dit", "one_frame",
        "one_frame_no_2x", "one_frame_no_4x", "precache", "quantized_qwen", "remove_first_image_from_target",
        "use_pinned_memory_for_block_swap", "vae_tiling", "validate_models", "video_only",
    }),
    integer_minimums=(("save_every_n_steps", 1), ("gradient_accumulation_steps", 1), ("prune_checkpoints_before_step", 1)),
    decided=(("lr_scheduler", LR_SCHEDULER_DECIDED),),
    # Generic defaults of the pinned trainer (v0.3.6), shared by every architecture.
    required=(
        _required("learning_rate", "2e-6 (training/parser_common.py:323)"),
        _required("optimizer_type", "'', read as AdamW (training/parser_common.py:310)"),
        _required("network_dim", "4 (networks/lora.py:428)"),
        _required("network_alpha", "1 (training/parser_common.py:561)"),
        # timestep_sampling, discrete_flow_shift, and weighting_scheme are shared flags every
        # Musubi trainer reads (training/parser_common.py, trainer_base.py:648), so they are
        # declared for every architecture. Upstream's guides train these architectures with
        # shift, flux_shift, or the model's own sampling and a model-specific shift
        # (docs/<architecture>.md), not the generic sigma and 1.0. Ideogram 4 sets its own
        # sampling (ideogram4_train_network.py:364) and MiniMax-H3 its own (minimax_h3_train_network.py:1725).
        _required(
            "timestep_sampling", "sigma (training/parser_common.py:444)",
            {"architecture": ("flux2", "wan", "krea2", "hidream_o1", "qwen_image", "zimage", "flux_kontext", "hunyuan_video", "hunyuan_video_1_5", "framepack", "kandinsky5")},
        ),
        _required(
            "discrete_flow_shift", "1.0 (training/parser_common.py:464)",
            {"architecture": ("wan", "qwen_image", "zimage", "hunyuan_video", "hunyuan_video_1_5", "framepack", "kandinsky5")},
        ),
        _required("h3_guidance_loss_scale", "0.0, no guidance loss (minimax_h3_train_network.py:1750)"),
        _required("h3_guidance_loss_sigma_min", "0.0 (minimax_h3_train_network.py:1764)"),
    ),
    config_value_choices=(
        ("architecture", tuple(sorted(MUSUBI_ARCHITECTURES))),
        ("h3_loss_method", ("guidance", "teacher_matching", "training_adapter")),
        ("model_bundle", tuple(MUSUBI_BUNDLES)),
        ("model_version", tuple(version for versions in MODEL_VERSIONS.values() for version in versions)),
        ("output_compatibility", ("comfyui", "none")),
    ),
    selector_defaults=(
        ("precache", True), ("one_frame", False), ("h3_loss_method", "guidance"),
        ("convrot_int8", False), ("gradient_checkpointing", False),
    ),
    unavailable=(
        ("mixed_precision", "Musubi training precision is fixed to bf16; save_precision controls only the saved checkpoint dtype"),
        ("fp8", "use fp8_base; Musubi Tuner's --fp8 belongs to its generation scripts, not training"),
    ),
    conditions=(
        _when("convrot_int8_bwd", architecture=("krea2",), convrot_int8=(True,)),
        _when("convrot_int8", architecture=("krea2",)),
        _when("dit_dtype", architecture=("ideogram4",)),
        _when("f1", architecture=("framepack",)),
        _when("fp8_base", architecture=("flux2", "wan", "krea2", "qwen_image", "zimage", "flux_kontext", "hidream_o1", "hunyuan_video", "hunyuan_video_1_5", "framepack", "kandinsky5")),
        _when("fp8_scaled", architecture=("flux2", "krea2", "qwen_image", "zimage", "flux_kontext", "hidream_o1", "hunyuan_video_1_5", "framepack", "kandinsky5")),
        _when_any("fp8_llm", {"architecture": ("zimage", "framepack")}, {"architecture": ("hunyuan_video",), "precache": (True,)}),
        _when_any("fp8_t5", {"architecture": ("flux_kontext",)}, {"architecture": ("wan",), "precache": (True,)}),
        _when("fp8_te", architecture=("hidream_o1",), precache=(True,)),
        _when("fp8_text_encoder", architecture=("flux2",), precache=(True,)),
        _when("fp8_vl", architecture=("qwen_image", "hunyuan_video_1_5")),
        _when("gradient_checkpointing_cpu_offload", architecture=("krea2",), gradient_checkpointing=(True,)),
        _when("include_turbo_dit", architecture=("krea2",), model_bundle=("krea2-raw",)),
        _when("model_bundle", architecture=tuple(dict.fromkeys(architecture for architecture, _ in MUSUBI_BUNDLES.values()))),
        _when("model_type", architecture=("hidream_o1",)),
        _when("model_version", architecture=tuple(MODEL_VERSIONS)),
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
        _when("vae_chunk_size", architecture=("hunyuan_video", "framepack"), precache=(True,)),
        _when_any("vae_dtype", {"architecture": ("flux2",)}, {"architecture": ("ideogram4",), "precache": (True,)}),
        _when("vae_tiling", architecture=("hunyuan_video",), precache=(True,)),
        _when("video_only", architecture=("minimax_h3",)),
    ),
    nested_config_fields=MUSUBI_DATASET_OPTION_CAPABILITIES,
)

SD_SCRIPTS_SURFACE = BackendSurface(
    fields=frozenset(CONFIG_KEYS - {"command", "extra_args"}),
    escape_hatches=frozenset({"command", "extra_args"}),
    renders_argv=True,
    boolean_fields=frozenset(BOOLEAN_CONFIG_KEYS),
    integer_minimums=(("save_every_n_steps", 1), ("save_last_n_steps", 1), ("gradient_accumulation_steps", 1)),
    decided=(("lr_scheduler", LR_SCHEDULER_DECIDED),),
    # Defaults of the pinned trainer (37a1cbb) that no one chose for the model: shared ones, and
    # FLUX.1 and Anima LLLite flow values whose argparse defaults differ from what upstream's own
    # guides train with (docs/flux_train_network.md:144-146, docs/anima_train_control_net_lllite.md:112-113).
    # Anima LoRA flow values and the LLLite network values left unset take the trainer's own defaults.
    required=(
        _required("learning_rate", "2e-6 (library/args.py:87)"),
        _required("optimizer_type", "AdamW (library/optimizer.py:53)"),
        _required("mixed_precision", "no, full fp32 (library/args.py:367)"),
        _required("network_dim", "4 (networks/lora.py:429)", {"mode": ("lora",)}),
        _required("network_alpha", "1 (train_network.py:1926)", {"mode": ("lora",)}),
        _required(
            "timestep_sampling", "sigma for FLUX.1 (library/flux_train_utils.py:780), sigmoid for Anima LLLite (library/anima_train_utils.py:103)",
            {"architecture": ("flux1",)}, {"architecture": ("anima",), "mode": ("controlnet_lllite",)},
        ),
        _required("guidance_scale", "3.5 (library/flux_train_utils.py:773)", {"architecture": ("flux1",)}),
        _required("model_prediction_type", "sigma_scaled (library/flux_train_utils.py:793)", {"architecture": ("flux1",)}),
        _required("discrete_flow_shift", "1.0 (library/anima_train_utils.py:97)", {"architecture": ("anima",), "mode": ("controlnet_lllite",)}),
    ),
    config_value_choices=(
        ("architecture", tuple(sorted({architecture for architecture, _ in ENTRYPOINTS}))),
        ("mode", tuple(sorted({mode for _, mode in ENTRYPOINTS}))),
    ),
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
        _when("text_encoder_lr", architecture=("sdxl",)),
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
        legacy_training_state=training_state_contract_ai_toolkit_before_frozen_decision,
        runtime_checks=runtime_checks_ai_toolkit,
        disk_cache=ai_toolkit_disk_cache,
        model_facts=ai_toolkit_model_facts,
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
        disk_cache=musubi_disk_cache,
        general_resolution=musubi_general_resolution,
        model_facts=musubi_model_facts,
    ),
    "sd-scripts": BackendAdapter(
        name="sd-scripts", image_name="sd-scripts", compile=_compile_sd_scripts, command=command_sd_scripts,
        display=display_sd_scripts, requirements=requirements_sd_scripts, surface=SD_SCRIPTS_SURFACE,
        project_dataset=project_sd_scripts_dataset,
        validate_authored=validate_sd_scripts_authored,
        download_specs=sd_scripts_model_download_specs,
        training_state=training_state_contract_sd_scripts,
        disk_cache=sd_scripts_disk_cache,
        model_facts=sd_scripts_model_facts,
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
    "optimizer": "optimizer_type",
    "output_format": "output_compatibility",
    "rank": "network_dim",
    "scheduler": "lr_scheduler",
}

_GENERAL_UNAVAILABLE = {
    "epochs": "Kura training recipes are step-based; use recipe.steps",
    "disk_cache_estimate_gb": "a disk cache is not sized by a declared estimate; the disk watch stops a local run before a store falls below docker.min_free_gb, so remove it",
}


def validate_backend_config(run: dict[str, Any], *, frozen: bool = False) -> None:
    """Refuse authored values that have no declared home on the selected adapter (see _validate_backend_config)."""
    try:
        _validate_backend_config(run, frozen=frozen)
    except SettingsRefused:
        raise
    except ValueError as exc:
        raise SettingsRefused(str(exc)) from exc


def _validate_backend_config(run: dict[str, Any], *, frozen: bool = False) -> None:
    """Reject authored values that have no declared home on the selected adapter.

    `frozen` reads a manifest an earlier compile accepted: its keys and values are checked as
    they are, but not values that became required, or the dataset settings an older compile
    accepted in another form.
    """

    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    adapter = get_backend(backend.get("name"))
    native = backend_config(run, adapter.name)
    accepted = adapter.surface.fields | adapter.surface.escape_hatches
    if frozen:
        # A manifest an older Kura compiled may still hold a key that Kura has since retired.
        accepted |= {path.removeprefix("backend.config.") for path in RETIRED_TRAIN_RUN_KEYS if path.startswith("backend.config.")}
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
    empty = sorted(key for key, value in native.items() if value is None)
    if empty:
        raise ValueError(
            f"{adapter.name} backend.config key(s) written with no value: " + ", ".join(empty)
            + "; write a value, or remove the key to leave it unset"
        )
    for field, choices in adapter.surface.config_value_choices:
        if field in native and native[field] not in choices:
            raise ValueError(
                f"{adapter.name} backend.config.{field} {native[field]!r} is not accepted; use one of: "
                + ", ".join(choices) + f". Run `kura run capabilities {adapter.name}` for accepted values."
            )
    not_boolean = sorted(key for key in adapter.surface.boolean_fields if key in native and not isinstance(native[key], bool))
    if not_boolean:
        raise ValueError(
            f"{adapter.name} backend.config field(s) must be true or false: "
            + ", ".join(f"{key}={native[key]!r}" for key in not_boolean)
        )
    for field, minimum in adapter.surface.integer_minimums:
        value = native.get(field)
        if field in native and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
            raise ValueError(
                f"{adapter.name} backend.config.{field} must be an integer of at least {minimum}, not {value!r}"
            )
    defaults = dict(adapter.surface.selector_defaults)

    def matches(clauses: tuple[tuple[tuple[str, tuple[Any, ...]], ...], ...]) -> bool:
        return any(all(native.get(name, defaults.get(name)) in allowed for name, allowed in clause) for clause in clauses)

    def selected(clauses: tuple[tuple[tuple[str, tuple[Any, ...]], ...], ...]) -> str:
        return ", ".join(
            f"{selector}={native.get(selector, defaults.get(selector))!r}"
            for selector in sorted({name for clause in clauses for name, _ in clause})
        )

    for condition in adapter.surface.conditions:
        if condition.field in native and not matches(condition.when_any):
            resolved_clauses = [
                " and ".join(f"{selector}=" + "|".join(repr(item) for item in allowed) for selector, allowed in clause)
                for clause in condition.when_any
            ]
            unset = sorted({name for clause in condition.when_any for name, _ in clause if native.get(name, defaults.get(name)) is None})
            raise ValueError(
                f"{adapter.name} backend.config.{condition.field} is not applicable for {selected(condition.when_any)}; "
                + (f"set backend.config.{', '.join(unset)}, which it depends on; " if unset else "")
                + f"it requires " + " or ".join(resolved_clauses)
                + f". Run `kura run capabilities {adapter.name}` for field applicability."
            )
    if adapter.surface.renders_argv and not frozen:
        # A true/false value for a field that takes a value would become a bare flag or vanish.
        flagged = sorted(
            key for key, value in native.items()
            if isinstance(value, bool) and key in adapter.surface.fields and key not in adapter.surface.boolean_fields
        )
        if flagged:
            raise ValueError(
                f"{adapter.name} backend.config field(s) take a value, not true or false: "
                + ", ".join(f"{key}={native[key]!r}" for key in flagged)
            )
    extra_args = native.get("extra_args")
    if isinstance(extra_args, list) and not frozen:
        # A declared field has one place, the field, and only where its condition says it applies;
        # extra_args never sets it, so it cannot reach an architecture the field does not work for.
        shadowed = sorted({
            (argument.split("=", 1)[0], field)
            for argument in extra_args if isinstance(argument, str) and argument.startswith("--")
            for field in adapter.surface.fields if f"--{field}".startswith(argument.split("=", 1)[0])
        })
        if shadowed:
            raise ValueError(
                f"{adapter.name} backend.config.extra_args sets "
                + ", ".join(f"{flag} (backend.config.{field})" for flag, field in shadowed)
                + "; write the declared field instead, where the field applies"
            )
    if adapter.validate_authored is not None and not frozen:
        adapter.validate_authored(run)
    # Last, so a more specific refusal of a value the run did write comes first.
    if "command" not in native and not frozen:
        missing = [
            item for item in adapter.surface.required if item.field not in native and matches(item.when_any)
        ]
        if missing:
            raise ValueError(
                f"{adapter.name} backend.config requires "
                + "; ".join(
                    item.field + (f" for {selected(item.when_any)}" if selected(item.when_any) else "")
                    + f" (the trainer would otherwise use {item.otherwise})"
                    for item in missing
                )
                + ". Kura adds no training default; write the value you chose."
                + f" Run `kura run capabilities {adapter.name}` for required fields; {REQUIRED_VALUE_SOURCES}."
            )


# Where a value for a required field (one with no upstream training default) is read. The
# refusal of a missing one and the capabilities listing say it in these words.
REQUIRED_VALUE_SOURCES = (
    "the pinned trainer's own guide for the model (in its docs/) and the family card "
    ".kura/knowledge/model-families/<architecture>.md, when one exists, say what upstream trains with"
)


def backend_capabilities(name: Any, arch: str | None = None) -> dict[str, Any]:
    """A backend's authored surface; with `arch`, a value of its architecture selector, only the
    conditional and required fields that can apply to it and that architecture's model facts."""
    adapter = get_backend(name)
    surface = adapter.surface
    selector, values = surface.config_value_choices[0]
    if arch is not None and arch not in values:
        raise ValueError(f"{adapter.name} has no {selector} {arch!r}; use one of: {', '.join(values)}")

    def applying(when_any: tuple[tuple[tuple[str, tuple[Any, ...]], ...], ...]) -> list[dict[str, list[Any]]]:
        clauses = [{key: list(allowed) for key, allowed in clause} for clause in when_any]
        return [clause for clause in clauses if arch is None or arch in clause.get(selector, [arch])]

    conditional = {item.field for item in surface.conditions}
    conditions = {item.field: applying(item.when_any) for item in surface.conditions}
    required = {item.field: (applying(item.when_any), item.otherwise) for item in surface.required}
    return {
        "backend": adapter.name,
        "common_recipe_fields": sorted(COMMON_RECIPE_FIELDS),
        "config_fields": sorted(surface.fields - conditional),
        "conditional_fields": {field: {"when_any": clauses} for field, clauses in conditions.items() if clauses},
        "unsupported_fields": {**_GENERAL_UNAVAILABLE, **dict(surface.unavailable)},
        "escape_hatches": {
            key: {"validation": "unverified", "recorded": True}
            for key in sorted(surface.escape_hatches)
        },
        "nested_config_fields": deepcopy(surface.nested_config_fields or {}),
        "config_value_choices": {field: list(choices) for field, choices in surface.config_value_choices},
        "integer_minimums": dict(surface.integer_minimums),
        "decided_fields": sorted(surface.decided_fields),
        "decided_notes": dict(surface.decided),
        "required_fields": {
            field: {"when_any": clauses, "otherwise": otherwise}
            for field, (clauses, otherwise) in required.items() if clauses
        },
        "boolean_fields": sorted(surface.boolean_fields),
        "pinned_trainer": {"image": PINNED_IMAGES[adapter.image_name], "source": dict([BUILD_SOURCES[adapter.image_name]])},
        "model_facts": adapter.model_facts(arch) if adapter.model_facts is not None else {},
    }
