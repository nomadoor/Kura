---
name: sd-scripts-backend
description: sd-scripts backend guidance for Kura. Use for sd-scripts model roles, native dataset TOML, run-scoped disk caches, LoRA or Anima LLLite commands, output conversion and validation, recovery, Docker images, or ComfyUI compatibility.
---

# sd-scripts backend

Use this skill for `backend.name: sd-scripts` work.

Author and validate backend input under the
[declared surface contract](../../../docs/adr/backend-config-surface-contract.md).

This backend owns training mechanics, not prompt semantics. For evaluation,
use `lora-evaluation` and its model-family knowledge; do not duplicate Anima,
SDXL, SD 1.5, or FLUX prompt policy here.

## Contract

- Never run sd-scripts directly on the host. Compile through Kura and execute
  with Docker or RunPod.
- The initial built-in selectors are SD 1.5 LoRA, SDXL LoRA, FLUX.1 LoRA,
  Anima LoRA, and Anima ControlNet-LLLite. Other upstream paths require a fully
  reviewed `backend.config.command`; do not describe them as built-in support.
- Keep architecture roles explicit in `model_paths` or `model_downloads`.
  Immutable Hugging Face revisions are preferred.
- Use the upstream two-level `[[datasets]]` / `[[datasets.subsets]]` TOML.
  Kura stages files below `runs/<id>/cache/sd-scripts/datasets`; never point a
  native disk cache at the shared `datasets/` tree.
- Before authoring `dataset_config`, run
  `uv run kura run capabilities sd-scripts`. Its nested field sections are the
  reviewed schema for `[general]`, `[[datasets]]`, and
  `[[datasets.subsets]]`; upstream keys absent there remain unsupported. Kura
  validates nested types and ranges during compile.
- The pinned schema supports `caption_dropout_rate`,
  `caption_dropout_every_n_epochs`, and `caption_tag_dropout_rate` at all three
  inheritance levels. Anima may preserve `caption_dropout_rate` in its
  text-encoder cache. Do not combine a text-encoder cache with enabled
  epoch-based or tag dropout, caption shuffling, or token warmup; those controls
  are dynamic and are rejected instead of being silently lost. The upstream
  disabled value `caption_dropout_every_n_epochs: 0` remains valid.
- Upstream acceptance alone is not Kura support. In particular,
  `validation_seed`, `validation_split`, and arbitrary `custom_attributes` are
  intentionally absent from the reviewed surface: the pinned Anima LLLite
  entrypoint does not consume its constructed validation dataset, and Kura does
  not accept unvalidated nested mappings as if they changed training.
- If a disk cache is enabled, record a measured
  `backend.config.disk_cache_estimate_gb`. Treat
  `safety.allow_unknown_disk_cache` as an explicit reviewed exception.

## Anima artifacts

- Anima LoRA trains native weights under the run cache. Validate and convert
  the final weight and every retained step checkpoint with the pinned upstream
  converter, validate every converted file, and only then publish the complete
  checkpoint to `outputs/` with its original filename as soon as it is stable.
  The training wrapper must also publish discoverable retained checkpoints when
  the trainer exits nonzero. Never leave user-requested checkpoints available
  only below `cache/`.
- If conversion fails after training, retain all discovered native weights
  below `recovery/sd-scripts/anima-native`. They are not final outputs and must
  not appear in `status.outputs`.
- Anima LLLite is a ComfyUI model patch, not a LoRA. Require safetensors
  metadata `lllite.version=2` and the core key family including
  `lllite_conditioning1.conv1.weight`.
- Load LLLite in ComfyUI core with `ModelPatchLoader` from
  `models/model_patches`, then apply it with `AnimaLLLiteApply`. Do not install
  custom nodes or modify a user's separately managed ComfyUI.

## Resources

- Do not introduce hidden low-memory defaults. Record checkpointing, disk
  caches, offload, block swap, and precision in `backend.config`.
- Keep flow-matching choices explicit in the reviewed plan. FLUX starts from
  `flux_shift`, guidance `1.0`, and raw prediction; Anima LoRA starts from
  sigmoid; Anima LLLite starts from shift with discrete flow shift `3.0`.
  Unknown built-in config keys are errors, not pass-through options.
- Review the plan's `dataset_config` section for effective batch, resolution,
  bucket/no-upscale, network multiplier, skipped resolution, and per-subset
  caption controls. Dataset and subset differences must not be inferred from a
  single top-level batch or resolution summary.
- Anima LoRA rejects `fp8_base`, conflicting text-encoder training/cache, and
  incompatible offload combinations. LLLite rejects block swap and the
  unsupported offload/DeepSpeed/fused-backward paths.
- Preserve recipe choices first. Explain any resolution, rank, effective batch,
  or model-size change before editing the run.

## Verification

Before launch, always run:

```sh
uv run kura doctor disk
uv run kura doctor sd-scripts
uv run kura run compile <run-id>
uv run kura run plan <run-id>
```

Show the plan and obtain explicit approval before a real launch. For the
initial milestone, run all five real smokes with local Docker; do not require
or launch RunPod. A Tier 1 claim
requires one real optimizer step for that selector. Anima additionally requires
loading and rendering the actual output with Kura's pinned managed ComfyUI core.
Record exact adapter/image/model identities and confirm `datasets/` is unchanged
for disk-cache smokes.
