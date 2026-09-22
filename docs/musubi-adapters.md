# Musubi Tuner adapters

For the cross-backend version, adapter, and real-hardware summary, see
[backend-support.md](backend-support.md). This document keeps the detailed
Musubi adapter mechanics and historical smoke commands.

Musubi Tuner and Kura's Musubi backend are not the same layer.

- **Musubi Tuner support** means the upstream tool has training scripts,
  arguments, and model conventions for an architecture.
- **Kura built-in adapter support** means Kura can generate the Musubi command,
  cache commands, model lock, validation, and output checks from `run.yaml`.
- **Image smoke** means the configured Docker image contains the scripts used by
  those adapters and each script can start its `--help` path. Run
  `uv run kura doctor musubi` after changing `MUSUBI_TUNER_REF` or rebuilding the
  image.
- **Real smoke** means at least one tiny training run has actually launched for
  that adapter. If this has not been done, call the adapter experimental or
  unverified rather than simply "done."

As of 2026-09-21, all adapters listed below passed image smoke on
`nomadoor/kura-musubi-tuner:dev`: the configured Docker image contains the
39 expected Musubi scripts and each script can start its `--help` path. Earlier
Docker launch smoke also proved that Kura can compile and start adapter commands
against the Musubi entrypoints, but dummy-path launch smoke is not a real
one-step training proof.

Real one-step smoke is tracked separately. It must use actual model files and
finish one optimizer step through Kura's normal Docker or RunPod executor. The
developer runner is `uv run python scripts/musubi_real_smoke.py <architecture>`.
Choose the first executor/GPU from the concrete model, dataset, precision,
memory, and disk facts before running an expensive smoke; do not probe GPU
classes blindly.

If an architecture is not listed as built-in below, do not say that Musubi Tuner
does not support it. Say that Kura does not yet have a built-in Musubi command
generator for it.

## Current Kura built-in adapters

| Architecture | Kura built-in adapter | Image smoke | Real smoke | Notes |
| --- | --- | --- | --- | --- |
| MiniMax-H3 | yes | passed `kura doctor musubi` on the pinned v0.3.5 image | T2VA guidance loss and plain one-frame image guidance loss passed A40 1-step with output/state recovery (`musubi-minimax-h3-runpod-2026-09-22`, `musubi-minimax-h3-image-runpod-2026-09-22`); other modes remain pending | `architecture: minimax_h3` or `minimaxh3`; typed dataset and command coverage includes `task: t2va`, `fl2va`, and `ref2va`, `one_frame`, timed controls, ordered references, and `h3_loss_method: guidance`, `training_adapter`, or `teacher_matching`. |
| FLUX.2 / FLUX.2 klein | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | verified in prior local/RunPod runs | `architecture: flux2` or `flux_2` |
| Wan | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker and RunPod 1-step paths; identity-bound records: `musubi-wan-t2v-1.3b-docker-2026-07-12`, `musubi-wan-t2v-1.3b-runpod-2026-07-12` | `architecture: wan` |
| Krea 2 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | base path passed local Docker 1-step on 2026-06-30 (`scripts/musubi_real_smoke.py krea2`); v0.3.5 ConvRot/offload paths remain compile-only | `architecture: krea2` or `krea_2`; typed `convrot_int8`, `convrot_int8_bwd`, and `gradient_checkpointing_cpu_offload` accommodations |
| Qwen-Image | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed RunPod A40 1-step on 2026-06-30 (`scripts/musubi_real_smoke.py qwen_image --executor runpod --gpu "NVIDIA A40"`); A5000 reached training start but was SIGKILLed, likely OOM for that 256px/fp8/block-swap recipe | `architecture: qwen_image` |
| Z-Image | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py zimage --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: zimage` or `z_image`; use upstream `qwen_3_4b.safetensors` text encoder |
| FLUX.1 Kontext | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py flux_kontext --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: flux_kontext` or `flux1_kontext`; requires paired/control dataset entries |
| Ideogram 4 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py ideogram4 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: ideogram4` or `ideogram_4` |
| HiDream-O1-Image | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py hidream_o1 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: hidream_o1` or `hidream`; use BF16 training checkpoint, not Comfy fp8-scaled checkpoint |
| HunyuanVideo | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py hunyuan_video --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: hunyuan_video` or `hunyuanvideo`; use a standard fp16 LLaMA text encoder with `fp8_llm` rather than Comfy fp8-scaled LLaMA weights |
| HunyuanVideo 1.5 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py hunyuan_video_1_5 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: hunyuan_video_1_5` |
| FramePack | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py framepack --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: framepack` or `frame_pack` |
| Kandinsky 5 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 with the Lite T2V model (`scripts/musubi_real_smoke.py kandinsky5 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: kandinsky5` or `kandinsky_5`; Pro 19B checkpoints remain capacity-dependent |

This list should be checked against the current upstream Musubi Tuner README
before adding adapters.

## Variant coverage

An architecture row is not a claim that every upstream variant has completed a
real smoke. Kura audits variants when they change the generated execution
contract: scripts, mandatory model roles, dataset shape, cache flags, training
flags, or outputs. Checkpoint substitutions that keep the same contract do not
need a separate adapter or exhaustive real-smoke run.

The v0.3.5 audit adds MiniMax-H3 T2VA, FL2VA, Ref2VA, one-frame, guidance,
training-adapter, and teacher-matching command compilation. Teacher matching
deliberately projects condition-specific latent-cache tasks separately from its
T2VA text-cache and trainer tasks. It also carries forward the distinct compile
paths identified at v0.3.4 for Wan 2.2
dual-noise training, Wan Single Frame, FramePack Single Frame, Qwen-Image
Edit/Layered model versions, HunyuanVideo 1.5 I2V, HiDream-O1 I2I, and
Kandinsky 5 I2V. These are not marked real-smoke verified until an actual model
finishes one optimizer step through Kura. The evidence boundary is defined in
the support-status vocabulary in [backend-support.md](backend-support.md).

MiniMax-H3 `training_adapter` requires a BF16 DiT because upstream must merge
`base_weights` before any ConvRot quantization. Kura rejects its known
pre-quantized ConvRot INT8 bundles for this loss method at compile time.

### Typed MiniMax-H3 datasets

Use `backend.config.h3_dataset_config` for MiniMax-H3 rather than the reviewed
native `dataset_config` escape hatch. It contains one entry for each run
`datasets[]` item and uses dataset-relative paths:

```yaml
backend:
  name: musubi-tuner
  config:
    architecture: minimax_h3
    task: fl2va
    one_frame: true
    video_only: true
    h3_dataset_config:
      general:
        resolution: [1024, 1024]
        batch_size: 1
      datasets:
        - source: image_directory
          path: targets
          control_subdir: controls
          fp_1f_clean_indices: [0]
          fp_1f_target_index: 24
```

The typed contract supports image/video directories and image/video JSONL.
Kura projects paths into the selected dataset, validates one-frame source and
task compatibility, checks timed FL2VA controls, validates Ref2VA reference
records and limits, and rejects paths that escape the dataset. JSONL target,
control, reference, and explicit audio paths are checked against the workspace
before launch. Video sources must declare `target_frames` on MiniMax-H3's
`5+17n` frame grid (for example `[124]`); Kura refuses to fall through to the
upstream one-frame default. `uv run kura run capabilities musubi-tuner` lists
the complete authored field surface.

Krea 2 exposes `convrot_int8` and `convrot_int8_bwd` (`bf16` or `int8`) as
typed v0.3.5 memory accommodations. ConvRot cannot be combined with FP8 or the
Turbo sampling DiT. `gradient_checkpointing_cpu_offload` is also typed and
requires `gradient_checkpointing: true`; Kura rejects invalid combinations
during compile rather than leaving them for a paid runtime to discover.

### Typed block swap controls

Musubi's shared block-swap controls are first-class `backend.config` fields for
all built-in architectures:

- `blocks_to_swap`: number of transformer blocks kept on CPU.
- `block_swap_h2d_only`: stream frozen base weights from CPU to reusable GPU
  buffers without copying them back. This is intended for LoRA-style training
  and requires both a positive `blocks_to_swap` value and
  `gradient_checkpointing: true`.
- `block_swap_ring_size`: positive GPU buffer count for H2D-only streaming.
  `2` overlaps transfer and compute; `1` uses less VRAM but does not overlap
  them. It is rejected unless H2D-only mode is enabled.
- `use_pinned_memory_for_block_swap`: pin the CPU swap memory. This can improve
  transfer speed but increases locked host-memory pressure and therefore must
  be chosen deliberately.

Kura rejects contradictory typed/`extra_args` declarations and invalid
dependencies before launch. These are execution accommodations, not evidence
that a particular architecture/model/precision combination has completed an
optimizer step; runtime evidence remains a separate matrix entry.

## Escape hatch

For an upstream-supported architecture without a Kura built-in adapter, a run may
still use `backend.config.command` to provide the exact command.
That keeps Kura responsible for workspace files, Docker/RunPod execution,
monitoring, downloads, and cleanup, while the Musubi command itself is explicit.

Use this as a temporary escape hatch, not as a substitute for adding adapters for
commonly used architectures.
