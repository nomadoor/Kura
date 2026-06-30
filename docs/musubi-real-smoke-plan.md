# Musubi real smoke plan

This is an internal developer plan for real one-step Musubi adapter smoke tests.
It is not a user-facing support matrix.

A real smoke must use real model files, launch through Kura's Docker or RunPod
executor, complete one optimizer step, and verify/download the produced LoRA
outputs. Image smoke and dummy-path launch smoke are useful, but they are not
real training proof.

Before launching any expensive smoke, write down the model files, resolution,
batch size, precision/offload flags, expected VRAM class, and chosen GPU. Do not
probe GPUs blindly from cheap to expensive.

Prefer local Docker first when the model and settings plausibly fit. Local runs
reuse the workspace HF cache and do not bill during model downloads. RunPod is
for cases that exceed local VRAM/RAM or require a clean remote proof; remember
that disposable Pods usually pay to redownload model files.

## Current verified results

| Architecture | Real smoke result | Capacity note |
| --- | --- | --- |
| FLUX.2 / FLUX.2 klein | Verified in prior local/RunPod runs | depends strongly on 4B vs 9B and fp8/block swap settings |
| Wan | Passed local Docker 1-step on 2026-06-30 | used Wan 2.1 T2V 1.3B, 256px, batch 1, bf16/fp8 base, gradient checkpointing; model files were stored once in HF cache and exposed through Kura's Musubi symlink tree |
| Krea 2 | Passed local Docker 1-step on 2026-06-30 | used tiny 256px image smoke with fp8 and block swap |
| Qwen-Image | Passed RunPod A40 1-step on 2026-06-30 | 256px, batch 1, fp8, `blocks_to_swap 45`; A5000 reached training start but was SIGKILLed, likely OOM. Treat A5000 as too small for this recipe, not as a general adapter failure |

## Capacity-first queue

| Architecture | Required model roles in Kura | First real-smoke target | Why / guardrail |
| --- | --- | --- | --- |
| Z-Image | `dit`, `vae`, `text_encoder` | Plan first; local if model size/cache makes 12GB plausible, otherwise A40 | large image model class; do not try A5000 unless model sizes make 24GB plausible |
| FLUX.1 Kontext | `dit`, `vae`, `text_encoder1`, `text_encoder2` | Plan first; local if using cached FLUX.1 stack and low-VRAM settings, otherwise A40 | FLUX-class image-edit stack; A5000 only for an explicit low-VRAM recipe |
| Ideogram 4 | `dit`, optional/required `vae`, `text_encoder` depending cache/sampling | Plan first; local only if model files are modest/cached, otherwise A40+ | model files and upstream recipe must be confirmed before spending GPU time |
| HiDream-O1-Image | `dit` | Plan first; local only if DIT size makes 12GB plausible, otherwise A40+ | DIT-only in Kura, but expected model size may still exceed 24GB |
| HunyuanVideo | `dit`, `vae`, `text_encoder1`, `text_encoder2` | Defer unless a small official smoke recipe exists | video stack; likely too expensive to probe casually |
| HunyuanVideo 1.5 | `dit`, `vae`, `text_encoder`, `byt5`, optional `image_encoder` | Defer unless a small official smoke recipe exists | large video stack with multiple encoders |
| FramePack | `dit`, `vae`, `text_encoder1`, `text_encoder2`, `image_encoder` | Defer unless a small official smoke recipe exists | video/image-to-video stack; estimate before launch |
| Kandinsky 5 | `dit`, `vae`, `text_encoder_qwen`, `text_encoder_clip` | Defer unless a small official smoke recipe exists | video-oriented task by default in Kura (`k5-pro-t2v-5s-sd`) |

## Execution rule

For each new architecture:

1. Confirm the current upstream Musubi recipe and exact model files.
2. Add a `SmokeSpec` to `scripts/musubi_real_smoke.py` with real `model_downloads`
   or explicit `model_paths`.
3. Run `--no-launch` and inspect `kura run plan`.
4. Decide the first executor/GPU from the capacity plan, not by blind retry.
   Prefer local Docker when plausible; use RunPod only when local capacity or
   confidence is insufficient.
5. Launch once. If it OOMs after reaching the Musubi entrypoint, record it as
   capacity data and choose the next GPU only when the estimate says it is
   likely to pass.
6. Update `docs/musubi-adapters.md` with the exact executor/GPU and result.

Do not count a test as passed merely because models downloaded or the train
script started.
