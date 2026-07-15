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
| Wan | Passed local Docker and RunPod 1-step paths | used Wan 2.1 T2V 1.3B, 256px, batch 1, bf16/fp8 base, gradient checkpointing; identity-bound records: `musubi-wan-t2v-1.3b-docker-2026-07-12`, `musubi-wan-t2v-1.3b-runpod-2026-07-12` |
| Krea 2 | Passed local Docker 1-step on 2026-06-30 | used tiny 256px image smoke with fp8 and block swap |
| Qwen-Image | Passed RunPod A40 1-step on 2026-06-30 | 256px, batch 1, fp8, `blocks_to_swap 45`; A5000 reached training start but was SIGKILLed, likely OOM. Treat A5000 as too small for this recipe, not as a general adapter failure |
| Z-Image | Passed local Docker 1-step on 2026-07-01 | used `Comfy-Org/z_image`, 256px, batch 1, fp8 base/scaled/LLM, `blocks_to_swap 24`; use `qwen_3_4b.safetensors`, not the Comfy fp8-mixed text encoder |
| HiDream-O1-Image | Passed local Docker 1-step on 2026-07-01 | used `Comfy-Org/HiDream-O1-Image` dev BF16 checkpoint, 256px, batch 1, `blocks_to_swap 24`, `skip_t2i_visual_dummy`; Comfy fp8-scaled checkpoint is not a Musubi training checkpoint |
| Ideogram 4 | Passed local Docker 1-step on 2026-07-01 | used `Comfy-Org/Ideogram-4`, 256px, batch 1, FP8 DiT as distributed, `blocks_to_swap 24`; completed with paired output checkpoints |
| FLUX.1 Kontext | Passed local Docker 1-step on 2026-07-01 | use gated BFL Kontext DiT/AE plus Comfy FLUX text encoders, 256px paired/control smoke dataset, fp8 base/scaled, fp8 T5, `blocks_to_swap 24`; requires control images |
| HunyuanVideo | Passed local Docker 1-step on 2026-07-01 | used `kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial`, Hunyuan VAE `.pt`, fp16 LLaMA/CLIP text encoders, 256px one-frame video smoke, `fp8_llm`, `blocks_to_swap 36`; Comfy fp8-scaled LLaMA weights failed Musubi loading with unexpected `scale_weight` keys |
| HunyuanVideo 1.5 | Passed local Docker 1-step on 2026-07-01 | used `Comfy-Org/HunyuanVideo_1.5_repackaged`, 256px one-frame video smoke, fp8 base/scaled/VL, `blocks_to_swap 51`, `vae_sample_size 128`, `vae_enable_patch_conv` |
| FramePack | Passed local Docker 1-step on 2026-07-01 | used `Kijai/HunyuanVideo_comfy` FramePack I2V BF16 DiT, Hunyuan VAE `.pt`, fp16 LLaMA/CLIP text encoders, SigLIP image encoder, 256px/37-frame video smoke, fp8 base/scaled/LLM, `blocks_to_swap 36`; real smoke caught the required `--image_encoder` latent-cache argument and `--fp8_base` train flag |
| Kandinsky 5 | Passed local Docker 1-step on 2026-07-01 | used `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s`, Hunyuan VAE safetensors, Qwen2.5-VL-7B + CLIP model IDs, 256px one-frame video smoke, quantized Qwen cache, fp8 base/scaled, `blocks_to_swap 16`; Pro 19B checkpoints should be capacity-planned separately |

## Capacity-first queue

| Architecture | Required model roles in Kura | First real-smoke target | Why / guardrail |
| --- | --- | --- | --- |
| Kandinsky 5 Pro | `dit`, `vae`, `text_encoder_qwen`, `text_encoder_clip` | Only when a Pro-sized run is explicitly needed | Pro DiT checkpoints are about 43GB before Qwen/CLIP; do not use Pro as the default smoke target |

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
