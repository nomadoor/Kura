# moss_image

## Training

- Backend and model: AI-Toolkit `model_arch: moss_image:base` with
  `model.base: OpenMOSS-Team/MOSS-Image-Base`.
  source: run 20261017-1752_myakumyaku-moss-image-te-cache_1b4d
- Files the trainer used: it resolved `model.base` to the Comfy-Org/MOSS-Image
  files and downloaded them itself: `moss_image_base_bf16.safetensors`
  (10.2 GB), `qwen3.5_9b_int8_convrot.safetensors` (9.7 GB), and `flux2-vae`
  (0.34 GB).
  source: owner measurement (2026-10-17)
- VRAM class: on an RTX 4070 Ti with 12 GB, with `quantize`, `quantize_te`,
  and `low_vram` on, a run with the text encoder resident took about 75 s per
  step.
  source: run 20261017-1724_myakumyaku-moss-image_ce28
- On 12 GB, cache and unload the text encoder: with `native_config.train`
  `cache_text_embeddings: true` and `unload_text_encoder: true` added, the run
  took about 2 s per step and about 8.5 GB of VRAM; 1000 steps at 768 px,
  rank 16, finished in 32 minutes.
  source: run 20261017-1752_myakumyaku-moss-image-te-cache_1b4d
- Recorded: a context-free acceptance test on the same GPU, with the text
  encoder cached and unloaded, reported 1000 steps in 38 minutes (1.87 s per
  step) and a 63 MiB LoRA. The run folder was not kept.
  source: acceptance test 2026-10-17/18 (local RTX 4070 Ti 12 GB, pins of Kura 0.6.0)
- The runs above used AI-Toolkit 0.13.30. From 0.13.32 the trainer saves a MOSS
  LoRA with separate q, k, and v keys instead of fused qkv keys, so its files
  differ from those runs' files; no MOSS run has used 0.13.32 yet.
  source: AI-Toolkit commit 9a1828f (moss_image.py `convert_lora_weights_before_save`)
- Edit on the SFT weights: `model_arch: moss_image:edit` with
  `model.base: OpenMOSS-Team/MOSS-Image-SFT` and `model_edit: true` fills the
  upstream UI's `kv_cache` and `match_target_res`. It is untested: Kura only
  compiles it.
  source: AI-Toolkit baseline entry `moss_image:edit` (pinned commit 9d408eb6)

## character

- no evaluated entry yet.

## style

- no entry yet.
