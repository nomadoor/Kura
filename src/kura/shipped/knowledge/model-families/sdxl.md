# sdxl (incl. Illustrious / WAI finetunes)

- Recorded: a 1-step sd-scripts LoRA smoke on Docker ran on a 12 GB
  RTX 4070 Ti at 512px, batch 1, dim 8 (alpha 4), U-Net only, bf16, AdamW8bit,
  with latents and text-encoder outputs cached to disk; peak VRAM was not
  recorded, and 1024px is unmeasured.
  source: run 20260801-0932_sd-scripts-smoke-sdxl_4d96 (Kura smoke evidence, 2026-08-01)
- Recorded: on the sd-scripts v0.12.0 image a 1-step LoRA smoke on Docker ran on
  the same GPU at 512px, batch 1, dim 4, U-Net only, bf16, AdamW8bit, with
  latents and text-encoder outputs cached to disk.
  source: real smoke sd-scripts-sdxl (evidence sd-scripts-sdxl-2026-10-19-0211)
- Recorded: a context-free acceptance test on the same GPU trained an
  sd-scripts SDXL LoRA for 750 steps in 32 minutes (2.27 s per step), on the
  sd-scripts image pinned at the time (commit 37a1cbbc). The run folder was not
  kept, so its settings are not recorded.
  source: acceptance test 2026-10-17/18 (local RTX 4070 Ti 12 GB, pins of Kura 0.6.0)
- Recorded: the same test trained an AI-Toolkit `sdxl` LoRA for 750 counted
  steps in 16 minutes (about 1 s per step) on AI-Toolkit 0.13.30. Kura at that
  time miscounted steps under gradient accumulation, so only 375 optimizer
  updates happened; Kura 0.7.0 counts one step as one optimizer update. The
  run folder was not kept.
  source: acceptance test 2026-10-17/18 (local RTX 4070 Ti 12 GB, pins of Kura 0.6.0)

- Disk: sd-scripts reads the single file `sd_xl_base_1.0.safetensors` (6.9 GB,
  fp16); AI-Toolkit's `sdxl` loads the diffusers tree of
  `stabilityai/stable-diffusion-xl-base-1.0` (13.2 GB, fp32 U-Net) from the same
  repository, so training on both stores about 20 GB.
  source: Hugging Face cache measurement 2026-10-18 (blob sha256 compared)

## character

- rank: 16–32
- lr: 1e-4 is a common SDXL option. On sd-scripts, `unet_lr` sets the
  learning rate of the U-Net LoRA parameters and overrides `learning_rate` for
  them; `learning_rate` is the fallback wherever `unet_lr` (or
  `text_encoder_lr` for the text encoders) is unset.
  source: upstream (sd-scripts at the pinned commit, `networks/lora.py`
  `prepare_optimizer_params`)
- batch: 2–4 (effective)
- resolution: 1024 is common SDXL practice; lower it when hardware or the task
  calls for it.
  source: upstream (AI-Toolkit SDXL practice) for 1024 and 1e-4.
- notes: unverified in this workspace — replace with `source: run <id>` after
  the first evaluated run.
