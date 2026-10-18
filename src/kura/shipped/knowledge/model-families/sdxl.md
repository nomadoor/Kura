# sdxl (incl. Illustrious / WAI finetunes)

- Recorded: a 1-step sd-scripts LoRA smoke on Docker ran on a 12 GB
  RTX 4070 Ti at 512px, batch 1, dim 8 (alpha 4), U-Net only, bf16, AdamW8bit,
  with latents and text-encoder outputs cached to disk; peak VRAM was not
  recorded, and 1024px is unmeasured.
  source: run 20260801-0932_sd-scripts-smoke-sdxl_4d96 (Kura smoke evidence, 2026-08-01)

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
