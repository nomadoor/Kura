# sdxl (incl. Illustrious / WAI finetunes)

- Recorded: a 1-step sd-scripts LoRA smoke on Docker ran on a 12 GB
  RTX 4070 Ti at 512px, batch 1, dim 8 (alpha 4), U-Net only, bf16, AdamW8bit,
  with latents and text-encoder outputs cached to disk; peak VRAM was not
  recorded, and 1024px is unmeasured.
  source: run 20260801-0932_sd-scripts-smoke-sdxl_4d96 (Kura smoke evidence, 2026-08-01)

## character

- rank: 16–32
- lr: 1e-4 is a common SDXL option
- batch: 2–4 (effective)
- resolution: 1024 is common SDXL practice; lower it when hardware or the task
  calls for it.
  source: upstream (AI-Toolkit SDXL practice) for 1024 and 1e-4.
- notes: unverified in this workspace — replace with `source: run <id>` after
  the first evaluated run.
