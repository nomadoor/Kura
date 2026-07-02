# sdxl (incl. Illustrious / WAI finetunes)

- VRAM class: LoRA at 1024 / batch 2 fits ~12 GB without memory aids
  (adamw8bit). Add gradient checkpointing only on OOM evidence.
  source: agent (2026-07-02)

## character

- rank: 16–32
- lr: 1e-4 (adamw8bit)
- batch: 2–4 (effective)
- resolution: 1024
- source: upstream (AI-Toolkit SDXL practice; owner's AI-Toolkit SDXL notes)
- notes: unverified in this workspace — replace with `source: run <id>` after
  the first evaluated run.
