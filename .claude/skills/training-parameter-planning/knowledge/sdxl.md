# sdxl (incl. Illustrious / WAI finetunes)

- VRAM class: character LoRA at 768 / batch 2 is the owner's practical
  starting point on this workspace. 1024 / batch 2 is a higher-cost option
  when the task benefits from it and hardware headroom is available. Add
  gradient checkpointing only on OOM evidence.
  source: owner (2026-07-02) + agent (2026-07-02)

## character

- rank: 16–32
- lr: 1e-4 (adamw8bit)
- batch: 2–4 (effective)
- resolution: 768 owner-preferred starting point; 1024 when the task/model and
  hardware justify it.
- source: owner (2026-07-02) for 768 practical starting point; upstream
  (AI-Toolkit SDXL practice) for 1024 as a common higher-cost option.
- notes: unverified in this workspace — replace with `source: run <id>` after
  the first evaluated run.
