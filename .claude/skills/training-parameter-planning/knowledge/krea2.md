# krea2

- VRAM class: ~24 GB comfortable with `fp8_base` + `fp8_scaled`; 12 GB needs
  rung-2 aids (smoke runs of 2026-06-30 on RTX 4070 Ti 12 GB used
  `blocks_to_swap: 26`). Training quality at 12 GB not yet evaluated.
  source: run 20260630-1905_musubi-real-smoke-krea2_1a38 (smoke only)

## character

- lr: 7e-5 provisional Kura default
  source: owner (2026-07-02)
- batch: 2 (effective)
  source: owner (2026-07-02)
- resolution: 768 as the owner-preferred practical character-LoRA starting
  point; raise toward 1024 only when hardware headroom and task goals justify
  the extra cost.
  source: owner (2026-07-02)
- notes: owner-stated baseline; attach a run id when first verified on an
  evaluated run.

## style

- no entry yet. Nearest starting point: the character entry with the usual
  style shifts (more varied dataset, expect more steps). Label proposals
  `source: agent` and confirm direction with the user.
