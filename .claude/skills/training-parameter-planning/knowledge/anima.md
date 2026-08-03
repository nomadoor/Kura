# anima

Prompt and evaluation knowledge belongs in `lora-evaluation/knowledge/anima.md`.
This card covers training parameters only.

- VRAM class: Anima character LoRA completed at 768, rank 16, batch 1, bf16,
  gradient checkpointing, and disk-backed latent/text caches on an RTX 4070 Ti
  with 12 GB VRAM. Observed speed was about 1.0 second per optimizer step after
  initialization.
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3

## character

- learning rate: start from 7e-5 with a constant scheduler for a small
  character dataset when no stronger architecture-specific evidence applies.
  source: owner (2026-08-04)
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- resolution: 768
  source: owner (2026-07-02)
- rank / alpha: 16 / 8 as a verified small-character starting point.
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- checkpoint review: retain checkpoints every 100 steps during exploratory
  runs. For a dataset near six images with one repeat and batch 1, evaluate
  800–1200 first and use 1000 as the current preferred checkpoint.
  source: owner (2026-08-04)
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- overtraining reminder: in the cited six-image run, checkpoints after roughly
  1500 increased learned rendering-style leakage without a clear general
  improvement in character usefulness. Do not assume more steps are better.
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- scope: the 1000-step preference is evidence for a very small, visually
  homogeneous character dataset. Do not copy the absolute step count to a
  larger dataset; compare optimizer steps, repeats, effective batch, dataset
  diversity, and retained checkpoints together.

## style

- no entry yet. The character evidence above explicitly showed unwanted style
  retention and must not be treated as a style-LoRA baseline.
