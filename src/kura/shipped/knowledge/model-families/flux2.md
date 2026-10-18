# flux2 (klein 4b/9b, base)

## Which variant to train on, which to generate with

Train on Klein Base, and generate with Klein Base as well. Plain FLUX.2 is the
distilled model; generating with it may be attempted, but it fails to work
properly roughly nine times out of ten.
source: owner (2026-08-15)

## Training

- Artifacts: do not infer from FLUX.1/Kontext. FLUX.2 has different model
  structure and text encoder handling; confirm the exact Musubi adapter recipe
  and available artifacts before proposing fp8/quantized variants.
  source: owner (2026-07-02) + agent (2026-07-02)
- VRAM class: klein-4b trained locally on 12 GB in a 2026-07-02 smoke (its
  memory settings were not recorded); 9b and base need more headroom or RunPod.
  source: run 20260702-1011_all-local-flux2-klein-flux-2-klein-4b-text2image_c3f7 (smoke only)
- AI-Toolkit `flux2_klein_4b` on an RTX 4070 Ti (12 GB), baseline `qfloat8`
  quantization, 768 px: with the text encoder resident (the baseline) about
  2.6 s per step at 11.15 GB; with `native_config.train.cache_text_embeddings:
  true` 1.3 to 2.3 s per step at 7.9 GB.
  source: acceptance run 2026-10-18 (local RTX 4070 Ti 12 GB)
- Musubi `architecture: flux2` Klein 4B on the same GPU with `fp8_base` and
  `fp8_scaled`, 768 px: 1.74 s per step on the first try.
  source: acceptance run 2026-10-18 (local RTX 4070 Ti 12 GB)
- Musubi 9B on an NVIDIA A40 (48 GB): at 1024-class resolution and rank 32,
  training without gradient checkpointing ran out of memory even at
  batch_size 1, and batch_size above 1 (a GPU micro-batch in Musubi) ran out of
  memory before step 1. One observation, not a rule: with these settings on an
  A40, suggest gradient_checkpointing: true and batch_size 1 with gradient
  accumulation for the effective batch, and say why; the user decides.
  source: Kura's former A40 guard (added 2026-07-04, moved here 2026-10-14); the observing run was not recorded
- No evaluated task entries yet.
