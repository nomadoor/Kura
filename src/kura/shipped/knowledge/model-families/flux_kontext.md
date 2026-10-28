# flux_kontext


- Recorded: a 1-step Musubi Tuner v0.3.6 LoRA smoke on Docker ran on a 12 GB
  RTX 4070 Ti at 256px, batch 1, dim 1, fp8 DiT (`fp8_base`, `fp8_scaled`),
  fp8 T5 file, 24 swapped blocks, gradient checkpointing, AdamW8bit; peak VRAM
  was not recorded.
  source: run 20261018-2300_musubi-flux-kontext_ee86 (Kura smoke evidence, 2026-10-18)
- Artifacts: owner baseline is fp8 DiT + fp8 T5 (`t5xxl_fp8_e4m3fn` with
  `fp8_t5: true`) for local/consumer-GPU LoRA training. Full-precision T5 is
  not worth the download for normal LoRA use unless a run has a specific reason
  to test it.
  source: owner (2026-07-02, efficiency ADR) + upstream (Musubi flux_kontext.md)
- VRAM class: ~16–24 GB with fp8 artifacts + gradient checkpointing; below
  that expect offload/swap (ask first).
  source: agent (2026-07-02)

## character / edit-LoRA

- rank: 16
- lr: 1e-4 is the common upstream reference.
- batch: 1 micro × accumulation 2–4
- source: upstream (Musubi examples) for the 1e-4 reference.
- notes: paired/control datasets change data needs — see the `dataset` skill.
