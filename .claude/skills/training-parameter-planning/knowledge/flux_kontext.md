# flux_kontext

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
- lr: 1e-4
- batch: 1 micro × accumulation 2–4
- source: upstream (Musubi examples)
- notes: paired/control datasets change data needs — see `dataset-prep`.
