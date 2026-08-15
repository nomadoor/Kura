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
- VRAM class: klein-4b trains locally on 12 GB with rung-2 aids (2026-07-02
  all-local smoke); 9b and base need more headroom or RunPod.
  source: run 20260702-1011_all-local-flux2-klein-flux-2-klein-4b-text2image_c3f7 (smoke only)
- No evaluated task entries yet.
