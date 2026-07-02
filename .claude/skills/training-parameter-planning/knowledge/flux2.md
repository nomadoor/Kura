# flux2 (klein 4b/9b, base)

- Artifacts: prefer fp8/quantized text encoder variants where the Musubi
  adapter supports them; same reasoning as flux_kontext.
  source: agent (2026-07-02)
- VRAM class: klein-4b trains locally on 12 GB with rung-2 aids (2026-07-02
  all-local smoke); 9b and base need more headroom or RunPod.
  source: run 20260702-1011_all-local-flux2-klein-flux-2-klein-4b-text2image_c3f7 (smoke only)
- No evaluated task entries yet.
