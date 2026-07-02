# Video architectures — placeholder cards

Split an architecture out into its own card (`wan.md`, `hunyuan_video.md`,
`framepack.md`, `kandinsky5.md`, …) as soon as it has architecture-specific
content. Until then, this is a weak fallback only; do not assume video
architectures share parameters just because they share this placeholder.

- VRAM class: ~24 GB is the practical floor for real training even with fp8
  and swap; local 12 GB is smoke-only. Prefer RunPod for real video training.
  source: agent (2026-07-02, based on 2026-06/07 smoke behavior)
- No task entries yet for: wan, hunyuan_video, hunyuan_video_1_5, framepack,
  kandinsky5.
