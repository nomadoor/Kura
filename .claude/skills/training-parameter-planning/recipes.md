# Training parameter recipes

Accumulated (architecture × task) knowledge. Read by the
`training-parameter-planning` skill; updated after user-evaluated runs.

Entry rules:

- Provenance is one of: `owner` (stated by the owner), `verified` (cite the
  run id and the user's quality judgment), `unverified` (seeded from upstream
  docs or analogy — treat as a starting hypothesis, not a recommendation).
- `batch` means **effective** batch (micro-batch × gradient accumulation).
- VRAM classes are rough planning figures for LoRA training with the listed
  memory flags, not measurements. Replace them with observed numbers as runs
  accumulate.
- Never overwrite `owner`/`verified` entries with speculation; on conflict,
  keep both lines with run ids.

---

## krea2

- VRAM class: ~24 GB comfortable with `fp8_base` + `fp8_scaled`; 12 GB
  requires rung-2 aids (observed working on RTX 4070 Ti 12 GB with
  `blocks_to_swap`, see smoke runs of 2026-06-30). `unverified` beyond smoke.

### character

- provenance: owner (2026-07-02)
- lr: 5e-5, batch: 2
- notes: owner-stated baseline; not yet linked to an evaluated run. Attach a
  run id when first verified.

### style

- no entry yet. Nearest starting point: character entry with the usual
  style-LoRA shifts (more varied dataset, expect more steps); label any
  proposal `unverified` and confirm direction with the user.

## sdxl (incl. Illustrious / WAI)

### character

- provenance: unverified (seeded from upstream AI-Toolkit SDXL practice and
  the owner's AI-Toolkit SDXL notes)
- rank: 16–32, lr: 1e-4 (adamw8bit), batch: 2–4, resolution: 1024
- VRAM class: fits ~12 GB without memory aids at batch 2; gradient
  checkpointing only if OOM evidence appears.

## sd1.5

- VRAM class: < 8 GB; never needs memory aids on this workspace's hardware.
  Used mainly for smoke/pipeline checks.

## flux family (FLUX.1 dev / Kontext, FLUX.2 klein)

- Artifacts: fp8 DiT + fp8 T5 (`t5xxl_fp8_e4m3fn`) are established
  quality-acceptable for LoRA training and are the rung-1 default at ≤ 24 GB.
  Full-precision T5 is not worth the download for LoRA. provenance: owner
  direction (2026-07-02, efficiency ADR) + upstream Musubi docs.
- VRAM class: ~16–24 GB with fp8 artifacts + gradient checkpointing; below
  that expect offload/swap (ask first).

### character (Kontext edit-LoRA included)

- provenance: unverified (upstream examples)
- rank: 16, lr: 1e-4, batch: 1 with accumulation 2–4
- notes: paired/control datasets change data needs — see `dataset-prep`.

## video (wan / hunyuan_video / framepack / kandinsky5)

- VRAM class: 24 GB is the practical floor even with fp8 + swap; prefer
  RunPod for real training. Local 12 GB is smoke-only. provenance:
  unverified, based on 2026-06/07 smoke behavior.
- No task entries yet.
