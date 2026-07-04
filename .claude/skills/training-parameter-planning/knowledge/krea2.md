# krea2

- VRAM class: ~24 GB is practical with `fp8_base` + `fp8_scaled` +
  `gradient_checkpointing` for 768/rank16/batch2 character LoRA. A5000-class
  24 GB OOMed at batch2 without gradient checkpointing.
  source: run 20260703-0013_myakumyaku-krea2-768-runpod-candidates_5a27
  source: run 20260703-0021_myakumyaku-krea2-768-runpod-gc_1c25
- 12 GB can run 768/rank16/effective-batch2 only as forced local mode with
  rung-2 aids: `gradient_checkpointing`, batch1+accum2, and block swap around
  26. Observed about 20-21s/step on RTX 4070 Ti.
  source: run 20260702-2343_myakumyaku-krea2-768-12gb-rootdata_1a0e

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
- evidence: 768/rank16/lr7e-5/effective-batch2 completed locally on 12 GB
  with heavy speed aids, and on RunPod A5000 with gradient checkpointing only.
  source: run 20260702-2343_myakumyaku-krea2-768-12gb-rootdata_1a0e
  source: run 20260703-0021_myakumyaku-krea2-768-runpod-gc_1c25

## style

- no entry yet. Nearest starting point: the character entry with the usual
  style shifts (more varied dataset, expect more steps). Label proposals
  `source: agent` and confirm direction with the user.
