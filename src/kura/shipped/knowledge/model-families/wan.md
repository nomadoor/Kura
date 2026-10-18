# wan

## Training

- Recorded: musubi-tuner `architecture: wan` passed Kura's conformance scenario
  (fresh, split, and Resume runs, up to 7 optimizer steps, training state kept)
  on local Docker and on RunPod on 2026-10-19, on trainer image
  sha256:527d4812bb0c (the image Kura 0.7.0 pins).
  source: evidence musubi-tuner-release-conformance-docker-2026-10-19
  source: evidence musubi-tuner-release-conformance-runpod-2026-10-19
- Recorded: musubi-tuner `architecture: wan` with `task: t2v-1.3B` completed a
  real optimizer step on local Docker from a generated video JSONL dataset,
  resumed from step 1 to 2, on 2026-09-29, on trainer image sha256:de5d31f26dde,
  an image older than the one Kura 0.7.0 pins.
  source: evidence musubi-wan-video-jsonl-docker-2026-09-29
- Not recorded yet: model files, VRAM, speed, and quality settings. The records
  above show that training started and stepped, not what to train with.
- AI-Toolkit's `wan21` and `wan22_*` values have no recorded run.

## character

- no entry yet.

## style

- no entry yet.
