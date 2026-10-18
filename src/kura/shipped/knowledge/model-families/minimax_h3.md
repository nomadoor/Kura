# minimax_h3

## Training

- Recorded: musubi-tuner `architecture: minimax_h3` completed one real optimizer
  step on RunPod in two modes the records name `t2va-guidance-loss` (124 target
  frames, 2026-09-22) and `one-frame-text-to-image-guidance-loss` (2026-09-22,
  and again on 2026-09-29 from an image JSONL dataset with a cached
  unconditional embedding). These ran on trainer image sha256:de5d31f26dde, an
  image older than the one Kura 0.7.0 pins.
  source: evidence musubi-minimax-h3-runpod-2026-09-22
  source: evidence musubi-minimax-h3-image-runpod-2026-09-22
  source: evidence musubi-minimax-h3-guidance-cache-runpod-2026-09-29
- Recorded: AI-Toolkit `model_arch: minimax_h3` completed one real optimizer
  step on RunPod in the modes the records name `lora` (2026-09-21) and
  `image-lora` (2026-09-22), both without gradient checkpointing, on trainer
  image sha256:9aa6861b0f54, an image older than the one Kura 0.7.0 pins.
  source: evidence ai-toolkit-minimax-h3-runpod-2026-09-21
  source: evidence ai-toolkit-minimax-h3-image-runpod-2026-09-22
- Not recorded yet: model files, VRAM, speed, and quality settings. The records
  above show that training started and stepped, not what to train with.

## character

- no entry yet.

## style

- no entry yet.
