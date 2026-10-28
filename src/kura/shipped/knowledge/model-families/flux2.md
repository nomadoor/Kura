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
- Recorded: in that acceptance test the AI-Toolkit Klein 4B LoRA ran 650 steps
  in 21 minutes on AI-Toolkit 0.13.30. The run folder was not kept.
  source: acceptance test 2026-10-17/18 (local RTX 4070 Ti 12 GB, pins of Kura 0.6.0)
- Disk: for Klein 4B, AI-Toolkit downloads `black-forest-labs/FLUX.2-klein-base-4B`,
  `Qwen/Qwen3-4B` and `ai-toolkit/flux2_vae`; Musubi's bundle downloads
  `Comfy-Org/vae-text-encorder-for-flux-klein-4b`. The transformer and VAE files
  are byte-identical across the two, but the Hugging Face cache keeps one copy
  per repository, so training on both backends stores about 15 GB twice.
  source: Hugging Face cache measurement 2026-10-18 (blob sha256 compared)
- Musubi `architecture: flux2` Klein 4B on the same GPU with `fp8_base` and
  `fp8_scaled`, 768 px: 1.74 s per step on the first try.
  source: acceptance run 2026-10-18 (local RTX 4070 Ti 12 GB)
- Recorded: that Musubi Klein 4B LoRA ran 1000 steps in 35 minutes on Musubi
  Tuner v0.3.5. The run folder was not kept.
  source: acceptance test 2026-10-17/18 (local RTX 4070 Ti 12 GB, pins of Kura 0.6.0)
- Recorded on Musubi Tuner v0.3.6 on RunPod: Klein Base 4B
  (`model_bundle: flux2-klein-base-4b`) on an RTX A5000 (24 GB) at 1024 px in
  bf16 (no fp8), rank 32 / alpha 16, learning rate 2e-4 constant, adamw8bit,
  `timestep_sampling: flux2_shift`, gradient checkpointing, batch size 1:
  2.32 to 2.36 s per step, 100 steps in 3 min 53 s. Each LoRA file (saved
  bf16) was 92,436,184 bytes.
  source: acceptance test 2026-10-19, run 20261019-0751_vivi-klein4b-runpod_8992
- Musubi 9B on an NVIDIA A40 (48 GB): at 1024-class resolution and rank 32,
  training without gradient checkpointing ran out of memory even at
  batch_size 1, and batch_size above 1 (a GPU micro-batch in Musubi) ran out of
  memory before step 1. One observation, not a rule: with these settings on an
  A40, suggest gradient_checkpointing: true and batch_size 1 with gradient
  accumulation for the effective batch, and say why; the user decides.
  source: Kura's former A40 guard (added 2026-07-04, moved here 2026-10-14); the observing run was not recorded
- Upstream's recommended timestep sampling: Musubi's own FLUX.2 guide trains
  with `timestep_sampling: flux2_shift` and `weighting_scheme: none`, and
  names `flux2_shift` as recommended for FLUX.2; it sets no
  `discrete_flow_shift`. Kura requires `timestep_sampling` for
  `architecture: flux2` because the trainer's generic default is `sigma`.
  source: upstream (Musubi Tuner flux_2.md guide at the pinned v0.3.5)
- Caption style for this family's Qwen3 text encoder (tag lists or natural
  language): not recorded yet; no recorded FLUX.2 run compared the two.
  source: acceptance test 2026-10-19 (the card had no entry)
- No evaluated task entries yet.

## Model files

- musubi-tuner `architecture: flux2` Klein Base 4B trained from Kura's model
  bundle, which downloads `dit`, `vae` and `text_encoder` from
  `Comfy-Org/vae-text-encorder-for-flux-klein-4b`. Write it in the run's
  `backend.config` next to your other settings:

  ```yaml
  backend:
    name: musubi-tuner
    config:
      architecture: flux2
      model_version: klein-base-4b
      model_bundle: flux2-klein-base-4b
  ```

  These are the files of Kura's real smoke `musubi-flux2-klein-4b`, which
  completed an optimizer step with a published adapter on RunPod, and of a
  100-step acceptance run on RunPod. The bundle pins no revision, so each
  download takes the repository's current files. The other bundles
  (`flux2-klein-4b`, `flux2-klein-base-9b`, `flux2-klein-9b`) exist, but no
  recorded run trained from them.
  source: real smoke musubi-flux2-klein-4b (evidence musubi-flux2-klein-4b-2026-09-29); acceptance test 2026-10-19, run 20261019-0751_vivi-klein4b-runpod_8992
