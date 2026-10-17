# zimage

## Training

- Backend of the entries below: musubi-tuner v0.3.5, `architecture: zimage`.
  They were not measured on AI-Toolkit's `zimage`.
- VRAM class: a LoRA at 768 px, rank 32 / alpha 16, learning rate 1e-4 with a
  cosine schedule, adamw8bit, `timestep_sampling: shift`,
  `discrete_flow_shift: 2.0`, `fp8_base` + `fp8_scaled` + `fp8_llm`, and
  gradient checkpointing completed 600 steps locally on an RTX 4070 Ti with
  12 GB VRAM and no block swap, at about 3.25 seconds per step.
  source: owner (2026-10-18)

## Model files

- musubi-tuner `architecture: zimage` trained from these files. Write them in
  the run's `backend.config` next to your other settings:

  ```yaml
  backend:
    name: musubi-tuner
    config:
      architecture: zimage
      model_downloads:
        dit:
          repo: Comfy-Org/z_image
          filename: split_files/diffusion_models/z_image_bf16.safetensors
        vae:
          repo: Comfy-Org/z_image
          filename: split_files/vae/ae.safetensors
        text_encoder:
          repo: Comfy-Org/z_image
          filename: split_files/text_encoders/qwen_3_4b.safetensors
  ```

  These are the files of Kura's real smoke `musubi-zimage`, which completed
  optimizer steps with a published adapter on RunPod and on local Docker. The
  entries pin no revision, so each download takes the repository's current
  files; add `revision:` to each entry to keep a run on the files you
  checked.
  source: real smoke musubi-zimage (evidence musubi-zimage-2026-10-01-1458, musubi-zimage-docker-2026-10-07)

## character

- no entry yet.

## style

- no entry yet.
