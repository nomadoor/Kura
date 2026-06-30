# Musubi Tuner adapters

Musubi Tuner and Kura's Musubi backend are not the same layer.

- **Musubi Tuner support** means the upstream tool has training scripts,
  arguments, and model conventions for an architecture.
- **Kura built-in adapter support** means Kura can generate the Musubi command,
  cache commands, model lock, validation, and output checks from `run.yaml`.
- **Image smoke** means the configured Docker image contains the scripts used by
  those adapters and each script can start its `--help` path. Run
  `uv run kura doctor musubi` after changing `MUSUBI_TUNER_REF` or rebuilding the
  image.
- **Real smoke** means at least one tiny training run has actually launched for
  that adapter. If this has not been done, call the adapter experimental or
  unverified rather than simply "done."

As of 2026-06-30, all adapters listed below passed image smoke on
`nomadoor/kura-musubi-tuner:dev`: the configured Docker image contains the
expected Musubi scripts and each script can start its `--help` path. Earlier
Docker launch smoke also proved that Kura can compile and start adapter commands
against the Musubi entrypoints, but dummy-path launch smoke is not a real
one-step training proof.

Real one-step smoke is tracked separately. It must use actual model files and
finish one optimizer step through Kura's normal Docker or RunPod executor. The
developer runner is `uv run python scripts/musubi_real_smoke.py <architecture>`.
Use `docs/musubi-real-smoke-plan.md` to choose the first executor/GPU before
running expensive smoke tests.

If an architecture is not listed as built-in below, do not say that Musubi Tuner
does not support it. Say that Kura does not yet have a built-in Musubi command
generator for it.

## Current Kura built-in adapters

| Architecture | Kura built-in adapter | Image smoke | Real smoke | Notes |
| --- | --- | --- | --- | --- |
| FLUX.2 / FLUX.2 klein | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | verified in prior local/RunPod runs | `architecture: flux2` or `flux_2` |
| Wan | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-06-30 (`scripts/musubi_real_smoke.py wan --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: wan` |
| Krea 2 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-06-30 (`scripts/musubi_real_smoke.py krea2`) | `architecture: krea2` or `krea_2` |
| Qwen-Image | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed RunPod A40 1-step on 2026-06-30 (`scripts/musubi_real_smoke.py qwen_image --executor runpod --gpu "NVIDIA A40"`); A5000 reached training start but was SIGKILLed, likely OOM for that 256px/fp8/block-swap recipe | `architecture: qwen_image` |
| Z-Image | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py zimage --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: zimage` or `z_image`; use upstream `qwen_3_4b.safetensors` text encoder |
| FLUX.1 Kontext | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py flux_kontext --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: flux_kontext` or `flux1_kontext`; requires paired/control dataset entries |
| Ideogram 4 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py ideogram4 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: ideogram4` or `ideogram_4` |
| HiDream-O1-Image | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py hidream_o1 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: hidream_o1` or `hidream`; use BF16 training checkpoint, not Comfy fp8-scaled checkpoint |
| HunyuanVideo | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py hunyuan_video --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: hunyuan_video` or `hunyuanvideo`; use a standard fp16 LLaMA text encoder with `fp8_llm` rather than Comfy fp8-scaled LLaMA weights |
| HunyuanVideo 1.5 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py hunyuan_video_1_5 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: hunyuan_video_1_5` |
| FramePack | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 (`scripts/musubi_real_smoke.py framepack --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: framepack` or `frame_pack` |
| Kandinsky 5 | yes | passed `kura doctor musubi` on `nomadoor/kura-musubi-tuner:dev` | passed local Docker 1-step on 2026-07-01 with the Lite T2V model (`scripts/musubi_real_smoke.py kandinsky5 --executor docker --image nomadoor/kura-musubi-tuner:dev`) | `architecture: kandinsky5` or `kandinsky_5`; Pro 19B checkpoints remain capacity-dependent |

This list should be checked against the current upstream Musubi Tuner README
before adding adapters.

## Escape hatch

For an upstream-supported architecture without a Kura built-in adapter, a run may
still use `backend_overrides.musubi-tuner.command` to provide the exact command.
That keeps Kura responsible for workspace files, Docker/RunPod execution,
monitoring, downloads, and cleanup, while the Musubi command itself is explicit.

Use this as a temporary escape hatch, not as a substitute for adding adapters for
commonly used architectures.
