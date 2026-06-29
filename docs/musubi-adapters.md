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

If an architecture is not listed as built-in below, do not say that Musubi Tuner
does not support it. Say that Kura does not yet have a built-in Musubi command
generator for it.

## Current Kura built-in adapters

| Architecture | Kura built-in adapter | Image smoke | Real smoke | Notes |
| --- | --- | --- | --- | --- |
| FLUX.2 / FLUX.2 klein | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | verified in prior local/RunPod runs | `architecture: flux2` or `flux_2` |
| Wan | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: wan` |
| Krea 2 | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | verified in prior RunPod runs | `architecture: krea2` or `krea_2` |
| Qwen-Image | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: qwen_image` |
| Z-Image | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: zimage` or `z_image` |
| FLUX.1 Kontext | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: flux_kontext` or `flux1_kontext` |
| Ideogram 4 | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: ideogram4` or `ideogram_4` |
| HiDream-O1-Image | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: hidream_o1` or `hidream` |
| HunyuanVideo | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: hunyuan_video` or `hunyuanvideo` |
| HunyuanVideo 1.5 | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: hunyuan_video_1_5` |
| FramePack | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: framepack` or `frame_pack` |
| Kandinsky 5 | yes | passed `kura doctor musubi` on `kura/musubi-tuner:dev` | unverified | `architecture: kandinsky5` or `kandinsky_5` |

This list should be checked against the current upstream Musubi Tuner README
before adding adapters.

## Escape hatch

For an upstream-supported architecture without a Kura built-in adapter, a run may
still use `backend_overrides.musubi-tuner.command` to provide the exact command.
That keeps Kura responsible for workspace files, Docker/RunPod execution,
monitoring, downloads, and cleanup, while the Musubi command itself is explicit.

Use this as a temporary escape hatch, not as a substitute for adding adapters for
commonly used architectures.
