# Musubi Tuner adapters

Musubi Tuner and Kura's Musubi backend are not the same layer.

- **Musubi Tuner support** means the upstream tool has training scripts,
  arguments, and model conventions for an architecture.
- **Kura built-in adapter support** means Kura can generate the Musubi command,
  cache commands, model lock, validation, and output checks from `run.yaml`.

If an architecture is not listed as built-in below, do not say that Musubi Tuner
does not support it. Say that Kura does not yet have a built-in Musubi command
generator for it.

## Current Kura built-in adapters

| Architecture | Kura built-in adapter | Notes |
| --- | --- | --- |
| FLUX.2 / FLUX.2 klein | yes | `architecture: flux2` or `flux_2` |
| Wan | yes | `architecture: wan` |
| Krea 2 | yes | `architecture: krea2` or `krea_2` |
| HunyuanVideo | not yet | Upstream Musubi supports it; adapter not implemented in Kura yet |
| HunyuanVideo 1.5 | not yet | Upstream Musubi docs exist; adapter not implemented in Kura yet |
| FramePack | not yet | Upstream Musubi supports it; adapter not implemented in Kura yet |
| FLUX.1 Kontext | not yet | Upstream Musubi supports it; adapter not implemented in Kura yet |
| Qwen-Image | not yet | Upstream Musubi supports it; adapter not implemented in Kura yet |
| Z-Image | not yet | Upstream Musubi supports it; adapter not implemented in Kura yet |
| Ideogram 4 | not yet | Upstream Musubi has experimental support; adapter not implemented in Kura yet |
| HiDream-O1-Image | not yet | Upstream Musubi has experimental support; adapter not implemented in Kura yet |
| Kandinsky 5 | not yet | Upstream Musubi docs exist; adapter not implemented in Kura yet |

This list should be checked against the current upstream Musubi Tuner README
before adding adapters.

## Escape hatch

For an upstream-supported architecture without a Kura built-in adapter, a run may
still use `backend_overrides.musubi-tuner.command` to provide the exact command.
That keeps Kura responsible for workspace files, Docker/RunPod execution,
monitoring, downloads, and cleanup, while the Musubi command itself is explicit.

Use this as a temporary escape hatch, not as a substitute for adding adapters for
commonly used architectures.
