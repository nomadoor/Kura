# AI-Toolkit adapter

Use this reference only for `backend.name: ai-toolkit`.

- AI-Toolkit owns model acquisition and model-specific loading. Kura owns the
  declared dataset sources, run identity, training output directory, and common
  process envelope.
- Prefer typed top-level controls and typed dataset fields. Use
  `native_config` only for reviewed upstream input that has no first-class Kura
  field; it may not replace Kura-owned process, dataset, or model identity.
- `model.base` is the trained model (`process.model.name_or_path`), so the
  adapter's own validation requires it. The pinned ModelConfig has no revision,
  so the adapter refuses `model.revision` and records no revision in the
  requirement identity.
- Unset training settings are filled from the pinned UI baseline
  (`docs/adr/upstream-training-baseline.md`). This covers the scheduler,
  precision, quantization, optimizer, learning rate, and latent caching.
  `kura run plan` shows the filled values under `baseline`. Author a value only
  to change it deliberately; do not copy baseline values into `native_config`.
  An architecture without a baseline entry needs an authored
  `native_config.train.noise_scheduler` and `mixed_precision`.
- After a pin upgrade, regenerate the baseline with
  `uv run python scripts/extract_ai_toolkit_baseline.py` and review its diff.
- Keep Hugging Face cache paths configurable and inside the mounted workspace.
- Build and run through the pinned image. Record its immutable digest and
  embedded source commit; never treat a mutable tag as sufficient provenance.
- Tiny runs are infrastructure evidence, not training recipes or quality
  evidence.
- Verify output and protected training state through Kura's wrapper rather than
  assuming upstream exit success implies publication success.

Useful checks:

```sh
uv run kura image inspect ai-toolkit
uv run kura doctor docker
uv run kura run capabilities ai-toolkit
uv run kura run compile <run-id>
```
