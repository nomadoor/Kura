# sd-scripts adapter

Use this reference only for `backend.name: sd-scripts`.

- Kura owns the reviewed selector and nested dataset surface. Discover both
  through `uv run kura run capabilities sd-scripts`; upstream acceptance alone
  is not first-class support.
- Generate the upstream two-level dataset TOML and keep run-scoped disk caches
  below the run directory. Never make a native cache mutate the shared dataset
  tree.
- Keep model roles explicit and revisions immutable when available.
- Reject dynamic caption controls that would be silently lost by a frozen text
  cache.
- Record measured disk-cache estimates or a visible, reviewed exception.
- When native outputs require conversion, validate the source, conversion, and
  published artifact. Preserve failed native conversion inputs as recovery
  material, never as successful outputs.
- Keep model-patch artifacts distinct from trained adapters and validate their
  type-specific metadata before publication.
- Resume passes `--resume` and `--skip_until_initial_step` with the logical
  target as `--max_train_steps`. Upstream `train_network.py` sets `global_step`
  to the step's remainder within its epoch after skipping whole epochs, so step
  names, the save cadence, and the stop drift once the source step passes an
  epoch boundary. The image applies
  `docker/sd-scripts/patches/0002-resume-continues-the-logical-step.patch`, which
  keeps `global_step` and the progress bar logical; the training-state contract
  therefore declares `native_progress: logical`, and a Resume's save cadence is
  not capped. Locks compiled before the patch froze `process_local` and are read
  that way. Re-check both hunks whenever the sd-scripts pin moves, and drop the
  patch once upstream counts logical steps. A one-item smoke dataset cannot show
  this (one step per epoch leaves no remainder); test with several items and a
  target that is not a multiple of the steps per epoch.

Verification starts with:

```sh
uv run kura doctor disk
uv run kura doctor sd-scripts
uv run kura run capabilities sd-scripts
uv run kura run compile <run-id>
uv run kura run plan <run-id>
```
