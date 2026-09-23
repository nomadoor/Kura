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

Verification starts with:

```sh
uv run kura doctor disk
uv run kura doctor sd-scripts
uv run kura run capabilities sd-scripts
uv run kura run compile <run-id>
uv run kura run plan <run-id>
```
