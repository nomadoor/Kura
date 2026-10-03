# Musubi Tuner adapter

Use this reference only for `backend.name: musubi-tuner`.

- Kura compiles dataset TOML, cache and training commands, model-role locks,
  validation commands, and output checks.
- Distinguish upstream architecture support from Kura's built-in adapter
  support. A built-in adapter must select the correct entrypoints, model roles,
  arguments, dataset shape, and validation rules.
- For an upstream path without a built-in adapter, an explicit reviewed command
  is an escape hatch, not a support claim.
- Resolve known bundles from declared model identity when possible. For
  explicit paths or downloads, validate each role from file structure or
  metadata rather than its name.
- Treat batch size as the GPU micro-batch and make gradient accumulation
  explicit when preserving effective batch.
- Dataset variants must remain distinct without duplicating the same sample
  pool merely to encode resolution choices.

When an adapter or image ref changes:

1. keep the registered entrypoint inventory synchronized with generated
   commands;
2. run `uv run kura doctor musubi` against the selected image;
3. add or update compile tests for the public `run.yaml -> resolved artifacts`
   seam;
4. clean up disposable launch-smoke containers;
5. require an actual optimizer update and validated output before promoting a
   path beyond image or entrypoint smoke.

Do not download weights merely to prove script presence. Plan real-smoke disk,
memory, executor, and cost from the concrete run before launch.
