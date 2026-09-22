# Backend upgrade validation

Backend upgrades are validated from an authored inventory, not from the model
chosen for a smoke run. Each upgrade has one YAML plan in
`docs/backend-validation/`. The plan records what changed upstream, the Kura
execution contracts affected by those changes, the evidence collected for each
contract, and every remaining gap.

This is deliberately separate from the two existing records:

- `backend-support.md` states current support conclusions.
- `backend-smoke-evidence.yaml` records identity-bound runtime observations.
- a backend-validation plan explains why those conclusions and observations do
  or do not cover the complete upstream delta.

Run `uv run python scripts/check_backend_validation.py` to validate all plans.
Each YAML plan has a generated Markdown table beside it so the contract matrix
can be reviewed without reading YAML. Regenerate tables with
`uv run python scripts/check_backend_validation.py --write`; the normal checker
and release gate fail when a table is missing or stale. The YAML remains the
only authored source of truth.

## Workflow

1. Freeze the exact old and new backend identities, including the embedded
   source commit and immutable image digest when applicable.
2. Compare the exact source range and authoritative release notes. Add every
   material model, mode, dataset, cache, output, dependency, and lifecycle
   change to `changes`.
3. Classify every change through at least one `contract`. Do this before
   choosing cost-bearing smoke runs.
4. Give each supported contract source, compile, and image evidence. Record
   optimizer evidence when observed; otherwise add a concrete `runtime_note`
   so compile-verified support is never mistaken for a completed training run.
   When the executor lifecycle changed, lifecycle evidence remains required.
5. Mark capabilities outside Kura's contract `unsupported` and name the exact
   `missing_capability`. Use `not_applicable` only with a rationale.
6. A representative optimizer smoke may be referenced with `covered_by` only
   when the native execution contract is identical. The plan must explain the
   equivalence in `coverage_rationale`.
7. Change `upgrade.status` to `complete` only after all supported contracts
   have the required source, compile, and image evidence, changed lifecycle
   paths have runtime evidence, and no implementation gaps remain. Visible
   `runtime_note` entries may remain for contracts without optimizer smokes.

## Contract identity

The checker treats these fields as the execution-contract identity:

- backend and model family;
- training mode;
- dataset kind and conditioning shape;
- adapter kind and native entrypoints;
- model roles and cache stages;
- output contract;
- whether executor lifecycle behavior changed.

The variant label is descriptive and is intentionally excluded from this
identity. Different weight variants may share evidence only when all fields
above are equal. A family, mode, image/video shape, conditioning path, role,
cache, output, or lifecycle difference is a different contract and cannot
inherit a representative smoke.

## Schema

The top-level shape is:

```yaml
schema_version: 1
upgrade:
  id: stable-upgrade-id
  status: in_progress # or complete
  backends:
    - name: backend-name
      from: {kind: git-commit, value: old-identity}
      to: {kind: git-commit, value: new-identity}
      source_refs: [authoritative-source-range]
changes:
  - id: stable-change-id
    backend: backend-name
    summary: Exact material upstream change
    kind: execution-contract # enhancement or outside-contract are also valid
    source_refs: [authoritative-source]
contracts:
  - id: stable-contract-id
    change_ids: [stable-change-id]
    backend: backend-name
    family: family-name
    variant: base
    mode: text-to-image
    dataset: {kind: image, conditioning: none}
    execution:
      adapter: generic # built-in or escape-hatch are also valid
      entrypoints: [native-entrypoint]
      model_roles: [transformer, text_encoder, vae]
      cache_stages: [latents]
      output: lora-safetensors
      lifecycle_changed: false
    disposition: support
    evidence:
      source: [source-audit-reference]
      compile: [test-reference]
      image: [image-smoke-reference]
      optimizer: []
      lifecycle: []
    runtime_note: Real optimizer execution has not been observed.
```

For `unsupported`, replace `execution` with `missing_capability`. For
`not_applicable`, provide `rationale`. Both still require source evidence.

Source labels are human-auditable references. Compile evidence uses
`repository/path.py::ClassName.test_method`; both the file and symbol must
exist. Image, optimizer, and lifecycle lists use identity-bound IDs from
`backend-smoke-evidence.yaml`. Image references must be passed parser-or-higher
observations; optimizer references must be passed `real-optimizer-step`
observations; lifecycle references must be passed runtime-or-higher
observations. Every record must belong to the same backend. The checker rejects
unknown, failed, weaker, or cross-backend evidence. Optimizer evidence raises
confidence but is not the support boundary: Kura owns faithful validation and
compilation of an upstream-supported contract, while the upstream trainer owns
the implementation behind that compiled contract. A supported contract without
optimizer evidence must retain a `runtime_note` in the generated table.

Plans are permanent audit records. Once an upgrade is complete, keep its plan
and add a new one for the next identity transition rather than rewriting the
old comparison.
