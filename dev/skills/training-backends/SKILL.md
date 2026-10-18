---
name: training-backends
description: Develop or diagnose Kura training backend adapters. Use when changing backend compile output, native dataset projection, model acquisition and validation, container images, output publication, Resume behavior, or adapter support claims.
---

# Training backends

Use this skill for adapter work below `src/kura/backends/` and for the trainer
images those adapters execute.

## Start from live contracts

1. Run `uv run kura run capabilities <backend>` before authoring or changing
   `backend.config`.
2. Read the family card in `src/kura/shipped/knowledge/model-families/` and
   the records in `docs/backend-smoke-evidence.yaml` to see what has actually
   run, when, and with which adapter source and trainer image. A family has run only
   where a card or an evidence record says so.
3. When the work comes from an upstream upgrade or new support claim, inventory
   the upstream delta first (`backend-upgrade-audit`). Do not implement or test
   only the user-named model while other inventoried contracts remain
   unclassified.
4. Read only the selected backend reference:
   - [AI-Toolkit](references/ai-toolkit.md)
   - [Musubi Tuner](references/musubi-tuner.md)
   - [sd-scripts](references/sd-scripts.md)
5. When changing a pin or evaluating newly upstream-supported behavior, also
   use `backend-upgrade-audit` before implementation.

Do not put architecture-specific facts in this skill. Current selectors,
fields, limits, and what has run belong in the capability output, adapter
code, family cards, the evidence ledger, or sourced repository knowledge.

## Shared contract

- Trainers run only through Kura's Docker or RunPod executors, never directly
  on the host.
- Backends compile native inputs and command specifications; they do not
  launch, reconcile, or stop runs.
- Every authored field has a declared consumer. Reject unknown or inapplicable
  values instead of accepting and ignoring them.
- Keep dataset sources, model identity, generated commands, and expected
  outputs visible in immutable compile artifacts.
- Do not add hidden precision, quantization, checkpointing, offload, swap, or
  batching defaults. Record execution accommodations in `run.yaml` and expose
  their trade-offs in the plan.
- Validate model roles and output contents from structure or metadata where a
  filename alone is insufficient.
- Treat a native escape hatch as reviewed, recorded input—not first-class Kura
  support.
- Backend training mechanics do not own prompt semantics or quality claims.

## Evidence levels

Keep these claims separate:

1. compile tests prove Kura generated the intended native input;
2. image smoke proves the pinned image contains and starts the entrypoint;
3. launch smoke proves the compiled run reaches that entrypoint;
4. real smoke proves actual weights complete an optimizer update and publish
   the expected artifact;
5. lifecycle evidence additionally proves recovery, download, and cleanup.

A support claim requires upstream source evidence, a tested Kura compile
projection, and a pinned-image entrypoint smoke, and it names the highest level
actually observed. A family is presented to users as supported only after one
optimizer step completed on both local Docker and RunPod: the two executors
differ in transfer, collection, and state handling. One
representative smoke covers another selector only when their execution
contracts (entrypoint, dataset shape, model roles, cache stages, outputs) are
demonstrably the same and the upgrade inventory records that relationship. A
shared Python entrypoint alone is not equivalence. Record each real run in the
evidence ledger and in the family card.

## Resource and Resume boundaries

- Preserve the training recipe before applying execution accommodations.
- Use `training-parameter-planning` for quality, memory, runtime, and cost
  trade-offs.
- Use `kura run resume`; never describe loading a standalone trained adapter
  into a fresh optimizer as Resume.
- Read the compiled plan and `docs/commands.md` for the selected backend's
  current restoration level. Use only revision-matching evidence.

## Verification

Run focused adapter tests first, then the checks appropriate to the changed
contract:

```sh
uv run kura run capabilities <backend>
uv run kura run compile <run-id>
uv run kura doctor <backend>
uv run python -m unittest discover -s tests
```

Before a billed or large-download smoke, follow `local-disk-safety`,
`runpod-lifecycle` when applicable, and the approval rules in `AGENTS.md`.
