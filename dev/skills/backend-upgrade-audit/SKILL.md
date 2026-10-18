---
name: backend-upgrade-audit
description: Audit Kura trainer upgrades and new support claims from complete upstream inventory through evidence. Use for a pinned backend version, image, source-ref, model-family, mode, dataset, entrypoint, dependency, or runtime-contract change, before implementation or any claim that newly upstream-supported training works in Kura.
---

# Backend Upgrade Audit

Use this skill whenever a Kura training backend's pinned version, image digest,
Git ref, or embedded source commit changes. Also use it when evaluating or
claiming a newly added upstream family, variant, training mode, dataset shape,
or runtime contract even if the pin has not changed. Also use
`training-backends` for compile and runtime mechanics.

An upgrade target named by the user is a validation target, not permission to
limit discovery to that model. Inventory the complete upstream delta before
calling the backend updated.

## Establish the comparison

- Record the exact old and new identities: version or tag, immutable digest,
  and embedded source commit when available.
- Use authoritative upstream release notes and compare the exact source commits.
  Do not infer the delta from a mutable README or the requested smoke model.
- If an image tag cannot be tied to source, report that provenance gap. Do not
  describe the comparison as exhaustive.

## Inventory the full delta

Inspect and classify every material change between the two identities:

- added, removed, or renamed model families and variants;
- new training modes, dataset shapes and fields, cache stages, and entrypoints;
- changed CLI/config defaults, validation, precision, quantization, offload, or
  optimizer behavior;
- model acquisition, companion weights, artifact formats, output validation,
  and Resume implications;
- Python, CUDA, PyTorch, attention-kernel, and container-runtime changes;
- executor or provider API assumptions that affect launch, observe, recover,
  download, stop, or billing safety.

Keep existing-model enhancements separate from genuinely new architectures.
For registries or plugin catalogs, mechanically compare the old and new
registrations instead of sampling notable files.

## Build the Kura impact matrix

Before implementation or smoke selection, write the inventory in the pull
request body, or, when it is too long for that, in a note under
`docs/smoke-evidence/` that the pull request links. The inventory, rather than
the model requested for one smoke, owns the complete delta and all open gaps.

For every added family, variant, and changed execution contract, record:

| Upstream delta | Kura status | Required action | Evidence |
| --- | --- | --- | --- |
| exact feature or model | built-in, generic, escape hatch, unsupported | code, docs, or none | source, compile, image, or real run |

`Generic` is not automatically `supported`: confirm that Kura can represent the
dataset, model roles, native fields, outputs, and executor lifecycle without an
undeclared bypass. Explicitly classify capabilities outside Kura's current
image/video training contract instead of silently omitting them.

Complete this matrix before choosing representative real runs. If the matrix
reveals missing Kura behavior, either implement it through the selected
`training-backends` reference or mark it unsupported with the exact missing
capability.

Do not collapse image-only, video, audio-video, edit/control, reference,
teacher/student, or materially different cache paths into one model-family
row. These are separate execution contracts: entrypoint, dataset shape, model
roles, cache stages, and outputs all have to be the same before one run speaks
for another.

## Verification levels

Keep these claims distinct:

1. **Source audit** — the exact upstream delta is identified.
2. **Expressible** — Kura validates and compiles the required configuration.
3. **Image smoke** — the pinned image contains and starts the relevant code.
4. **Real smoke** — actual weights complete at least one optimizer step and
   materialize the expected artifact through Kura. Lower levels may be
   recorded, but a family is presented to users as supported only after this
   step ran on both local Docker and RunPod (`training-backends`).
5. **Operationally verified** — recovery, download, and executor cleanup are
   confirmed where that scope is claimed.

A representative real run does not promote untested families, modes, or
dataset shapes. Conversely, exhaustive real runs are not required when several
checkpoints share an unchanged execution contract; say which run covers which
checkpoint, and why, in the inventory.

## Completion gate

Do not report an upgrade complete until:

- the old and new pins (version or tag, image digest, source commit) are stated
  in the pull request;
- the full upstream delta and Kura impact matrix are written down, in the pull
  request body or a linked `docs/smoke-evidence/` note;
- affected built-in adapters, generic projections, and family cards agree;
- every family the pull request calls runnable has one real run on the new
  image, recorded in `docs/backend-smoke-evidence.yaml` (with
  `uv run python scripts/real_smoke.py evidence`) and in a `Recorded:` line of
  that family's card under `src/kura/shipped/knowledge/model-families/`; a
  family called supported has run on both local Docker and RunPod (`AGENTS.md`);
- a family Kura only compiles is called untested in its card, or not listed;
- requested real runs reach a terminal state with artifacts and executor
  cleanup verified, and changed executor lifecycle paths have runtime evidence;
- `uv run python scripts/check_release.py` passes.

Do not turn a compile-verified family into a claim that real training or output
quality was observed.

When cost-bearing validation is needed, finish the no-cost audit first, then
show the concrete `kura run plan` and obtain the approval required by
`AGENTS.md`.
