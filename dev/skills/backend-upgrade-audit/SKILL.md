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

Before implementation or smoke selection, create or update the upgrade plan in
`docs/backend-validation/` following `docs/backend-validation.md`. Run
`uv run python scripts/check_backend_validation.py` after every material edit.
The plan, rather than the model requested for one smoke, owns the complete
inventory and all open gaps.

For every added family, variant, and changed execution contract, record:

| Upstream delta | Kura status | Required action | Evidence |
| --- | --- | --- | --- |
| exact feature or model | built-in, generic, escape hatch, unsupported | code, docs, or none | source, compile, image, or real smoke |

`Generic` is not automatically `supported`: confirm that Kura can represent the
dataset, model roles, native fields, outputs, and executor lifecycle without an
undeclared bypass. Explicitly classify capabilities outside Kura's current
image/video training contract instead of silently omitting them.

Complete this matrix before choosing representative real-smoke runs. If the
matrix reveals missing Kura behavior, either implement it through the selected
`training-backends` reference or mark it unsupported with the exact missing
capability.

Do not collapse image-only, video, audio-video, edit/control, reference,
teacher/student, or materially different cache paths into one model-family
row. These are separate execution contracts unless every contract field named
in `docs/backend-validation.md` is identical.

## Verification levels

Keep these claims distinct:

1. **Source audit** — the exact upstream delta is identified.
2. **Expressible** — Kura validates and compiles the required configuration.
3. **Image smoke** — the pinned image contains and starts the relevant code.
4. **Real smoke** — actual weights complete at least one optimizer step and
   materialize the expected artifact through Kura. This raises confidence but
   is not required for every upstream-supported contract when Kura's source,
   compile, and pinned-image evidence is complete.
5. **Operationally verified** — recovery, download, and executor cleanup are
   confirmed where that scope is claimed.

A representative real smoke does not promote untested families, modes, or
dataset shapes. Conversely, exhaustive real smokes are not required when
several checkpoints share an unchanged execution contract; link them with
`covered_by` and record the exact equivalence rationale. The validation checker
rejects cross-contract coverage and cycles.

## Completion gate

Do not report an upgrade complete until:

- old/new identities and provenance are recorded;
- the full upstream delta and Kura impact matrix are documented;
- affected built-in adapters, generic projections, and support claims agree;
- focused tests, backend doctor/image smoke, and the repository release gate
  pass at the evidence level being claimed;
- requested real smokes reach a terminal state with artifacts and executor
  cleanup verified;
- unsupported or unverified additions remain visibly classified.
- every supported contract has source, compile, and pinned-image evidence;
  contracts without optimizer evidence retain an explicit `runtime_note`;
- changed executor lifecycle paths have runtime evidence;
- the plan status is `complete`, contains no implementation `gap`, and
  `scripts/check_backend_validation.py` passes.

An `in_progress` plan is the required truthful state while Kura cannot yet
faithfully validate or compile a supported contract, the pinned image cannot
start its entrypoint, or a changed lifecycle path lacks runtime evidence. Do
not turn a compile-verified row into a claim that real training or output
quality was observed; the generated table must preserve that distinction.

When cost-bearing validation is needed, finish the no-cost audit first, then
show the concrete `kura run plan` and obtain the approval required by
`AGENTS.md`.
