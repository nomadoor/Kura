---
name: training-parameter-planning
description: Plan Kura training quality, memory, runtime, and cost trade-offs. Use when drafting or reviewing training run parameters, interpreting plan resources, reacting to OOM, or choosing execution accommodations and GPU capacity.
---

# Training parameter planning

Choose a useful training recipe, not merely one that starts. Preserve the
user's intent and make every quality, speed, memory, and cost trade-off visible
in `run.yaml` and the compiled plan.

## Approval model

Draft and revise freely. The mandatory launch approval is the final
`kura run plan <run-id>` review required by `AGENTS.md`.

Ask earlier only when a missing fact cannot be inferred and a wrong assumption
would waste a run, or when the proposal changes cost, quality, or expected
runtime materially. Batch necessary questions instead of asking one setting at
a time. Record answers in dataset or run intent so they are not asked again.

## Gather facts first

Before proposing parameters, inspect:

1. `uv run kura run capabilities <backend> --json` for the accepted authored
   vocabulary and selector conditions;
2. `uv run kura dataset inspect <dataset>` and dataset validation for item
   count, resolutions, captions, roles, and pair integrity;
3. `uv run kura run plan <run-id>` for model artifacts, executor, detected or
   selected GPU resources, cache state, and download estimates;
4. the matching `knowledge/model-families/<family>.md`, when present;
5. `knowledge/user-preferences.md` and prior evaluated runs under comparable
   conditions;
6. the selected reference in `training-backends` for backend mechanics.

Do not guess field names from adapter source. A capability refusal is the
public contract speaking.

## Build the proposal

Choose each setting from the strongest available source:

1. explicit user instruction;
2. recorded user preference covering this task;
3. an evaluated comparable run;
4. a sourced family card;
5. current technical judgment, identified as an inference.

Summarize provenance in the plan so a light user can approve the recommendation
without answering a questionnaire. Separate:

- the **recipe**: dataset, resolution, learning rate, rank, effective batch,
  optimizer, schedule, and update count;
- **execution accommodations**: precision or quantized artifacts,
  checkpointing, micro-batch/accumulation, offload, swap, cache, and GPU class.

## Fit and adjustment ladder

Use the smallest resource class that should satisfy the declared plan, based on
artifact sizes, prior runs, backend constraints, and the plan's resource facts.
State confidence. If the plan fits with headroom, stop adding memory-saving
options.

When it does not fit, stop at the first sufficient rung:

1. **Meaning-preserving:** compatible artifact variants and reusable caches
   established as neutral for the requested training contract.
2. **Execution-only:** checkpointing, a smaller micro-batch with accumulation
   preserving effective batch, then supported offload or swap. State the
   expected slowdown.
3. **Recipe-changing:** resolution, effective batch, rank, learning rate,
   model size, precision with quality impact, or dataset reduction. Present
   concrete alternatives and recommend one before editing the approved plan.

Changing GPU class or cost, applying a recipe-changing adjustment, or accepting
an expected slowdown beyond roughly twofold requires a new plan decision.
After an OOM, diagnose the actual log and move one rung; never silently change
multiple dimensions and retry.

## Last look

Immediately before presenting the approval plan, read
`knowledge/regrets.md`. Return only relevant `trigger -> reminder` notes. This
does not modify the run, issue a verdict, or create another approval gate.

## Resume

Treat Resume as a continuity-risk analysis, not a promise of exact numerical
equivalence.

- Use `kura run resume` and the protected training-state artifact.
- Read the plan's restoration level and its restored and missing components.
- Use numerical evidence only when backend identity, training envelope, and
  dataset conditions match.
- Keep Resume distinct from starting a new optimizer from trained weights.
- State uncertainty rather than transferring evidence across revisions or
  materially different datasets.

## Own execution through completion

After approval, run `kura run execute <run-id>` through the host's tracked
long-running mechanism unless the user explicitly requests detachment. Do not
replace it with an untracked background shell.

When execution returns:

1. reconcile when needed;
2. verify terminal `status.json`, exit code, realization, and logs;
3. confirm the expected artifact exists and is recorded;
4. for remote runs, confirm download and billing-resource cleanup through
   `runpod-lifecycle`;
5. report mechanical completion separately from output quality.

Run multiple approved trainings sequentially. Stop on failure, unknown state,
or a new decision. Losing the tracked session is not evidence of completion and
never authorizes relaunch.

## Knowledge feedback

Record settings, hardware, observed behavior, and the user's judgment in the
run's `notes.md` first. Promote only generalizable findings to the matching
family card, citing the run. Record an actual owner-stated regret as one concise
entry in `knowledge/regrets.md`; do not seed hypothetical regrets.
