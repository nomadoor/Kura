---
name: training-parameter-planning
description: Choosing training parameters and resource trade-offs for Kura LoRA runs. Use when proposing or reviewing run.yaml training parameters, judging VRAM fit from `kura run plan` Resources, reacting to CUDA OOM, choosing fp8/quantized artifacts, or deciding which memory-saving or quality trade-off to apply and what needs user approval.
---

# Training parameter planning

This skill is where training judgment lives. The owner decision that defines it:

> Code provides facts and irreversible-accident guards. The agent makes the
> trade-off decisions. Any adjustment that touches output quality requires user
> approval. "It runs" is not the goal — the goal is the best result the
> hardware allows.

Two corollaries:

- Rich VRAM buys speed and quality headroom. Never apply memory-saving options
  "just in case" — an H100 run with gradient checkpointing enabled for no
  reason is as wrong as an OOM.
- Recommended parameters depend on the model **and** the task (character vs
  style vs concept vs paired-edit vs video). There is no single good default.

## Inputs (facts, not guesses)

Gather all of these before proposing parameters:

1. `uv run kura run plan <run-id>` — the Resources section (local GPU and
   VRAM, executor, architecture, artifact filenames, memory flags) and the
   model download estimate.
2. The dataset: item count, resolution distribution, captions, task type.
   Use `kura dataset validate` and the `dataset-prep` skill.
3. `recipes.md` in this skill directory — accumulated per-(architecture ×
   task) parameter knowledge with provenance labels.
4. Prior evidence: `runs/` history for the same architecture on this
   hardware — successful configs, OOM failures, observed speed. Real outcomes
   on this machine beat any rule of thumb.
5. Backend mechanics: `musubi-tuner-backend` / `ai-toolkit-backend` skills for
   flag names, constraints, and interactions.

## Procedure

1. **Pin down the task.** Architecture, task type, and the user's quality/time
   priorities. If the task type is ambiguous, ask one focused question — do
   not run a survey.
2. **Start from knowledge, not from scratch.** Look up `recipes.md` for
   (architecture × task). Exact match: use it as the baseline. Near match:
   adapt it and say so. No match: derive from upstream docs plus the backend
   skill, and label the proposal unverified.
3. **Estimate fit.** Compare the recipe's VRAM class against the detected GPU
   (or the RunPod GPU class for remote runs). This is class-based reasoning,
   not arithmetic — state your confidence, and prefer prior-run evidence on
   the same hardware when it exists.
4. **If it fits with headroom: stop optimizing.** Propose the recipe as-is.
   Consider spending the headroom the way the recipe recommends (e.g. larger
   batch where it helps quality), not on unnecessary safety margins.
5. **If it does not fit, walk the ladder in order** and stop at the first
   sufficient rung:
   - **Rung 1 — meaning-preserving.** Artifact variants that are established
     quality-neutral for LoRA training (fp8 DiT / fp8 text encoder where
     recipes or the backend skill say so), reuse of cached files. Apply
     freely; always report what was chosen and why.
   - **Rung 2 — speed-only.** In order: `gradient_checkpointing`; micro-batch
     reduction **with** a matching `gradient_accumulation` increase so the
     effective batch is preserved; then offload/swap (`blocks_to_swap`, CPU
     offload). Gradient checkpointing and accumulation may be applied without
     asking, but the report must state the expected slowdown. Offload/swap —
     or anything expected to slow training beyond roughly 2x — needs the
     user's go-ahead first.
   - **Rung 3 — quality-touching.** Resolution, effective batch size, rank,
     learning rate, training precision below established practice, dataset
     reduction. **Never apply silently.** Present two or three concrete
     options with their trade-offs, and include the alternative of running
     remote on a larger GPU with its approximate cost. The user picks.
6. **Plan, approve, launch.** Show `kura run plan` output and get explicit
   approval before launching (this is also an AGENTS.md rule). If OOM still
   occurs, diagnose from the actual log, move exactly one rung, record the
   change in `run.yaml`, recompile, and show the plan again.

## Autonomy boundaries

- **Decide freely, always report:** recipe selection, rung 1 artifact
  choices, gradient checkpointing, accumulation-preserving micro-batch
  changes.
- **Ask first:** offload/swap or any >~2x slowdown, every rung 3 option,
  extrapolating an unverified recipe into an expensive run, changing the
  RunPod GPU class (cost).
- **Never:** changing anything between the approved plan and launch; leaving
  an applied trade-off out of the run report or `notes.md`.

## Knowledge accumulation loop

`recipes.md` is Kura data, expected to grow more valuable than any single
feature. After a run **the user has actually evaluated**:

1. Append or refine the (architecture × task) entry: parameters used,
   hardware, memory flags, the user's quality judgment in their own words,
   and the run id as evidence.
2. Provenance labels: `owner` (stated by the owner), `verified` (evaluated
   successful run, cite run id), `unverified` (seeded or derived).
3. Never overwrite an `owner` or `verified` entry with speculation. If new
   evidence contradicts an entry, keep both with run ids and note the
   conflict — contradictions are data.
4. Keep entries terse and structured. `recipes.md` is a lookup table, not an
   essay.
