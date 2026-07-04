---
name: training-parameter-planning
description: Choosing training parameters and resource trade-offs for Kura LoRA runs. Use when proposing or reviewing run.yaml training parameters, judging VRAM fit from `kura run plan` Resources, reacting to CUDA OOM, choosing fp8/quantized artifacts, or deciding which memory-saving or quality trade-off to apply and what needs user approval.
---

# Training parameter planning

This skill is where training judgment lives. The owner decision that defines it:

> Code provides facts and irreversible-accident guards. The agent makes the
> trade-off decisions. Kura's value is that a light user gets a quite-good
> configuration without thinking; the price of that is transparency, not
> extra questions. "It runs" is not the goal — the goal is the best result
> the hardware allows.

Corollaries:

- Rich VRAM buys speed and quality headroom. Never apply memory-saving
  options "just in case" — an H100 run with gradient checkpointing enabled
  for no reason is as wrong as an OOM.
- Recommended parameters depend on the model **and** the task (character vs
  style vs concept vs paired-edit vs video). There is no single good default.
- Knowledge cards are starting points, not law. Your own current knowledge
  may be better than a card; deviating is allowed and must be stated.

## The one approval gate

There is exactly **one mandatory approval: the `kura run plan` review before
launch** (an AGENTS.md rule). Editing `run.yaml` is drafting, not commitment —
you may draft freely. Do not add per-knob approval questions on top of the
plan review; that destroys the light-user experience. The items listed under
"ask first" below are the only ones that need their own conversation *before*
they appear in a proposed plan.

## Inputs (facts, not guesses)

Gather these before proposing parameters:

1. `uv run kura run plan <run-id>` — the Resources section (local GPU and
   VRAM, executor, architecture, artifact filenames, memory flags) and the
   model download estimate.
2. The dataset facts, gathered **before** proposing anything: item count,
   resolution distribution, caption statistics (empty / duplicates / trigger
   word occurrences), pair integrity, task type. Use `kura dataset inspect`
   once it exists; until then use `kura dataset validate` plus a manual look,
   and the `dataset-prep` skill. Never propose parameters without this
   material.
3. Knowledge cards: read **only** the cards that match this run —
   `knowledge/<architecture>.md` (the architecture string from the plan's
   Resources section) and `knowledge/user-preferences.md`. Do not bulk-read
   the whole knowledge directory. If no exact card exists and the architecture
   is a video adapter (`wan`, `hunyuan_video`, `hunyuan_video_1_5`,
   `framepack`, `kandinsky5`), read `knowledge/video-architectures.md` as a
   placeholder and treat it as a weak starting point.
4. Prior evidence: `runs/` history for the same architecture on this
   hardware — successful configs, OOM failures, observed speed. Real outcomes
   on this machine beat any rule of thumb.
5. Backend mechanics: `musubi-tuner-backend` / `ai-toolkit-backend` skills for
   flag names, constraints, and interactions.

## Building the proposal

Assemble each parameter from the first source that covers it:

1. An explicit user instruction in this conversation.
2. `knowledge/user-preferences.md` — the user's own tested preferences
   outrank Kura baselines when they cover this model/task.
3. A `source: run <id>` entry for the same architecture × task in the
   architecture card — verified evidence from this workspace.
4. The architecture card's baseline values.
5. Your own current knowledge, for anything the cards do not cover.

You may deviate from sources 2–4 when your current knowledge says there is a
better choice: state the deviation and the reason in one short line as part
of the proposal. No separate approval is needed beyond the normal plan
review — but never rewrite a card to match your opinion (see the update
rules below), and quality-touching deviations follow the same ladder rules
as any other quality change.

The proposal shown to the user must include a one-line provenance summary,
e.g. `lr: your stated preference · rank/batch: Kura baseline (sdxl ×
character) · fp8_t5: verified in run 20260701-0126…`. Light users just say
yes; advanced users can drill into any value.

## Fit check and the adjustment ladder

Estimate whether the proposal fits the detected GPU (or the RunPod GPU class
for remote runs). This is class-based reasoning using card VRAM notes and
prior runs — state your confidence.

**If it fits with headroom: stop optimizing.** Spend headroom the way the
card recommends (e.g. larger batch where it helps quality), not on
unnecessary safety margins.

**If it does not fit, walk the ladder in order** and stop at the first
sufficient rung:

- **Rung 1 — meaning-preserving.** Artifact variants that are established
  quality-neutral for LoRA training (fp8 DiT / fp8 text encoder where the
  card says so), reuse of cached files. Propose freely; report what was
  chosen and why.
- **Rung 2 — speed-only.** In order: `gradient_checkpointing`; micro-batch
  reduction **with** a matching `gradient_accumulation` increase so the
  effective batch is preserved; then offload/swap (`blocks_to_swap`, CPU
  offload). Gradient checkpointing and accumulation may go straight into the
  proposal with the expected slowdown stated. Offload/swap — or anything
  expected to slow training beyond roughly 2x — needs the user's go-ahead
  first.
- **Rung 3 — quality-touching.** Resolution, effective batch size, rank,
  learning rate, training precision below established practice, dataset
  reduction. **Never silently.** Present two or three concrete options with
  trade-offs, recommend one so a light user can simply accept it, and
  include the alternative of a larger RunPod GPU with approximate cost. For
  character LoRAs, 768px is often a sufficient starting point; use 1024px or
  higher only when the task/model and hardware headroom justify the cost.

After launch approval: if OOM still occurs, diagnose from the actual log and
move exactly one rung. If the approved plan recorded a contingency envelope
in `run.yaml` (e.g. "on OOM, enable gradient checkpointing and relaunch"),
you may relaunch within that envelope without a second plan review; anything
outside the envelope goes back through recompile and plan approval. When you
expect an OOM risk, propose the envelope as part of the original plan — that
is what keeps the approval count at one.

## Last look (regret reminder)

Immediately before presenting the plan for approval, read
`knowledge/regrets.md` and check the run against it. This is not a review
and not a gate — hard constraints:

- Last look does **not** modify the plan or `run.yaml`.
- Last look does **not** return a launch verdict.
- Last look returns at most a few lines of note attached to the plan
  summary, phrased as `trigger -> reminder`, never `trigger -> block`.

Example tone: "Note: the trigger word 'myaku' appears in 0/40 captions —
intentional?" or "A past run with these conditions was regretted (forced
low-VRAM + heavy block swap, ~20 s/step). Continue if intentional." The user
decides; you just make sure the regret is visible at the moment they are
already looking.

## Autonomy boundaries

- **Propose freely (provenance line always):** card lookup and selection,
  rung 1 artifact choices, gradient checkpointing,
  accumulation-preserving micro-batch changes, deviations from cards with a
  stated reason.
- **Ask before proposing:** offload/swap or any >~2x slowdown, every rung 3
  option, extrapolating an untested hypothesis into an expensive run,
  changing the RunPod GPU class (cost).
- **Never:** changing anything between the approved plan and launch; leaving
  an applied trade-off or card deviation out of the plan discussion and
  `notes.md`.

## Knowledge cards

Layout under this skill directory:

- `knowledge/<architecture>.md` — one card per architecture id as it appears
  in the plan Resources section (e.g. `krea2.md`, `sdxl.md`,
  `flux_kontext.md`). Task-type sections inside. Cards are self-contained;
  do not create family-wide cards that quietly cover several architectures.
- `knowledge/user-preferences.md` — this user's own tested preferences and
  tendencies. Outranks baselines. Personal by nature: entries move to a
  baseline card only when the owner says they should apply generally.
- `knowledge/regrets.md` — the mirror of the cards: things that were
  regretted after a run. Read at Last look; grows one line per real regret.

Every value carries a `source:` line — this is evidence, orthogonal to where
the file sits (location = precedence, source = why we believe it):

- `source: owner (<date>)` — stated by the owner.
- `source: run <run-id>` — verified by a run the user actually evaluated;
  the run's `notes.md` is the primary record, the card cites it.
- `source: upstream (<doc>)` — upstream documentation or examples.
- `source: agent (<date>)` — seeded from model knowledge; a hypothesis.
  Treat as a starting guess, never as a recommendation to defend.

### Update rules

1. Evidence is recorded in the run's `notes.md` first (params, hardware,
   flags, the user's quality judgment in their own words). That always
   happens; it needs no generalization decision.
2. Promote to a card only what generalizes: after the user has evaluated a
   run, add or refine the (architecture × task) entry citing
   `source: run <id>`. When unsure whether a result is general or
   dataset-specific, keep it in `notes.md` and add at most a
   `source: agent` hypothesis line to the card.
3. Never overwrite `owner`/`run` sourced values with `agent` opinion. If new
   evidence contradicts an entry, keep both lines with their sources —
   contradictions are data.
4. Keep cards terse and structured; they are lookup tables, not essays.
5. When the user expresses regret about a finished run ("I wish I'd
   noticed…"), add one `trigger -> reminder` line to `knowledge/regrets.md`
   citing the run id. Successes feed cards; regrets feed the regret list —
   the two halves of the same loop.
