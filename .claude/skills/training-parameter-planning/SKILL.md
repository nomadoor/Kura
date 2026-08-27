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

## Own the run through completion

Launching an approved training run starts the waiting phase; it does not
complete the training request. Unless the user explicitly asks to start and
detach:

1. Run `kura run execute <run-id>` with the current agent host's tracked
   long-running execution mechanism. It must keep the agent task active across
   ordinary turns and return control when the command exits. Do not replace it
   with an untracked detached command such as `nohup ... &`.
   Examples are Claude Code's tracked background Bash execution and Codex Goal
   mode for long-running work; use them only when the current host exposes the
   corresponding mechanism, and treat the capability requirements above as
   authoritative.
2. While execution is active, wait for the host's completion event. Do not wake
   the model on a timer merely to pass time. Perform periodic log or metric
   review only when the user requests it or concrete evidence requires
   diagnosis.
3. When execution returns, verify the mechanical result before reporting it:
   reconcile when needed; read `status.json`; confirm the terminal state and
   exit code; check the latest realization and logs for a conflicting failure;
   and confirm the expected training artifact exists. This establishes that
   training completed mechanically, not that the result has good quality.
4. Send the final training report only after that verification. Quality claims
   require the separate evaluation flow and user judgment; training approval
   does not authorize an automatic render.

For multiple training runs explicitly approved in the conversation, repeat
this contract one run at a time. Start the next run only after the previous run
is mechanically complete. Stop on failure, unknown or inconsistent state, or
when a new decision is required. Agent recovery notes may identify the active
and next run, but they never carry approval; if conversation context no longer
establishes approval for the next launch, ask the user again.

If the tracked execution session is lost, inspect or reconcile the run. Never
infer completion, silently relaunch it, or advance to the next run. If the run
is still active and the host cannot reattach a tracked wait, report that limit
and keep the existing run intact.

For RunPod, also follow the `runpod-lifecycle` recovery flow immediately:
session loss is a billing exposure, but the Pod must not be stopped until
remote exit and local output download are confirmed. If either is uncertain,
report the live billing exposure and exact recovery/stop commands. Treat the
Pod-side maximum lease as a best-effort fuse, not confirmed cleanup.

## Question budget

Questions are the second way to destroy the light-user experience (the first
is extra gates). Rules:

- **Prefer stated assumptions over questions.** When intent is missing
  (trigger word, task type, pair direction), infer it from evidence and
  state the assumption with its evidence in the plan provenance line —
  "captions start with 'myaku' in 38/40, treating it as the trigger word".
  Plan approval doubles as confirmation; a wrong guess costs one correction,
  not one interrogation.
- **Ask directly only when** inference is impossible **and** being wrong
  wastes a run. Batch such questions into one message at first contact with
  a dataset — never a drip.
- **Write every answer down** (`dataset.yaml`, `run.yaml` intent) so it is
  never asked again. Undeclared intent cannot be detected mechanically;
  once recorded, it becomes a measurable fact for every future run.

## Inputs (facts, not guesses)

Gather these before proposing parameters:

1. `uv run kura run capabilities <backend> --json` — the public authored
   vocabulary for the selected adapter. Check each field's architecture/mode
   applicability before writing `backend.config`; do not grep adapter source or
   guess a near-synonym such as `optimizer`/`scheduler`. Treat listed escape
   hatches as unverified native input, not ordinary Kura configuration.
2. `uv run kura run plan <run-id>` — the Resources section (local GPU and
   VRAM, executor, architecture, artifact filenames, memory flags) and the
   model download estimate. A surface-contract refusal here means the draft is
   not an approvable plan; correct `run.yaml` before showing it to the user.
3. The dataset facts, gathered **before** proposing anything: item count,
   resolution distribution, caption statistics (empty / duplicates / trigger
   word occurrences), pair integrity, task type. Use `kura dataset inspect`
   once it exists; until then use `kura dataset validate` plus a manual look,
   and the `dataset-prep` skill. Never propose parameters without this
   material.
4. Knowledge cards: read **only** the cards that match this run —
   `../../../knowledge/model-families/<family>.md` (matched from the
   architecture string in the plan's Resources section) and
   `knowledge/user-preferences.md`. Do not bulk-read
   the whole knowledge directory. If no exact card exists and the architecture
   is a video adapter (`wan`, `hunyuan_video`, `hunyuan_video_1_5`,
   `framepack`, `kandinsky5`), read `knowledge/video-architectures.md` as a
   placeholder and treat it as a weak starting point.
5. Prior evidence: `runs/` history for the same architecture on this
   hardware — successful configs, OOM failures, observed speed. Real outcomes
   on this machine beat any rule of thumb.
6. Backend mechanics: `musubi-tuner-backend` / `ai-toolkit-backend` skills for
   flag names, constraints, and interactions.

## Resume continuity warnings

Treat a Resume plan as continuity-risk analysis, not as a promise of bitwise
equivalence. Read the plan's `Resume` section and repeat its `continuity`,
`restored`, and `not_restored` facts in the approval summary.

Before asking for approval, explain in plain language that Resume is primarily
a recovery and extension safety net: it can avoid throwing away useful training
after a crash or an undersized run, but it may not reproduce an uninterrupted
session exactly. Never leave that caveat implicit in a capability label. Name
the selected backend, what it restores, what it does not restore exactly, and
whether matching evidence exists. Keep Weight Continue or Fork from Weight
separate; neither is a fallback that may be silently substituted for Resume.

Use revision-specific equivalence evidence only when backend revision or image
digest, architecture, optimizer/scheduler, and dataset cardinality match. The
local 100-step versus 50+50 evidence is recorded in
`../../../docs/smoke-evidence/2026-08-27-training-resume-equivalence-local.yaml`.
Disposable-Pod transfer evidence is recorded separately in
`../../../docs/smoke-evidence/2026-08-27-training-resume-runpod.yaml`; it proves
artifact portability and lifecycle recovery, not uninterrupted-run numerical
equivalence.
Do not transfer a one-item result to a shuffled or multi-item dataset.

Do not infer warning severity from the restoration contract's missing-state
list. That list is a fact about what the backend restores, not a measurement of
the resulting divergence. When matching evidence contains two uninterrupted
controls, use their difference as the observed nondeterminism baseline and
classify the user warning as follows:

- **HIGH** — Resume introduced learned-weight or optimizer divergence materially
  beyond the uninterrupted-control baseline, including any difference when the
  two controls matched exactly. State both baseline and Resume tensor counts,
  maximum absolute error, and relative L2 error. A small numeric error describes
  magnitude; it does not make the Resume exact.
- **CAUTION** — Resume learned-state divergence stayed within the same measured
  scale as the uninterrupted-control baseline, or learned weights, optimizer, and scheduler
  matched while RNG, application counters, sampler, or exact dataloader position
  differed. Name the narrow tested conditions and the baseline comparison; one
  control pair is evidence, not a statistical bound.
- **UNVERIFIED** — no evidence matches the selected revision and conditions.
  This includes evidence without an uninterrupted repeat control. Report the
  restoration contract without inventing an expected error size or severity.

Never label Resume exact from weight equality alone. Exactness requires all
declared training-state components and the data position to match. File SHA
differences alone are not semantic differences when recursive state comparison
shows equal values; report both facts separately.

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
  proposal with the expected slowdown stated. Offload/swap may also go into
  the proposal without a separate question when it is the least
  meaning-changing fit and the expected elapsed-time increase stays below
  roughly 2x. Ask first when any execution accommodation is expected to cross
  that threshold.
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
  stated reason, offload/swap below the ~2x elapsed-time threshold.
- **Ask before proposing:** any >~2x slowdown, every rung 3
  option, extrapolating an untested hypothesis into an expensive run,
  changing the RunPod GPU class (cost).
- **Never:** changing anything between the approved plan and launch; leaving
  an applied trade-off or card deviation out of `run.yaml` and the plan
  discussion. Reserve `notes.md` for human evaluation and observations after
  the run.

## Knowledge cards

Model-family cards live outside this skill, in `knowledge/model-families/`
at the repository root. They are shared with `lora-evaluation` and are not
agent-specific. Read only the card matching this run.

Layout under this skill directory:
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

1. Evidence is recorded in the run's `notes.md` first (settings, hardware,
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
