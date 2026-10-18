---
name: training-parameter-planning
description: Plan Kura training quality, memory, runtime, and cost trade-offs. Use when drafting or reviewing training run parameters, interpreting plan resources, reacting to OOM, or choosing execution accommodations and GPU capacity.
---

# Training parameter planning

Choose a useful training recipe, not merely one that starts. Preserve the
user's intent and make every quality, speed, memory, and cost trade-off visible
in `run.yaml` and the compiled plan. Training configuration and compute
selection are one plan: dataset size, resolution, batch, accumulation,
precision, rank, optimizer, and low-memory options all affect quality, runtime,
memory, and cost.

## Approval model

Draft and revise freely. The mandatory launch approval is the final
`kura run plan <run-id>` review required by `AGENTS.md`.

Ask earlier only when a missing fact cannot be inferred and a wrong assumption
would waste a run, or when the proposal changes cost, quality, or expected
runtime materially. Batch necessary questions instead of asking one setting at
a time. Record answers in dataset or run intent so they are not asked again.

## Gather facts first

Before proposing parameters, inspect:

1. `kura run capabilities <backend> --arch <selector value> --json` for the
   accepted authored vocabulary, the fields that apply to that architecture,
   and its model facts (AI-Toolkit's baseline entry, the model roles
   sd-scripts and Musubi read); without `--arch` it prints the whole surface;
2. `kura dataset inspect <dataset>` and dataset validation for item
   count, resolutions, captions, roles, and pair integrity;
3. `kura run plan <run-id>` for model artifacts, executor, detected or
   selected GPU resources, cache state, and download estimates. The plan
   reads a run, so first create it with `kura run new --experiment <name>
   --slug <words> --backend <backend>` and fill a draft
   `backend.config` that `kura run capabilities <backend>` accepts (for
   sd-scripts, a `dataset_config` whose datasets have their `subsets`, each naming its manifest dataset with `dataset_id`);
   revise the draft as the proposal takes shape;
4. the matching family card, when present: `.kura/knowledge/model-families/<family>.md`
   (shipped) and `knowledge/model-families/<family>.md` (the user's, which wins);
   a missing card is not a reason to refuse the model;
5. `knowledge/user-preferences.md` and prior evaluated runs under comparable
   conditions;
6. `kura run capabilities <backend>` and the compiled plan for backend mechanics.

Do not guess field names from adapter source. A capability refusal is the
public contract speaking.

## Name the model

Where the weights come from depends on the backend:

- **ai-toolkit** trains from `model.base`, which compile requires: a Hugging
  Face repository id (`owner/name`), which the trainer downloads itself, or an
  absolute path as the container sees it (`/workspace/...` for a file in the
  workspace, or under a `docker.mounts` target in `workspace.yaml`) for a local
  Docker run. `backend.config.model_arch` says what the weights are.
- **sd-scripts** and **musubi-tuner** take their weights from
  `backend.config.model_paths` and `backend.config.model_downloads`; musubi-tuner
  can instead name a `model_bundle`, which downloads its own listed files and is
  never combined with `model_downloads`. `model.base` is optional and only a
  label. For musubi-tuner `architecture: flux2`, `model_version` alone picks the
  variant; a bundle or a Klein name in `model.base` for another variant is
  refused. `backend.config.architecture` says what the weights are. Kura keeps
  no list of default weights: a card's `Model files` section, when it has one,
  gives the `model_downloads` entries a recorded run trained from, and a
  refusal for a missing model role points there.

The family cards are named after the family, not the selector. The selector
values each shipped card covers:

| Card | ai-toolkit `model_arch` | sd-scripts `architecture` | musubi-tuner `architecture` |
|---|---|---|---|
| `sd15` | `sd1` | `sd15` | — |
| `sdxl` | `sdxl` | `sdxl` | — |
| `anima` | `anima` | `anima` | — |
| `flux2` | `flux2`, `flux2_klein_4b`, `flux2_klein_9b` | — | `flux2` |
| `flux_kontext` | `flux_kontext` | — | `flux_kontext` |
| `krea2` | `krea2` | — | `krea2` |
| `iris3b` | `iris3b` | — | — |
| `moss_image` | `moss_image:base` | — | — |
| `zimage` | `zimage` | — | `zimage` |

Every other selector value `kura run capabilities <backend>` lists has no
shipped card; a user card in `knowledge/model-families/` may cover it. For a
family without a card, read the pinned trainer's own source and docs for that
architecture: `kura run capabilities <backend>` prints the pinned trainer image
and the source it was built from (for AI-Toolkit, also the commit of the
pinned upstream UI baseline; `--arch <model_arch>` prints that value's
baseline entry).
Say which file you read; it is evidence about the trainer, not a recipe
someone has verified.

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

Saved weight precision is the trainer's unless the run sets it, and Kura does
not choose it. Musubi Tuner and sd-scripts save LoRA weights in fp32, the
precision the network trains in, unless `save_precision` is set (for
sd-scripts, `--full_fp16`/`--full_bf16` also trains and saves the network in
that precision).
Musubi Tuner takes `backend.config.save_precision`; sd-scripts has no declared
field for it, so only its escape hatch sets it (`backend.config.extra_args:
["--save_precision", "bf16"]`). AI-Toolkit saves fp16 by default (its
`SaveConfig` dtype `float16`), and `backend.config.native_config.save.dtype`
changes it. At the same rank an fp32 file is about twice the size of an fp16
or bf16 one.

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
   expected slowdown. `recipe.steps` counts optimizer updates on every
   backend, so with `gradient_accumulation_steps: N` each step runs N
   micro-batches and takes about N times as long.
3. **Recipe-changing:** resolution, effective batch, rank, learning rate,
   model size, precision with quality impact, or dataset reduction. Present
   concrete alternatives and recommend one before editing the approved plan.

Changing GPU class or cost, applying a recipe-changing adjustment, or accepting
an expected slowdown beyond roughly twofold requires a new plan decision.
After an OOM, diagnose the actual log and move one rung; never silently retry
with changed settings.

### AI-Toolkit on a small local GPU

AI-Toolkit's baseline keeps the text encoder in VRAM for the whole run
(`train.cache_text_embeddings: false`). On a local GPU with about 12 GB this
can spill into shared system memory: the run does not fail, it runs tens of
times slower (Z-Image and MOSS-Image on 12 GB ran 35 to 55 times slower). For a local AI-Toolkit run on such a GPU, set
`native_config.train.cache_text_embeddings: true` unless a limit below
applies; the family cards give measured figures. In the pinned AI-Toolkit
(dcebc744) it encodes each item's caption once before training, writes the
embeddings to a `_t_e_cache` folder beside the images, and removes the text
encoder from the GPU. Its limits:

- training the text encoder is refused;
- each item's caption is encoded once, so `shuffle_tokens` and
  `random_triggers` take effect once rather than every step, and
  `token_dropout_rate` is skipped; `caption_dropout_rate` still works (the
  dropped caption is cached too and chosen per step);
- it applies to every dataset in the run.

With the cache on, `unload_text_encoder: true` adds nothing. Never set
`unload_text_encoder: true` without the cache: the trainer then trains on a
blank prompt (or the trigger word only) instead of the captions.

Recognise a spill while a run executes: seconds per step tens of times above
the family card's figure for comparable settings, or GPU memory pinned at the
GPU's total. Stop the run (`kura run stop`), tell the user, and plan it again
with the cache rather than waiting.

## Last look

Immediately before presenting the approval plan, read the shipped
`.kura/knowledge/regrets.md` and the user's `knowledge/regrets.md`. Return only relevant `trigger -> reminder` notes. This
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

The plan's **Training state** block says what a run leaves to resume from:

- `enabled`: `recovery.training_state.enabled` in `run.yaml` (default yes).
- `keep`: how many of the newest saved states are kept, 1 or 2
  (`recovery.training_state.keep_generations`, default 2).
- `cadence`: steps between state saves, from `backend.config.save_every_n_steps`
  (the only place it is set; escape hatches refuse it). Every trainer counts a
  Resume's steps from the step it resumes at, so a Resume keeps the cadence and
  saves on its multiples; it always saves state at its end too. When none is
  set, sd-scripts saves state once, at the end of the steps the run trains (the
  recipe's steps, or the steps a Resume adds), Musubi Tuner is given the
  recipe's steps, and the plan shows that number, while AI-Toolkit saves at its
  own default cadence, which the plan shows as `trainer default`
  (`trainer_default_cadence: true` and `cadence_steps: null` in `--json`).
  Otherwise `cadence_steps` is an integer, or null for a draft with no steps
  yet.
- `capability`: how far the backend can restore this architecture and mode:
  `partial_resume`, `best_effort_resume`, or `unsupported`.
- `saved`: whether Kura manages the state: enabled, Kura builds the command
  (no `backend.config.command`), and the capability is not `unsupported`.
  Only then does the trainer save state, and a completed run that leaves none
  ends `recovery_required`.

The **Trainer write locations** block lists the paths outside the run's own
directories that the trainer writes to (a model or backend cache); Kura
creates them and checks they are writable before the trainer starts. They
come from the compiled command, so an uncompiled run shows `none declared`.

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
run's `notes.md` first. Promote only generalizable findings to the user's family card in
`knowledge/model-families/`, citing the run. Record an actual regret confirmed by a run or
named by the user as one concise entry in the user's `knowledge/regrets.md`,
citing `source: run <id>` or `source: user (<date>)` respectively. Do not seed
hypothetical regrets.
