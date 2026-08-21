---
name: lora-evaluation
description: Plan, review, and judge reproducible LoRA and trained-adapter evaluations across Kura-supported image and video model families. Use before authoring evaluation prompts, comparing checkpoints or strengths, requesting step reviews, contact sheets or XY plots, testing identity/style/outfit/pose/motion transfer, or treating generated samples as evidence.
---

# LoRA evaluation

Use this skill before any render intended to judge a trained LoRA. Trainer
backends provide training facts; model-family knowledge owns prompt semantics;
this skill designs the evaluation; the available render skill executes it.

## Responsibility boundary

- Trainer backends expose training facts; they do not own prompt semantics.
- Model-family knowledge owns sourced prompt and inference guidance.
- This skill judges the evaluation design; the user approves it.
- `run.yaml` and its frozen manifest record the approved intent and inputs.

Do not add a core prompt parser or trigger-word gate. Effective prompts may be
weighted, aliased, split across workflow nodes, or transformed by the consumer;
their meaning requires family knowledge and the complete workflow context.

## Required inputs

Before writing prompts, inspect only the material relevant to the run:

1. training `run.yaml`, `resolved/`, checkpoint list, and realizations;
2. `dataset.yaml`, `items.jsonl`, captions, declared task and trigger terms;
3. trainer backend and training settings;
4. target workflow, its default positive/negative prompts, model references,
   sidecar, pinned model revision, and LoRA insertion/strength;
5. `knowledge/model-families/<family>.md` at the repository root and its
   cited upstream primary sources;
6. prior evaluated runs and their `notes.md`.

If no family card exists, inspect upstream primary documentation and record
`evaluation.knowledge.card: none`, the source URLs, and
`revision_match: unverified`. Missing knowledge is not a render gate and must
not be hidden.

## Evaluation design

Choose an evaluation category and vary one axis where practical. Canonical
image categories are `reconstruction`, `identity_retention`,
`outfit_transfer`, `pose_composition_transfer`, `background_transfer`,
`style_transfer`, `prompt_adherence`, `checkpoint_comparison`,
`strength_sweep`, and `variant_comparison`. These are vocabulary, not a closed
core enum; use a clearer new category when the task requires one.

Categories name the question being tested, not a verdict. Reconstruction is
not evidence of generalization. Keep confounded or superseded evaluations and
state their limits in `notes.md` rather than replacing their records.

Represent an evaluation matrix as one explicit ordered case JSONL referenced by
`inputs.cases`. List every intended combination; Kura does not provide a
Cartesian-product DSL. For `checkpoint_comparison`, repeat the same prompt set,
seeds, workflow settings, and strengths in each checkpoint's rows wherever the
question calls for a clean checkpoint axis. A separate run is appropriate when
it represents a separately intended evaluation, not an ad-hoc way around the
case contract.

Video categories additionally include `motion_retention`,
`temporal_consistency`, `camera_motion_transfer`, `action_transfer`,
`frame_adherence`, and `identity_drift`. Kura currently has no video render
execution path. Define and present the plan, but do not invoke ComfyUI or
another generator outside Kura. Tell the user that execution needs a separately
designed Kura path or their explicit decision.

Do not infer prompt correctness with substring gates. Triggers may be weighted,
aliased, structured, distributed across workflow nodes, or transformed by the
consumer. Read the complete effective prompt context and judge it using the
family card, training captions, workflow defaults, and evaluation goal.

## User review before execution

When presenting the plan, read
[the review template](references/review-template.md). Match the user's language
and make the recommended plan easy to approve or correct in one reply.

Present together:

- purpose and category;
- what is fixed and what varies;
- the explicit case axes, authored row count, case order, and expected image count;
- checkpoints, LoRA strengths, workflows, variants, and seeds;
- complete positive and negative prompts;
- model-family policy sources and revision confidence;
- dataset overlap relevant to the category;
- what the test can and cannot establish.

Obtain approval before execution. For image execution, then use
`comfyui-render-workflow`; its endpoint and external-mutation rules still
apply.

## Record the plan

Add an `evaluation:` mapping to the render `run.yaml`. It is frozen by the
existing manifest mechanism without core interpretation. Use the schema in
`references/knowledge-schema.md`. Keep category as an extensible string and
record knowledge absence explicitly.

After review, append observations and limitations to `notes.md`. Never erase a
failed or confounded evaluation; mark what evidence it can and cannot support.
Contact sheets and XY plots are presentation-only derivatives of existing
rendered images, not another Kura generation path. Kura owns raw generation and
complete per-result metadata; the agent owns presentation layout. Preserve the
source images and record their paths in `notes.md`.

## Skill order

`dataset-prep -> training-parameter-planning -> backend skill -> training -> lora-evaluation -> model-family knowledge -> render execution -> notes`
