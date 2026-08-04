# LoRA evaluation contract

Status: accepted for Phase 1

## Decision

Kura separates LoRA evaluation into four responsibilities:

- trainer backends expose training facts and do not own prompt semantics;
- model-family knowledge files record sourced prompt and inference guidance;
- `lora-evaluation` judges the evaluation design and presents it to the user;
- existing run files freeze the approved intent and execution artifacts.

This applies across ai-toolkit, musubi-tuner, and sd-scripts, and across image
and video model families. Family knowledge is independent of the trainer
because one family may be trained by several backends.

## Why core does not judge prompts

The final prompt may be weighted, aliased, structured, split across workflow
nodes, or transformed by the consumer. A substring gate over a promptset cannot
reliably determine whether a concept was exercised. Phase 1 therefore adds no
prompt parser, generator, trigger hard gate, or model-specific rule to
`src/kura/`.

The existing render compiler deep-copies `run.yaml` into
`resolved/manifest.lock.yaml`. An optional `evaluation:` block therefore
records provenance without a new state store or core schema. Mechanical checks
validate its declared shape and references only; the skill judges meaning and
the user approves the plan.

## Evidence discipline

Evaluation categories describe the intended question, not a rigid enum. Change
one axis where practical and state what remains fixed. Reconstruction samples
are not generalization evidence. Confounded or superseded evaluations remain
append-only; annotate their limits in `notes.md`.

Knowledge absence does not block execution, but must be declared with the
primary sources consulted and `unverified` confidence. Upstream recommendations
must not be rewritten as prohibitions, and applicability to pinned model
revisions must be explicit.

## Video boundary

Kura can train several video families but currently has no video render result
path. The evaluation skill may design video categories, but agents must not
bypass Kura by directly invoking an external generator. A video execution path
requires a separate design and user decision.
