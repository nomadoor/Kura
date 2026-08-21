# Evaluation knowledge and run schema

## Family cards

Store prompt/inference knowledge in `knowledge/model-families/<family>.md` at
the repository root, in the same card as that family's training knowledge. Use the backend architecture identifier when it is a
stable family name. Put materially different prompt cultures under variant
headings; do not duplicate a family merely because several trainers support it.

A card that carries upstream-sourced prompt guidance must contain these
machine-checkable metadata lines, because that guidance goes stale when the
upstream model card changes. A card whose facts are cited inline with `source:`
lines does not carry them; do not invent a URL or a date to satisfy the shape.

```text
source_url: <upstream primary URL>
source_revision: <commit, tag, or main>
verified_at: YYYY-MM-DD
applies_to_model_revision: <revision, mapping description, or unverified>
confidence: confirmed | inferred | unverified
```

Mark individual conclusions with `source: owner`, `source: run`,
`source: upstream`, or `source: agent`. Distinguish upstream statements from
agent inference. A stale or mismatched revision prompts re-verification; it is
not automatically a gate.

## Render intent

Recommended minimal block:

```yaml
evaluation:
  category: outfit_transfer
  fixed: [checkpoint, seed, workflow, lora_strength, prompt_prefix]
  varied: [outfit]
  model_family: anima
  model_variant: aesthetic
  knowledge:
    card: knowledge/model-families/anima.md
    card_verified_at: '2026-08-03'
    source_url: https://huggingface.co/circlestone-labs/Anima
    source_revision: main
    applies_to_model_revision: 594c27fea35648b87c86a9b4d5436a6024c820b5
    revision_match: unverified
  prompt_policy:
    prefix_origin: knowledge_card
    transformations:
    - Changed only the outfit description.
  limits: This tests outfit transfer only.
```

For a checkpoint comparison, the run references an explicit case queue:

```yaml
inputs:
  train_run: example-train-run
  cases: {path: examples/lora-evaluation/cases-checkpoint-comparison.jsonl, digest: null}
evaluation:
  category: checkpoint_comparison
  fixed: [seed, workflow, lora_strength, prompt_policy]
  varied: [checkpoint]
  model_family: anima
  model_variant: aesthetic
  knowledge:
    card: knowledge/model-families/anima.md
    card_verified_at: '2026-08-03'
    source_url: https://huggingface.co/circlestone-labs/Anima
    source_revision: main
    applies_to_model_revision: 594c27fea35648b87c86a9b4d5436a6024c820b5
    revision_match: unverified
  prompt_policy:
    prefix_origin: knowledge_card
    transformations: []
  limits: This compares saved training steps; it does not isolate every cause of quality differences.
```

The referenced JSONL explicitly lists every intended combination:

```json
{"id":"step-0200-reconstruction","values":{"prompt":"...","negative_prompt":"","seed":42,"strength":1.0},"checkpoint":{"id":"step-0200","path":"runs/example-train-run/outputs/example-step00000200.safetensors","hash":null},"meta":{"checkpoint_step":200,"prompt_axis":"reconstruction"}}
{"id":"step-0400-reconstruction","values":{"prompt":"...","negative_prompt":"","seed":42,"strength":1.0},"checkpoint":{"id":"step-0400","path":"runs/example-train-run/outputs/example-step00000400.safetensors","hash":null},"meta":{"checkpoint_step":400,"prompt_axis":"reconstruction"}}
```

There is no Cartesian DSL: the agent repeats the complete fixed values in every
row and authors every varied combination. The `evaluation` block describes the
question; the case queue is the executable matrix. Compile freezes it as
`resolved/cases.jsonl`, and generated-image records retain each row's complete
logical values, actual applied values, checkpoint provenance, and metadata.

When no card exists, use:

```yaml
knowledge:
  card: none
  basis:
  - https://upstream.example/model-card
  confidence: unverified
```

The mechanical check validates structure and references only. It does not
judge prompts, restrict category vocabulary, or affect runs without an
`evaluation:` block.
