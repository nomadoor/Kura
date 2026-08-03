# Evaluation knowledge and run schema

## Family cards

Store prompt/inference knowledge in `knowledge/<family>.md`, separate from
training-parameter cards. Use the backend architecture identifier when it is a
stable family name. Put materially different prompt cultures under variant
headings; do not duplicate a family merely because several trainers support it.

Every family card must contain these machine-checkable metadata lines:

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
    card: .claude/skills/lora-evaluation/knowledge/anima.md
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
