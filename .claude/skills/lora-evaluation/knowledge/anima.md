# Anima prompt and evaluation knowledge

source_url: https://huggingface.co/circlestone-labs/Anima
source_revision: main
verified_at: 2026-08-03
applies_to_model_revision: base f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b; aesthetic 594c27fea35648b87c86a9b4d5436a6024c820b5; correspondence unverified
confidence: unverified

Training parameters live in the matching training run and backend knowledge;
this card owns prompt and evaluation context only.

## Confirmed upstream guidance

- Anima was trained with Danbooru-style tags, natural-language captions, and
  mixtures of both. The documented tag order is quality/meta/year/safety,
  count, character, series, artist, then general tags.
- Human quality tags and `score_9` through `score_1` are supported. Tags may be
  dropped during training; the documented order and examples are guidance,
  not proof that every listed tag is mandatory.
- For pure natural language, upstream recommends at least two descriptive
  sentences and standard capitalization for character and series names.
- Current upstream guidance recommends avoiding `score_*` in both positive and
  negative prompts for Anima-Aesthetic. This is a recommendation, not a ban.
  source: upstream (Anima README main, verified 2026-08-03)

## Variants

### base

- Current README recommends the positive prefix
  `masterpiece, best quality, score_7, safe, `.
- Confirm its relationship to the pinned base revision before treating it as
  revision-specific truth.
  source: upstream (Anima README main); confidence: unverified for pinned model

### aesthetic

- `masterpiece, best quality` is documented as safe to retain; upstream
  recommends omitting `score_*` tags.
- The authored aesthetic workflow currently contains the same `score_9`,
  `score_8_up`, and negative `score_1` defaults as the base workflow. Preserve
  that external workflow, but disclose this difference and choose a policy
  deliberately in the evaluation plan.
  source: upstream (Anima README main) + workspace workflow observation

## Dataset-caption hypothesis

Aligning training-caption culture and inference prompts may matter, but the
effect of character position, count tags, safety tags, and capitalization on a
trained LoRA has not been isolated here. Treat any causal claim as inferred,
not established.
source: agent (2026-08-03); confidence: inferred
