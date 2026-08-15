# anima

## Which variant to train on, which to generate with

Train on Base. Aesthetic can be used for generation. For an ordinary
confirmation render, prefer Aesthetic: it stays workable under rough settings.
source: owner (2026-08-15)

## Training

- VRAM class: Anima character LoRA completed at 768, rank 16, batch 1, bf16,
  gradient checkpointing, and disk-backed latent/text caches on an RTX 4070 Ti
  with 12 GB VRAM. Observed speed was about 1.0 second per optimizer step after
  initialization.
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3

## character

- learning rate: start from 7e-5 with a constant scheduler for a small
  character dataset when no stronger architecture-specific evidence applies.
  source: owner (2026-08-04)
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- resolution: 768
  source: owner (2026-07-02)
- rank / alpha: 16 / 8 as a verified small-character starting point.
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- checkpoint review: retain checkpoints every 100 steps during exploratory
  runs. For a dataset near six images with one repeat and batch 1, evaluate
  800–1200 first and use 1000 as the current preferred checkpoint.
  source: owner (2026-08-04)
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- overtraining reminder: in the cited six-image run, checkpoints after roughly
  1500 increased learned rendering-style leakage without a clear general
  improvement in character usefulness. Do not assume more steps are better.
  source: run 20260803-2140_vivi-anima-7e5-2000_d8d3
- scope: the 1000-step preference is evidence for a very small, visually
  homogeneous character dataset. Do not copy the absolute step count to a
  larger dataset; compare optimizer steps, repeats, effective batch, dataset
  diversity, and retained checkpoints together.

## style

- no entry yet. The character evidence above explicitly showed unwanted style
  retention and must not be treated as a style-LoRA baseline.

## Prompt and evaluation


source_url: https://huggingface.co/circlestone-labs/Anima
source_revision: main
verified_at: 2026-08-03
applies_to_model_revision: base f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b; aesthetic 594c27fea35648b87c86a9b4d5436a6024c820b5; correspondence unverified
confidence: unverified

### Confirmed upstream guidance

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

### Prompt policy by variant

#### base

- Current README recommends the positive prefix
  `masterpiece, best quality, score_7, safe, `.
- Confirm its relationship to the pinned base revision before treating it as
  revision-specific truth.
  source: upstream (Anima README main); confidence: unverified for pinned model

#### aesthetic

- `masterpiece, best quality` is documented as safe to retain; upstream
  recommends omitting `score_*` tags.
- The authored aesthetic workflow currently contains the same `score_9`,
  `score_8_up`, and negative `score_1` defaults as the base workflow. Preserve
  that external workflow, but disclose this difference and choose a policy
  deliberately in the evaluation plan.
  source: upstream (Anima README main) + workspace workflow observation

### Dataset-caption hypothesis

Aligning training-caption culture and inference prompts may matter, but the
effect of character position, count tags, safety tags, and capitalization on a
trained LoRA has not been isolated here. Treat any causal claim as inferred,
not established.
source: agent (2026-08-03); confidence: inferred
