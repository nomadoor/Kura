---
name: dataset-prep
description: Dataset preparation and validation for Kura training runs. Use when creating or editing datasets/, dataset.yaml, items.jsonl, captions, trigger words, paired/control datasets, dataset roles, or dataset validation behavior.
---

# Dataset Prep

Use this skill for dataset operations.

## Rules

- Dataset payloads are normally not committed.
- Commit only small manifests/examples/fixtures.
- Preserve `datasets` as an array in run intent and locks.
- Keep role/digest visible for paired/control datasets.
- Do not add repeats/weights unless explicitly intended.

## Caption edits

- Make caption transformations deterministic and reviewable.
- For trigger words, prepend consistently and avoid duplicate prefixes.
- Preserve original files unless the user asks for in-place edits.

## Validation

```sh
uv run kura dataset validate datasets/<id>
uv run kura run compile <run-id>
```

## Visual review

- Choose the amount of visual review from the dataset's size, content, and the
  decision being made. Kura's measured facts and structural validation should
  guide that choice; visual inspection is an agent aid, not a prerequisite for
  using the CLI.
- For a large routine dataset, prefer a useful sample selected from measured
  outliers (resolution, aspect ratio, missing or unusual captions, duplicate
  candidates) plus ordinary examples. Review more when the task genuinely
  benefits from it, and state whether the review was sampled or exhaustive.
- If images are sensitive, unsuitable for visual processing, or unavailable to
  the agent, continue with file, dimension, caption, and manifest facts. Explain
  the resulting limit; do not make visual inspection a hidden gate.
- Never copy dataset pixels into repo documentation, run metadata, or fixtures.

Minimal `items.jsonl` rows need `id` and `path`; include `caption` and `hash`
when available:

```json
{"id":"one","path":"images/one.png","caption":"trigger word, short caption","hash":"sha256:..."}
```
