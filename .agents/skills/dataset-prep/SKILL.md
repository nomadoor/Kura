---
name: dataset-prep
description: Dataset preparation and validation for Kura training runs. Use when creating or editing datasets/, dataset.yaml, items.jsonl, captions, trigger words, paired/control datasets, dataset roles, or dataset validation behavior.
---

# Dataset Prep

Use this skill for dataset operations.

## Rules

- Never commit dataset payloads.
- Commit only small manifests or synthetic metadata fixtures that contain no
  dataset payloads.
- Preserve `datasets` as an array in run intent and locks.
- Keep role/digest visible for paired/control datasets.
- Do not add repeats/weights unless explicitly intended.

## Caption edits

- Make caption transformations deterministic and reviewable.
- Before creating captions, inspect the target model-family's caption culture
  through the family card (`.kura/knowledge/model-families/<family>.md`, or the
  user's `knowledge/model-families/<family>.md`) or upstream primary sources.
  Record whether captions are tags, natural language, structured data, or a
  mixture. Do not infer that a documented inference-time tag order is a proven
  training requirement.
- For trigger words, prepend consistently and avoid duplicate prefixes.
- Preserve original files unless the user asks for in-place edits.

## Validation

```sh
kura dataset inspect datasets/<id>
kura dataset validate datasets/<id>
kura run compile <run-id>
```

During preparation, read the inspect output for declared-count, missing-caption,
condition-pair, and aspect-ratio mismatch facts before authoring a run. Present
those measurements to the user without treating the command as a verdict or
editing the dataset automatically.

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

First-class training uses manifest v2. Set `items_schema_version: 2` in
`dataset.yaml`; each `items.jsonl` row uses ordered typed file references and
an explicit caption value:

```json
{"id":"one","files":[{"type":"file","role":"target","path":"images/one.png"}],"caption":{"text":"trigger word, short caption"},"metadata":{}}
```

Pair target, control, reference, and audio inputs by placing their typed
references in the same row. Role meaning belongs to the selected backend. In
particular, Musubi FLUX.2 reference images are authored with `role: "control"`
because its generated JSONL consumes them as `control_path` /
`control_path_N`; do not author a separate `reference` role for that path.
Use `sha256` on an individual file reference only when the author intends to
assert that exact digest. `kura dataset validate` refuses rows outside the
closed schema.

## Migrating old AI-Toolkit dataset selectors

Historical run records stay unchanged. When using an old `run.yaml` as the
starting point for a new run:

- replace `backend.config.dataset_folder` by listing each selected media file
  in manifest-v2 `items.jsonl` with `role: "target"`;
- replace `backend.config.dataset_config.control_subdir` by adding the matching
  per-sample files with `role: "control"`, in the order the trainer must
  receive them; and
- remove both old selectors, validate the dataset, then compile again.

Kura does not infer either replacement during compile. Use `kura dataset draft`
only to create a reviewable candidate manifest; confirm its sample associations
before adopting it.
