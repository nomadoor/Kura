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

Minimal `items.jsonl` rows need `id` and `path`; include `caption` and `hash`
when available:

```json
{"id":"one","path":"images/one.png","caption":"trigger word, short caption","hash":"sha256:..."}
```
