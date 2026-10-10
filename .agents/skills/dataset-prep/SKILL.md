---
name: dataset-prep
description: Dataset preparation and validation for Kura training runs. Use when creating or editing datasets/, dataset.yaml, items.jsonl, captions, trigger words, paired/control datasets, or dataset roles, or when dataset validation reports a problem.
---

# Dataset Prep

Use this skill for dataset operations.

Built-in training requires the dataset manifest contract: the dataset manifest
inventories inputs, `run.yaml` selects them, and, for a run using the backend's
built-in command, the `resolved/` input lock records the projection actually
handed to the trainer. A run that sets `backend.config.command` gets an input
lock marked `unverified-native-source` instead, because Kura cannot see what
that command reads.

## Rules

- If the workspace is under version control, never commit dataset payloads;
  only `dataset.yaml` and `items.jsonl` may be committed.
- Preserve `datasets` as an array in run intent and locks.
- Keep role/digest visible for paired/control datasets.
- Do not add repeats/weights unless explicitly intended.

## Starting from a folder of images

1. Put the images in `datasets/<id>/` or in `datasets/<id>/images/` (one of
   the two, not both), each with a caption file of the same name ending in
   `.txt` or `.caption`.
2. Write `datasets/<id>/dataset.yaml`:

   ```yaml
   id: <id>                 # must match the directory name
   items_schema_version: 2
   trigger_word: <word>     # optional; inspect counts it, and `kura run plan`
                            # warns about captions the trainer receives without it
   ```

3. `kura dataset draft <id> --write` writes `items.jsonl`. Check that each
   row pairs the right image and caption (each row points at its caption file
   as `caption.file`). It never replaces a file and never half adopts a draft:
   when `items.jsonl` already exists, the draft reports `review required`, or
   `dataset.yaml` lacks `items_schema_version: 2`, it writes
   `items.v2.candidate.jsonl` instead (and `dataset.v2.candidate.yaml` in the
   last case) and prints what to do with each. Resolve what it reports, then
   rename or move the candidates yourself. When it drafts no rows it writes
   nothing.
4. `kura dataset validate <id>`, then `kura dataset inspect <id>`. Every
   dataset command takes either the ID or the dataset's path.

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

- Open dataset images only when `workspace.yaml` sets `agents.view_images:
  true` (the `AGENTS.md` "User images" rule); otherwise every point below works
  from file, dimension, caption, and manifest facts. With permission, stop at the
  first image your own service's policy does not let you handle, and say so.
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

Kura does not infer either replacement during compile. `kura dataset draft`
proposes rows from the files it finds and does not read `control_subdir`;
confirm its sample associations and add any missing control references before
validating. When `items.jsonl` already exists, `--write` writes
`items.v2.candidate.jsonl`: if the dataset is already v2, compare the two and
edit `items.jsonl`; otherwise move the candidate over it only after that
review.
