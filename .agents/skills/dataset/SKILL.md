---
name: dataset
description: Make, prepare, and check a Kura training dataset. Use to collect and select images or videos, add crops, duplicates, or augmentations, write or fix captions, build edit pairs (before and after images) or data with references or audio, review a dataset, or when dataset validation reports a problem.
---

# Dataset

This skill has two jobs.

| | Make and review the content | Make it a Kura dataset and check it |
|---|---|---|
| What | select, crop, duplicate, augment, caption, review | place the files, write the list (`items.jsonl`), validate |
| Looks at image content | yes (needs the user's permission) | no; Kura measures shape and counts |
| Whose feature | outside Kura; you, with tools already installed | Kura's |

Whichever the user asks for, finish with "Make it a Kura dataset and check it". Do "Make and
review the content" only when the user asks. When you come here from training to fix the
dataset's shape, do part 2 only.

## Who decides

| | Decides |
|---|---|
| Kura | whether the shape is valid; counts, resolutions, captions present, duplicates, missing pairs; the shapes a model accepts |
| You | proposing and doing the work; reporting what Kura measured |
| The user | what the LoRA should learn, which data to use, how to edit it, how captions are written |

## 1. Make and review the content

### Ask first
- What the LoRA should learn (a person, a style, clothes, motion, a way of editing...).
- Whether to use a trigger word, and which.
- What you may do (select, crop, duplicate, augment, caption). Do only what you were allowed.

### Permission to look
- This work looks at image content: follow "The user's images" in `AGENTS.md`
  (`agents.view_images: true`).
- Without it, ask the user to do that part, or to allow it.

### Keep the originals
- Never move or overwrite the original images or videos.
- Put what you make (crops, augmented images) in the dataset folder as new files.
- Write what you did (from which original, how) in `description` in `dataset.yaml`, or in a
  note in the dataset folder.
- Use tools already installed. Ask before installing one.

### Captions
- **Write them the way the model's official guide does.** Follow the **Prompts** section of the
  family card (`.kura/knowledge/model-families/<family>.md`): tags, sentences, or structured.
  Render prompts use the same way. Without it, read the model's official documents and name
  your source.
- **What to write and what to leave out** follows from what the LoRA should learn; the user
  decides. A common way: leave out what the LoRA should learn (the person's face, the style) and
  let the trigger word carry it; write what should stay changeable (clothes, hair, pose, place).
- With a trigger word, put it once, in the same place, in every caption.
- In an edit dataset a caption is the instruction (what to change), not a description of the
  image. Follow the model's official way.
- Write them all by one rule. Show the user a few first, and continue after their OK.

### Review
- Look at: images (blur, duplicates, ones unlike the rest), captions (missing, off the agreed
  rule, the trigger word), pairs (does the after image match the before image).
- List what you found for the user. Remove or fix only after they decide.

## 2. Make it a Kura dataset and check it

### What a dataset is
A dataset is the folder `datasets/<id>/` with two files:

- `dataset.yaml`: its name and settings
  ```yaml
  id: <id>                 # the same as the folder name
  items_schema_version: 2
  trigger_word: <word>     # only when a trigger word is used
  ```
- `items.jsonl`: one line per training item: its files, each with a role, and its caption. An
  edit pair (before image as `control`, after image as `target`):
  ```json
  {"id":"001","files":[{"type":"file","role":"control","path":"before/001.png"},{"type":"file","role":"target","path":"after/001.png"}],"caption":{"text":"make the jacket red"},"metadata":{}}
  ```
  An image-only item has one `target` file. A caption can point at a file instead:
  `"caption":{"file":{"type":"file","path":"001.txt"}}`.

| Role | What it is |
|---|---|
| `target` | the image or video trained on (the result) |
| `control` | an edit's before image, a condition image |
| `reference` | a reference image or video |
| `audio` and others | audio |

How to write each kind (image only, edit pairs, references, video, video with audio) is in
references/rows.md.

### The shapes a model accepts
- Models need different roles and counts (one control, one to three, any number...).
- Check with `kura run capabilities <backend> --arch <value>`; if it does not say, see the card.
  Compile of the training run refuses a shape that does not fit, naming the roles and counts it
  needs.

### Make it
- When the user's files are elsewhere, copy them into `datasets/<id>/`; never move the
  originals.
- **Images only**: put the images in `datasets/<id>/` (or `datasets/<id>/images/`) with captions
  as same-name `.txt` files, write `dataset.yaml`, and run `kura dataset draft <id> --write`. It
  writes the list.
- **Anything else** (edit pairs, references, video, audio): `draft` cannot pair them. Write
  `items.jsonl` by references/rows.md.
- When `items.jsonl` already exists, `draft` does not overwrite it; it writes
  `items.v2.candidate.jsonl`. Decide with the user which to keep.

### Check it
- `kura dataset validate <id>`: whether the shape is valid. If not, follow the fix it names.
- `kura dataset inspect <id>`: counts, resolutions, missing or duplicate captions, trigger-word
  coverage, missing pairs, aspect-ratio mismatches. It does not look at image content.
- Tell the user the results as facts, not a verdict. Do not fix things on your own.

## References

- references/rows.md: `items.jsonl` lines for each kind of data, groups, and moving an older
  dataset over
