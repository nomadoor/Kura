# `items.jsonl` lines

## The line

```json
{"id":"001","group":"close-ups","files":[{"type":"file","role":"target","path":"images/001.png"}],"caption":{"text":"..."},"metadata":{}}
```

| Key | |
|---|---|
| `id` | required, unique |
| `group` | optional; splits a dataset into blocks (see Groups) |
| `files` | required, not empty; each `{"type":"file","role":<role>,"path":<path>}`. `path` is relative to the dataset folder. An optional `sha256` (64 lowercase hex digits) must match the file |
| `caption` | required: `{"text": "..."}`, `{"file": {"type":"file","path":"001.txt"}}`, or `null` (training refuses `null`) |
| `metadata` | a mapping; no file references in it |

Files Kura accepts by extension:
- images: png, jpg, jpeg, webp, bmp, avif, gif, jxl (each trainer reads fewer: AI-Toolkit png,
  jpg, jpeg, webp)
- video: mp4, mov, mkv, webm, avi, m4v, mpeg, mpg, flv, wmv
- audio: wav, flac, mp3, m4a, aac, ogg, opus

Every media file in the folder must be in the list or excluded (`excluded_files`,
`excluded_directories` in `dataset.yaml`); validation names the ones left over.

## Kinds of data

All lines of one dataset have the same shape; put different shapes in different datasets.

**Images only**
```json
{"id":"001","files":[{"type":"file","role":"target","path":"001.png"}],"caption":{"file":{"type":"file","path":"001.txt"}},"metadata":{}}
```

**Edit pairs** (before → after). The caption is the instruction.
```json
{"id":"001","files":[{"type":"file","role":"control","path":"before/001.png"},{"type":"file","role":"target","path":"after/001.png"}],"caption":{"text":"make the jacket red"},"metadata":{}}
```
Several before images: several `control` files, in the order the model takes them. Some models
need the same number in every line.

**References** (images or videos the result should follow): `reference` files beside the
`target`. Musubi Tuner's FLUX.2 takes reference images as `control`.

**Video**: a `target` video. Frame counts and fps are set in the training run
(`backend.config.dataset_config` for AI-Toolkit, `dataset_options` for Musubi Tuner); see
`kura run capabilities <backend> --arch <value>`.

**Video with audio**: either the target video's own sound track (AI-Toolkit LTX), or a separate
`audio` file in the line (Musubi Tuner MiniMax H3). Reference variants exist for some models
(`reference-muted`, `reference-audio`); compile names what a model takes.

## Groups

`group` splits a dataset into blocks, for example to repeat close-ups more often. A training
run then either declares a block for each group (with its repeats) or sets
`flatten_groups: true` to train them as one.

## Moving an older dataset over

Older runs selected images with `backend.config.dataset_folder` or `control_subdir`. Kura now
refuses both. List the files in `items.jsonl` instead: each selected image as `target`, each
matching control image as `control` in the same line, in the order the trainer takes them.
`kura dataset draft` does not read `control_subdir`; check its rows and add the controls.
