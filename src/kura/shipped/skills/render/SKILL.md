---
name: render
description: Render with ComfyUI through Kura. Use to try a trained LoRA (compare its steps across prompts, compare strengths), make images with a workflow, assemble a comparison sheet, or when a render does not work.
---

# Render

Run a ComfyUI workflow from Kura to make a set of images under different conditions. The
images and each image's conditions stay in the run.

## Flow

```text
ask what the user wants to see ─→ choose a workflow ─→ decide the prompts
   ↓
kura render new              create the render run
   ↓ write run.yaml and the cases (one image's conditions per line)
kura render compile          freeze it
kura render launch --dry-run check it, then show the user what it renders
   ├─ change it ─→ make a new run with kura render new (back to the top)
   └─ yes
kura render launch --yes     generate in ComfyUI; follow it to the end
   ↓
(if asked) assemble the comparison sheet ─→ give its location
```

## Who decides

| | Decides |
|---|---|
| Kura | whether the workflow takes a value, whether ComfyUI sees the models, how the LoRA is placed, generation and its record |
| You | the proposed workflow, prompts, and cases; following the render; the comparison sheet; the report |
| The user | what to see, which workflow and ComfyUI, whether the images are good |

## Steps

### 1. Ask what the user wants to see
- A training approval does not approve a render. A render gets its own yes.
- The user decides what to see. Common ones:
  - **step comparison**: each step saved during training × a few prompts. With a sample
    workflow, a line without the LoRA shows the base model beside them in the same run; ask
    whether to add it.
  - **strength comparison**: the LoRA at several strengths
  - **plain generation**: a few images from one checkpoint and prompt
- The checkpoints are listed in the output of `kura run status <training run>`. Ask which to
  use and how many images to make.
- When the user wants a comparison, plan the comparison sheet (one image laying them out) from
  the start.

### 2. Choose a workflow
- Look in `workflows/samples/<family>/` first: workflows their author has run. Use the
  `_api.json` (the `.json` beside it is for people to open in ComfyUI).
- A sample may use a different variant of the model than the one trained (trained on Base,
  sample on SFT, say). When it does, say so and let the user choose.
- For a family without a sample, offer two ways and let the user choose:
  1. the user gives a workflow they trust
  2. you build one, and first render one image without the LoRA to check the workflow itself:
     that it made an image (`kura run status`); give the user its path. Judging it needs
     `agents.view_images: true`.

### 3. Decide the prompts
- Follow the **Prompts** section of the family card (`.kura/knowledge/model-families/<family>.md`,
  the user's in `knowledge/model-families/`): how the model reads text, the same way the
  training captions were written.
- Without a Prompts section, follow the model's official documents and name your source.
- If the LoRA has a trigger word, put it in (`trigger_word` in the dataset's `dataset.yaml`).
- Show the user the full prompts, negative prompt, and seeds.

### 4. Write the run and its cases
A **case** is one image's conditions: which checkpoint, which prompt, which seed. Write one
line per image you want.

For example, a step comparison of 3 steps × 2 prompts is 6 lines:

```json
{"id":"s1000-kitchen","checkpoint":{"id":"s1000","path":"runs/<training run>/outputs/<name>_000001000.safetensors"},"values":{"prompt":"<kitchen prompt>","seed":1}}
{"id":"s1000-pier","checkpoint":{"id":"s1000","path":"..._000001000.safetensors"},"values":{"prompt":"<pier prompt>","seed":1}}
{"id":"s2000-kitchen", ...}
(then step 2000 pier, and the two for step 3000)
```

Create the run with `kura render new --slug <words>` and fill in `run.yaml`:

| Field | What goes there |
|---|---|
| `intent` | what the user wants to see, in one sentence |
| `inputs.train_run` | the training run (its status then lists this render) |
| `inputs.workflow.path` | the `_api.json` to use, relative to the workspace (`workflows/...`) |
| `inputs.cases.path` | the cases JSONL, relative to the workspace: `cases/<name>.jsonl` at the top of the workspace, not in the run's folder |
| `workflow_patches` | which workflow node each case value (prompt, seed...) goes into |
| `generator.endpoint` | the ComfyUI to use, copied from `workspace.yaml` (see "ComfyUI") |
| `executor.name` | `local` (ComfyUI on this machine); `runpod` for RunPod (references/runpod.md) |

- List every combination; Kura does not multiply them out. `workflow_patches` says which node
  takes each key in `values`. How to find the nodes, and the details, are in
  references/cases.md.
- A sample workflow already says where the LoRA goes; writing the checkpoint in the case is
  enough. Any other workflow needs that place named (references/cases.md).
- The LoRA strength is 0.8 unless set. `render.lora_strength` sets it for the run; to compare
  strengths, make the strength input a case value (references/cases.md).
- `kura render compile <run-id>`. If it refuses, follow the fix it names.

### 5. Show it and get a yes
- Check it with `kura render launch <run-id> --dry-run`. It prints every case as JSON for you
  to check, not to show.
- Show it by the `writing` skill, from your cases: what is compared, the checkpoints, the full
  prompts, the seeds, how many images, how the sheet is laid out.
- **On a yes**, go to step 6.
- **When the user asks for changes**, a compiled run cannot change: make a new one with
  `kura render new` and continue from step 4. Remove the unused one with
  `kura run discard <run-id> --yes`.

### 6. Generate and follow it
- Run `kura render launch <run-id> --yes` (the user's yes reaches Kura this way) through your
  host's way of tracking a long command, as for training: a render often takes longer than a
  command's time limit. Stay with it until it returns. If your session ends,
  follow it again with `kura run execute <run-id>` (as for training); it never generates twice.
- A render cut short does not continue. The rest is a new render run.
- When it returns, check how many images were made with `kura run status <run-id>`.

### 7. Show the result
- Give the folder's full path. When the user asks you to open it, `kura open <run-id>` opens it
  on their desktop. Do not open it unasked.
- **The comparison sheet**, when it was planned, is made without asking again:
  - lay out only the finished images, with tools already installed: Pillow, ImageMagick
    (`montage`), or ffmpeg (scale each image to one size, then `hstack` and `vstack`). Ask
    before installing one. Never overwrite the images.
  - When all compared checkpoints come from one training run, save it in
    `comparisons/<training run id>/` at the top of the workspace (not inside a run's folder:
    a run's folder holds its record) and write the images it used in that run's `notes.md`.
  - `kura run status <training run>` then lists the render and the sheet's folder, and
    `kura open <training run>` opens that folder.
  - Report the sheet's full path; the single images' folder only when asked.
- Laying images out does not need you to look at them. Judging their content does, and needs
  `agents.view_images: true`.

## ComfyUI

- `kura render new` copies `comfyui.endpoint` from `workspace.yaml` into the run's
  `generator.endpoint`. The run's value is the one used.
- When it does not answer, ask the user to start it or to say which one to use. **Do not switch
  to another port you found on your own.**
- When the user decides to change it, change `comfyui.endpoint` in `workspace.yaml` (for runs
  from now on). For the current run, make it again.
- When `comfyui.lora_dir` (ComfyUI's LoRA folder) is not set, ask the user once where it is and,
  with their yes, write it in `workspace.yaml`.
- Check with `kura doctor comfyui --endpoint <url> --workflow <api.json>`: whether the models
  are visible and the LoRA can be placed.
- Kura neither starts ComfyUI nor downloads models. Tell the user which model is missing.

## When Kura cannot do it

- The workflow has no node to take the value, a model is not visible, ComfyUI is not there:
  stop and say what is missing.
- Do not make one workflow per value, split the work into many runs to get around a refusal, or
  call ComfyUI directly.

## References

- references/cases.md: writing cases, finding nodes, image inputs, the LoRA's place
- references/comfyui.md: how Kura places LoRAs and images, and what to check when they are not
  visible
- references/runpod.md: rendering on RunPod (a cost decision)
