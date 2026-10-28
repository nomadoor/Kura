---
name: train
description: Train a LoRA with Kura, on this machine (Docker) or on RunPod. Use when the user wants to train, retrain, or continue training a LoRA, change training settings, or when a training run is slow, out of memory, failed, stopped, or needs to be followed again or recovered.
---

# Train

Turn what the user wants to train into one run, get their yes to its plan, and follow it until
training ends.

## Flow

```text
kura run new            create the run (runs/<run-id>/)
   ↓ write run.yaml
kura run compile        freeze it; from here this run never changes
kura run plan           show it to the user
   ├─ change it ─→ make a new run with kura run new --from (back to the top)
   └─ yes
kura run execute --yes  train; follow it to the end
kura run status         check the result
```

## Who decides

| | Decides |
|---|---|
| Kura | the settings a backend accepts, whether the run fits the disk, RunPod stock, price, and cost ceiling, what changed from the previous run, the run's state |
| You | the proposed backend, model, and settings with their reasons; following the run; the report |
| The user | what to train and how far, the backend, the settings, the machine, every cost |

## Steps

### 1. Check the machine and the dataset
- Local: `kura doctor docker` and `kura doctor disk`. RunPod: `kura doctor runpod`. If the
  disk warns, read references/disk.md.
- Check the dataset: `kura dataset validate <dataset>`. Go to part 2 of the `dataset` skill
  ("Make it a Kura dataset and check it"), then come back, when:
  - the user only handed you a folder of images or videos that is not a dataset yet
  - validation fails
  - compile refuses because the dataset's shape does not fit this model (step 5)
- Do not touch the dataset's content (selection, crops, rewriting captions) unless the user
  asks.

### 2. Choose the backend and the model
- First read the family card ("Model-family cards" below): it says which variant to train and
  which backend has records.
- If the user named a backend, use it.
- If not, look for the model among the selector values (`model_arch` or `architecture`) of
  `kura run capabilities ai-toolkit`, `musubi-tuner`, and `sd-scripts`.
  - One backend has it: use it.
  - Several do: recommend one with the difference (which one the card has records for, what
    each can do) and let the user choose. Do not pick silently.
- Each backend names the model differently: references/models.md.

### 3. Create the run
Three ways, by what the user wants:

| To | Command |
|---|---|
| train something new | `kura run new --experiment <name> --slug <words> --backend <backend>` |
| train again with changed settings | `kura run new --from <run-id> --slug <words>` |
| continue a run's training | `kura run resume <run-id> --additional-steps <N>` |

- For RunPod add `--executor runpod`; choosing the GPU is in references/runpod.md.
- Give runs the user wants to compare the same experiment name; the plan then lists what
  changed from the previous one. If the user gave none, name it from the dataset and the model
  and say so.
- Whichever way, continue from step 4 (check `run.yaml`, compile, plan, yes).
- `--from` copies the settings, so it takes no `--backend`, `--executor`, or `--gpu`; change
  those in `run.yaml`.
- A resume continues from the training state the source run saved. Starting fresh from trained
  weights is a different thing; do not call it a resume. Restoring the state needs
  `lr_scheduler: constant` and an `adamw` or `adamw8bit` optimizer (`kura run capabilities`
  says so).

### 4. Fill in `run.yaml`
`kura run new` writes a draft with empty fields. Fill each:

| Field | What goes there |
|---|---|
| `intent` | what the user wants, in one sentence |
| `datasets` | `- id: <dataset id>` |
| `model.base` | the model to train from (references/models.md) |
| `backend.config` | the model selector (`model_arch` or `architecture`) and the settings you chose |
| `recipe` | `steps` (training steps) and `seed` |
| `compute` | where it runs; for RunPod, the GPU too |
| `sampling` | leave as it is (no sample images during training; trying the LoRA is the `render` skill) |

- `kura run capabilities <backend> --arch <value>` lists the `backend.config` fields and, for
  AI-Toolkit, that model's baseline. AI-Toolkit fills the fields you leave out from the
  baseline, so write only what you chose.
- The baseline has no rank or alpha (`network_dim`, `network_alpha`). Always propose both and
  put them in the plan; without a rank the plan cannot estimate the file sizes.
- Take each setting from the strongest source:
  1. the user's words
  2. the user's preferences (`knowledge/user-preferences.md`)
  3. an earlier run the user judged (same experiment)
  4. the card, or the baseline
  5. your own judgment (say it is yours)
- Propose a save interval (`save_every_n_steps`) and put it in the plan. Every save is kept,
  and the plan shows how many. The user needs them to compare steps.
- Dataset repeats differ by backend: sd-scripts needs `num_repeats` (usually 1); AI-Toolkit and
  Musubi Tuner mean 1 when it is left out. Do not raise it unless the user wants to.

### 5. Show the plan and get a yes
- `kura run compile <run-id>`, then `kura run plan <run-id>`. If compile refuses, follow the
  fix it names; if the reason is the dataset's shape, go to part 2 of the `dataset` skill.
- Read in the plan: what changed from the previous run, the disk and checkpoint lines, and on
  RunPod the GPU, hourly price, and cost ceiling (references/runpod.md).
- The plan gives no time. Estimate it from the card's measured speed (seconds per step × steps)
  and say it is an estimate from the card; without a measurement, say it is not known.
- Read the user's `knowledge/regrets.md` (what they regretted in earlier runs) for anything
  that applies, and add it to the plan.
- Show the plan by the `writing` skill. A value that does not match what the user said, or that
  changed though they did not ask for it, comes first, as "Is this value right?".
- **On a yes**, go to step 6.
- **When the user asks for changes**, a compiled run cannot change, so make it again:
  1. `kura run new --from <run-id> --slug <words>` copies the settings
  2. change what they asked, compile, and show the plan again
  3. remove the run you no longer use with `kura run discard <run-id> --yes` (only a run never
     started can be removed)

### 6. Follow the training
- Run `kura run execute <run-id> --yes` through your host's way of tracking a long command
  (never `nohup ... &`) and stay with it until it returns.
- If your session ends, the run goes on. Follow it again with `kura run execute <run-id>` (no
  `--yes`); it never starts twice.
- Kura watches the run; do not write a watcher of your own.
- **Exit code 3**: the run is still going and needs a look. Show the user the line Kura printed
  and the end of `kura run logs <run-id>`, and ask whether to stop it (`kura run stop`) or keep
  following.

### 7. Check the result
- Read `kura run status <run-id>`. What each state means and what comes next is in
  references/states.md.
- Completed: check the outputs it lists and report. Renders and comparisons only when the user
  asks (the `render` skill).
- Failed or interrupted: report what Kura says and the next step it names. A fix is a new run.
- Write what was learned in the run's `notes.md`: settings, machine, what happened, the user's
  judgment.

## Getting to the plan quickly

Do not spend long before showing the plan; `kura run plan` itself takes seconds.

- Look up only what a decision needs.
- Read the model's facts with `kura run capabilities <backend> --arch <value>`, for the model
  you use only. Read it as text; do not parse it. Its selector list is one long line: search it
  for your model.
- From `kura doctor docker`, read `diagnosis` (the first line); the rest is detail.
- Do not check again by hand what the plan reports (disk, downloads, what is cached, what
  changed from the previous run).
- When trustworthy information (the card, the baseline, the user's earlier runs) is not
  enough, look it up in official documents and name your source.
- Write `notes.md` after training, not before the plan.

## When memory is short or training is slow

Make the smallest change that is enough, and stop there.

| Level | Kind of change | Examples |
|---|---|---|
| 1 | keeps the result | a lighter equivalent model file, a cache |
| 2 | costs only time (say how much slower) | gradient checkpointing, a smaller batch with accumulation, offloading |
| 3 | changes the result | resolution, batch, rank, learning rate, the model, the dataset |

At every level the change is a new run with a plan and a yes. For level 3, offer choices and
recommend one.

- A different GPU or machine is always the user's decision and a new plan.
- After an out-of-memory failure, read the log first.
- What to do for a given model (speed and memory measured on a GPU) is in its card.

## Model-family cards

- Kura's are in `.kura/knowledge/model-families/<family>.md`, the user's in
  `knowledge/model-families/<family>.md`; the user's win where they disagree.
- They hold settings that worked, model files, and how the model reads text (**Prompts**).
- Training works without a card: say once that there is none, and work from the model's
  official documents.
- Some families train one variant and generate with another. Similar names do not make them
  compatible; ask.
- Only a run the user judged, or the user, adds to the user's card, with a `source:` line.

## References

- references/models.md: how each backend names the model
- references/states.md: run states and what comes next
- references/runpod.md: cost, lease, and recovery on RunPod
- references/disk.md: disk checks, and what to do when space is short
