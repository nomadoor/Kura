---
name: train
description: Train a LoRA with Kura, on this machine (Docker) or on RunPod. Use when the user wants to train, retrain, or continue training a LoRA, change training settings, or when a training run is slow, out of memory, failed, stopped, or needs to be followed or recovered.
---

# Train

Turn what the user wants into one run, get their yes to its plan, and see it through to a
result they can use. A run is a folder under `runs/`: `run.yaml` is what you write, and Kura
freezes it when it compiles, so a compiled run never changes. Any change is a new run.

## What matters

1. **The recipe serves the user's intent.** Every trade-off of quality, memory, time, or cost
   is theirs. Propose one recipe with the reason for each setting; do not make them fill in a
   questionnaire.
2. **Facts come from Kura, not from memory.** What a backend accepts, what fits, what changed
   since the last run, and what a run costs: ask Kura.
3. **One plan, one yes, then the run to its end.** Nothing changes silently: not a setting,
   not the GPU, not the place it runs.

## Who decides

- **Kura**: the settings a backend accepts (`kura run capabilities`), whether the run fits the
  disk, RunPod stock, price, and cost ceiling, what this run changes from the previous one in
  its experiment, and the run's state. Its refusals name the fix.
- **You**: the proposed recipe and the reason for each setting, following the run, and the
  report.
- **The user**: what to train and how far, the recipe, the machine, and every cost.

## Steps

1. **Check the machine.** Local: `kura doctor docker` and `kura doctor disk`. RunPod:
   `kura doctor runpod`. If disk warns, read references/disk.md before going on.
2. **Check the dataset.** It must validate (`kura dataset validate <dataset>`); otherwise use
   the `dataset` skill first.
3. **Create the run.** `kura run new --experiment <name> --slug <words> --backend <backend>`
   (add `--executor runpod --gpu <type>` for RunPod). Runs answering one question share an
   experiment name; the plan then compares them.
4. **Gather facts.** `kura run capabilities <backend> --arch <value>` (the fields and the
   model's baseline), `kura dataset inspect <dataset>`, the family card (below), the user's
   `knowledge/user-preferences.md`, and earlier runs of the experiment.
5. **Write `run.yaml`.** Name the model as references/models.md says. Take each setting from
   the strongest source: the user's words, their preferences, an earlier run they judged, the
   family card, and last your own judgment, said as such. Set `save_every_n_steps` so the user
   can compare steps later; every save is kept unless the run sets a limit.
6. **Compile and read the plan.** `kura run compile <run-id>`, then `kura run plan <run-id>`.
   Read what it changes from the previous run, the disk and checkpoint lines, the time, and on
   RunPod the GPU, hourly price, and cost ceiling. Then read `.kura/knowledge/regrets.md` and
   `knowledge/regrets.md` for anything that applies.
7. **Ask for the yes.** Show the plan following the `report` skill: the recipe with its
   sources, what changed, time and cost, and anything that looks wrong. A value far from the
   previous run or the card goes first, as a question. Wait for the user's yes to this plan.
8. **Start it.** `kura run execute <run-id> --yes`, run through your host's way of tracking a
   long command (never `nohup ... &`), and stay with it until it returns. Kura's job runner
   keeps the run going if your session ends; `kura run execute <run-id>` (no `--yes`) follows
   it again and never starts it twice. Kura watches the run; do not write a watcher of your own.
9. **When it returns,** read the exit code and `kura run status <run-id>`
   (references/states.md says what each state means and what to do next).
   - Exit 3: the run needs a look and is still going. Show the user the line Kura printed and
     the end of `kura run logs <run-id>`, and ask whether to stop it (`kura run stop`) or keep
     following.
   - Completed: check the outputs it lists and report. Where to look and how to compare
     steps is the `render` skill; offer it only if the user wants to see results.
   - Anything else: report what Kura says and the next step it names. A fix is a new run.
10. **Write down what was learned** in the run's `notes.md`: settings, machine, what happened,
    and the user's judgment.

## When the run does not fit or is too slow

Fix the smallest thing that is enough, and stop there:

1. Changes that keep the result: a lighter but equivalent model file, a cache.
2. Changes that only cost time: gradient checkpointing, a smaller batch with accumulation
   (each step then takes that many times longer), offloading. Say how much slower.
3. Changes to the result: resolution, batch, rank, learning rate, the model, or the dataset.
   Offer concrete choices and recommend one.

Anything in 3, a slowdown beyond about twice, or a different GPU or machine is the user's
decision and a new plan. After an out-of-memory failure, read the log first.

**AI-Toolkit on a GPU of about 12 GB:** keeping the text encoder in memory can spill into
system memory; the run then does not fail but runs tens of times slower. Set
`native_config.train.cache_text_embeddings: true` (each caption is then encoded once, so
caption shuffling and token dropout apply once). Never set `unload_text_encoder: true` without
it: the trainer would train on empty captions.

## Continuing a run

`kura run resume <run-id> --additional-steps <N>` (or `--to-step <N>`) makes a new run that
continues from the source run's saved training state. Compile it and show its plan: it says
how much of the state is restored. Starting fresh from trained weights is a different thing;
do not call it a resume. To retrain with changed settings, `kura run new --from <run-id>`.

## Model-family cards

Cards say what has worked for a model family: `.kura/knowledge/model-families/<family>.md`
(Kura's) and `knowledge/model-families/<family>.md` (the user's, which wins). Their
**Prompts** section is how the model reads text, for captions and prompts alike. A card is
knowledge, not a list of what Kura can train: `kura run capabilities` is. Without a card, say
so once and work from the model's official documents. Two variants of one family (one to train,
one to generate) are not compatible because their names look alike; ask. Only a run the user
judged, or the user, adds to their card, with a `source:` line.

## References

- references/models.md: how to name the model for each backend.
- references/states.md: what each run state means and what to do next.
- references/runpod.md: cost, lease, and recovery on RunPod.
- references/disk.md: disk checks, and what to do when space is short.
