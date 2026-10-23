# Moving from a Kura clone to a workspace

Until now, setting up Kura meant `git clone`, `uv sync`, and `uv run kura init`
inside the clone, so your datasets, runs, and models lived in the Kura source
folder. Kura is now installed as a tool, and your work lives in a separate
folder, the workspace. This page moves an existing clone's work into one.

Nothing in your run records needs rewriting: paths inside them are either
relative to the workspace or historical facts, and a moved run reads, cleans
up, and resumes as before.

## Why not keep using the clone?

Kura recognizes a source checkout and never writes its agent files there, so
the clone's `AGENTS.md` and skills stay the development guide for changing
Kura, written for `uv run kura` from that checkout. An agent working there would
use that copy of Kura and its rules, not the Kura you installed. Move your work
out instead.

## Steps

The commands assume the clone is `~/Kura` and the new workspace is
`~/kura-workspace`. Use any folder you like, on the same drive if you can (a move
within one drive is instant; across drives it copies).

1. **Finish the runs in progress.** In the clone, run `uv run kura monitor`.
   Wait until no training or render is running, waiting for a GPU, or being
   downloaded. A run that is still running writes to the clone, and a local
   container keeps the clone's path. If a run looks stuck, run
   `uv run kura run reconcile <run-id>` and follow what it says. Then stop the
   job runner:

   ```sh
   cd ~/Kura
   uv run kura runner stop
   ```

2. **Install Kura and create the workspace.**

   ```sh
   uv tool install git+https://github.com/nomadoor/Kura
   mkdir ~/kura-workspace && cd ~/kura-workspace
   kura init
   ```

3. **Move your work.** Use `mv`, not a copy. A run's saved training state is
   shared with its `outputs/` by hard links, and a copy made folder by folder
   stores it twice. Move `runs/` and `datasets/` together: links inside a run
   point into `datasets/` by relative path.

   ```sh
   mv ~/Kura/runs/* runs/
   mv ~/Kura/artifacts/training-state/* artifacts/training-state/
   mv ~/Kura/datasets/* datasets/
   mv ~/Kura/promptsets/* promptsets/
   ```

   Also move these if you have them:

   - your own files in `~/Kura/workflows/` (not `workflows/samples/`, which
     Kura writes itself);
   - `~/Kura/cases/`;
   - your own notes in `~/Kura/knowledge/`, into `knowledge/` (replacing the
     two starter files `kura init` wrote there).

4. **Bring your settings.** Copy the clone's `workspace.yaml` over the new one:

   ```sh
   cp ~/Kura/workspace.yaml workspace.yaml
   ```

   Absolute paths in it (such as `comfyui.lora_dir`) still work. A relative path
   now means a path inside the new workspace. If a command then refuses the
   file, run `kura workspace migrate` and then `kura workspace migrate --yes`.

5. **Keep the downloaded models.** `cache/huggingface/` holds the models (often
   tens of GB). Move it so nothing is downloaded again:

   ```sh
   rmdir cache/huggingface
   mv ~/Kura/cache/huggingface cache/
   ```

   Or leave it where it is and point the workspace at it in `workspace.yaml`:

   ```yaml
   docker:
     hf_cache: /home/<you>/Kura/cache/huggingface
   ```

   Moving it is better. Kura treats a cache outside the workspace as possibly
   shared, so `kura cleanup --models unreferenced` refuses it, and deleting the
   clone would delete the models. If `workspace.yaml` already sets
   `docker.hf_cache`, keep that setting and skip this step.

6. **Set up your secrets.** Either move the clone's secrets file:

   ```sh
   mv ~/Kura/.env.local .
   ```

   or store each secret once for every workspace and leave `.env.local` out:

   ```sh
   kura secrets set RUNPOD_API_KEY
   kura secrets set HF_TOKEN
   ```

   A workspace's `.env.local` overrides the user-level file. `kura doctor secrets`
   shows which one each secret comes from, without showing values.

7. **Rebuild the run list and check.**

   ```sh
   kura index rebuild
   kura doctor workspace
   kura monitor
   kura cleanup plan
   ```

   Your runs appear with their outputs, and `kura run status <run-id>` reads as
   before.

## What not to move

| In the clone | Why it stays |
| --- | --- |
| `index.jsonl` | `kura index rebuild` writes it from `runs/` |
| `.kura/` | The clone's job runner state; the new workspace has its own |
| `cache/models/` | Links that each launch writes again |
| `cache/` (anything else) | Temporary or downloaded again when needed |
| `transfer/` | Temporary files |
| `AGENTS.md`, `.claude/`, `.agents/`, `src/`, `docs/` | Kura's source; `kura init` wrote the workspace's own agent files |

## Afterwards

- Start your AI agent in the workspace, not in the clone.
- To resume a run trained before the move, use `kura run resume` as usual. The
  plan may warn that the Kura that trained it was a clone and this one is
  installed; that warning does not stop the resume.
- Stopped containers of moved runs belong to the workspace now, and
  `kura cleanup plan` lists them.
- Once `kura monitor` and `kura cleanup plan` look right, delete the clone
  (`rm -rf ~/Kura`). Check first that you moved `cache/huggingface/` or no
  longer need it. If you work on Kura itself, keep a clone for that only and
  do not train in it.
- To update Kura from now on, run `uv tool upgrade kura` instead of `git pull`.
