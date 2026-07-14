# Kura

[![日本語 README](https://img.shields.io/badge/README-日本語-blue)](README.ja.md)

Kura is a file-first experiment workspace for working with AI agents on LoRA training and image-comparison runs.

You decide what you want and which data to use; the agent creates, runs, monitors, and compares the runs for you (you can also drive it manually from the CLI).

<img width="1920" height="1080" alt="Kura" src="https://github.com/user-attachments/assets/a23f92f8-460c-40d8-be8f-e42a5ef06f72" />

## What Kura is

Kura is not a trainer itself. It is a thin management layer around training tools such as [AI-Toolkit](https://github.com/ostris/ai-toolkit) and [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner), designed to make them **safer, reproducible, and easy for agents to operate**.

Training runs in Docker (locally) or on RunPod (remotely), and everything — settings and results — is stored as plain files, so any run can be reviewed and reproduced later.

If you keep ComfyUI running, you can also test-generate with the LoRA you trained using your chosen workflow.

## Getting started

### What you need

| Requirement | What it's for | How to get it |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | The Python environment manager that runs Kura | The one-line command in setup step 0 |
| [Docker](https://docs.docker.com/get-started/get-docker/) | Runs the training tools in containers (you don't need to know the internals) | Just install Docker Desktop |
| NVIDIA GPU | Needed for practical local training | Not needed if you only use RunPod |
| [RunPod](https://www.runpod.io/) account | For training on cloud GPUs | Get an API key |

- **Windows / WSL2:** enable Docker Desktop's WSL integration for your distribution.
- A **Hugging Face token** is only needed for gated/private models.
- For **render runs**, have ComfyUI running at `http://127.0.0.1:8188`.

### Setup

```sh
# 0. Install uv (only if you don't have it yet; macOS / Linux / WSL)
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex

# 1. Get Kura
git clone https://github.com/nomadoor/Kura.git
cd Kura

# 2. Install Kura and its dependencies
uv sync

# 3. Create the workspace folders and default config (datasets/, runs/, workspace.yaml, ...)
uv run kura init

# 4. Set up your secrets
cp .env.example .env.local
```

Open `.env.local` and fill in only what you need. **`.env.local` is ignored by Git and loaded automatically by every `kura` command** — there is nothing to source or export.

| Variable | When to set it |
| --- | --- |
| `RUNPOD_API_KEY` | If you train on RunPod (almost always) |
| `HF_TOKEN` | Only for gated/private models |
| `KURA_NTFY_TOPIC` | Only if you want completion notifications (optional) |

There are no other variables to set. (If you publish your own Docker images, just run `docker login` first.)

Local Docker runs reuse Hugging Face downloads through `cache/huggingface/` by default. The directory is ignored by Git. If you want the cache somewhere else, change the mount source in `workspace.yaml`.

## Working with an AI agent

You (🧑) decide the direction; the agent (🤖) does the hands-on work.

1. 🧑 Put your dataset under `datasets/`.
2. 🧑 Tell the agent the goal — e.g. `Train a Krea 2 character LoRA with this dataset.` (You can also spell out exact parameters like `rank 16, lr 5e-5, ...` if you want.)
3. 🤖 Inspect the dataset and write `run.yaml`. For RunPod, use a draft `kura run plan` to check GPU stock and record the immediate/wait choice; then compile and show the final plan with assumptions, resource facts, and trade-offs.
4. 🧑 Approve that plan once, or tell the agent what to change.
5. 🤖 Run `kura run execute <run-id>` using the local Docker or RunPod executor frozen in the plan. Infrastructure smoke checks are used when the backend or environment needs them; they are not a second user workflow.
   For RunPod, `execute` stops the Pod immediately after outputs are downloaded. Use `kura run remote <run-id> --hold-for 30m` instead when you need a review window before shutdown.
6. 🧑 Watch progress with `uv run kura monitor` (or have 🤖 report progress).
7. 🤖 Generate ComfyUI comparison images in a render run linked to the training run and checkpoint.
8. 🧑 Record your judgment in the render run's `notes.md` and decide whether to stop or run another experiment (🤖 carries out the instruction).

> 💡 For the basics of building a dataset, [Training an SDXL (Illustrious) LoRA with AI-Toolkit](https://comfyui.nomadoor.net/en/notes/ai-toolkit-sdxl-lora-training/) is a useful reference (it targets SDXL, but the approach carries over).

## Monitoring

`kura monitor` is a **read-only** TUI showing active and historical runs, loss, progress, GPU/RunPod info, and output paths. It does not start or stop training.

```sh
uv run kura monitor            # list
uv run kura run watch <run-id> # one run in detail
```

## Image generation with ComfyUI (optional)

Start ComfyUI and drop an **API-format** workflow into `workflows/`, and the agent can generate images with your trained LoRA — test renders, step/strength comparison grids, promptset batches, and so on, depending on the workflow you provide.

No local GPU? `uv run kura render launch <run-id> --executor runpod` runs the same on a disposable RunPod ComfyUI Pod (models are pulled from Hugging Face, only the LoRA is uploaded, and the Pod stops automatically when done).

> Export the workflow with ComfyUI's "File → Export (API)" (the regular UI export is not accepted by `/prompt`). See [Using ComfyUI from an AI agent](https://comfyui.nomadoor.net/en/data-utilities/ai-agent-api/) for details.

## RunPod safety

RunPod runs use **disposable Pods**. Kura uploads only the inputs it needs, trains, downloads the outputs, and then **stops the Pod automatically**, so you don't leave GPUs billing in the background.

- By default it does not use Network Volumes (no persistent storage left behind, flexible GPU placement).
- Normal `kura run execute` stops the Pod immediately after confirmed output recovery.
- A draft `kura run plan` checks current stock for the ordered RunPod GPU candidates. If the selected GPU is unavailable, choose an available alternative or record `compute.capacity: {mode: wait, timeout: 6h}` before compile; the final plan approves that choice once. Closing the terminal ends a foreground wait.
- Use the low-level `kura run remote <run-id> --hold-for 30m` only when you intentionally need a review window.
- A Pod-side `--max-lease 12h` guard is a last billing fuse if the local controller dies.

## Where files live & cleanup

Kura keeps everything as files inside the workspace.

| Location | Contents |
| --- | --- |
| `datasets/<id>/` | Your datasets (images + captions) |
| `runs/<run-id>/outputs/` | Trained LoRAs and other results |
| `cache/huggingface/` | Downloaded model weights (can be tens of GB) |

None of these are tracked by Git. If disk is a concern, start with a read-only look:

```sh
uv run kura doctor disk                                           # see what's using space and how much is free (read-only)
uv run kura cleanup all                                           # preview deletion candidates (dry-run; add --yes to apply)
uv run kura run prune                                             # remove old runs (add --yes to apply)
uv run kura run prune --docker-containers --docker-volumes --yes  # also remove Kura-managed stopped containers/volumes
```

To reclaim the model cache, delete `cache/huggingface/` (it re-downloads when needed).

## Updating Kura

```sh
git pull
```

That is the whole update. Dependencies re-sync automatically on the next `uv run kura ...`, and the training Docker images are part of Kura: they are pulled automatically when needed and move forward together with Kura releases. You never build or manage images yourself.

Two notes on how trainer versions work:

- **AI-Toolkit** runs on a Kura-tested version of the upstream official image. Kura updates that pin deliberately after compatibility checks; it does not follow the mutable `latest` tag during normal runs.
- **Musubi Tuner** has no official image, so Kura ships one that is tested against Kura's own Musubi support. A newer Musubi alone would not add new model support anyway — that always needs a Kura update too — so the image updates together with Kura, and "what does my Kura support?" always has one answer: whatever your current Kura supports.

## Learn more

- [docs/commands.md](docs/commands.md): full command reference
- [docs/agent-first-cli.md](docs/agent-first-cli.md): what the agent authors, what the CLI guarantees, and how runs work without conversational state
- [docs/backend-support.md](docs/backend-support.md): pinned backend versions and verified support paths
- [AGENTS.md](AGENTS.md) and [.claude/skills/](.claude/skills/): always-loaded agent rules and task-specific instructions (have your agent read `AGENTS.md` first)
- [docs/smoke-test.md](docs/smoke-test.md): smoke test notes
- [README.ja.md](README.ja.md): Japanese version

## License

MIT
