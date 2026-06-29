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
2. 🧑 Tell the agent the goal — e.g. `Train a Krea 2 character LoRA with this dataset.`
3. 🧑 Tell the agent the parameters — e.g. `rank 16, alpha 16, lr 5e-5, batch 2, 768px, 1500 steps, save every 100 steps.`
4. 🤖 Create and try a local smoke run.
5. 🤖 If it passes, run locally or on RunPod.
6. 🧑 Watch progress with `uv run kura monitor` (or have 🤖 report progress).
7. 🤖 (Optional) Pull intermediate checkpoints and generate ComfyUI comparison images.
8. 🧑 Review the results and decide whether to stop or keep training (🤖 carries out the instruction).

> 💡 For the basics of building a dataset, [Training an SDXL (Illustrious) LoRA with AI-Toolkit](https://comfyui.nomadoor.net/en/notes/ai-toolkit-sdxl-lora-training/) is a useful reference (it targets SDXL, but the approach carries over).

## Monitoring

`kura monitor` is a **read-only** TUI showing active and historical runs, loss, progress, GPU/RunPod info, and output paths. It does not start or stop training.

```sh
uv run kura monitor            # list
uv run kura run watch <run-id> # one run in detail
```

## RunPod safety

RunPod runs use **disposable Pods**. Kura uploads only the inputs it needs, trains, downloads the outputs, and then **stops the Pod automatically**, so you don't leave GPUs billing in the background.

- By default it does not use Network Volumes (no persistent storage left behind, flexible GPU placement).
- After download it keeps the Pod for `--hold-for 30m` by default so you can inspect the LoRA.
- It then terminates automatically unless you tell it otherwise.
- A Pod-side `--max-lease 12h` guard is a last billing fuse if the local controller dies.

## Where files live & cleanup

Kura keeps everything as files inside the workspace.

| Location | Contents |
| --- | --- |
| `datasets/<id>/` | Your datasets (images + captions) |
| `runs/<run-id>/outputs/` | Trained LoRAs and other results |
| `cache/huggingface/` | Downloaded model weights (can be tens of GB) |

None of these are tracked by Git. When you no longer need them:

```sh
uv run kura run prune                                              # remove old runs (add --yes to apply)
uv run kura run prune --docker-containers --docker-volumes --yes   # also remove Kura-managed stopped containers/volumes
```

To reclaim the model cache, delete `cache/huggingface/` (it re-downloads when needed).

## Learn more

- [docs/commands.md](docs/commands.md): full command reference
- [AGENTS.md](AGENTS.md) and [.claude/skills/](.claude/skills/): always-loaded agent rules and task-specific instructions (have your agent read `AGENTS.md` first)
- [docs/smoke-test.md](docs/smoke-test.md): smoke test notes
- [README.ja.md](README.ja.md): Japanese version

## License

MIT
