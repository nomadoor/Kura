# Kura

[![日本語 README](https://img.shields.io/badge/README-日本語-blue)](README.ja.md)
[![Krea 2 LoRA guide](https://img.shields.io/badge/Guide-Krea_2_LoRA-blue)](https://comfyui.nomadoor.net/en/notes/kura-krea2-lora-training/)

Kura is a workspace for training LoRAs together with an AI agent.

You decide only what you want to make and which data to use. The agent can work out the settings, run the training, and test-render in ComfyUI to compare the results.

<img width="1905" height="1154" alt="kuramonitor" src="https://github.com/user-attachments/assets/89d09a7e-d5da-4496-86ee-aa14cda30058" />

## What Kura does

Kura is not a trainer itself. It manages the training tools [AI-Toolkit](https://github.com/ostris/ai-toolkit), [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner), and [sd-scripts](https://github.com/kohya-ss/sd-scripts) so an agent can use them safely while keeping a record of everything.

- **Trains on exactly the data you gave it**: only the files listed in the dataset are used for training. Old files or draft images left in the folder never slip in unnoticed.
- **Starts from settings that work**: Kura fills each model's detailed settings with the training tool's own recommended values. You or the agent set only what you want to change.
- **Goes from training to test renders in one flow**: test-render the LoRA you trained in your own ComfyUI workflow and compare results under different conditions.
- **Runs on your PC or in the cloud**: train on your own GPU or on a [RunPod](https://www.runpod.io/) cloud GPU with the same steps.
- **Keeps everything as files**: settings, the data used, logs, and outputs all stay in the workspace as plain files. You can inspect them later or run the same training again.

## What you need

| What | Why | When |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Runs Kura | Always (the setup below installs it) |
| Docker | Runs the training tools | When you train on your own PC. [Docker Desktop](https://docs.docker.com/get-started/get-docker/) on Windows and Mac, Docker Engine on Linux |
| NVIDIA GPU | Trains | When you train on your own PC |
| [RunPod](https://www.runpod.io/) account | Trains on cloud GPUs | When you use RunPod |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Test-renders | When you test-render |

If you train only on RunPod, you do not need a local GPU. Macs have no NVIDIA GPU, so training runs on RunPod.

## Using Kura on Windows

On Windows, Kura runs inside WSL2 (Ubuntu running inside Windows). Kura cannot run directly on Windows.

1. **Install WSL2 and Ubuntu**: open PowerShell as administrator, run `wsl --install`, and restart your PC. In the Ubuntu window that opens after the restart, choose a user name and password. Skip this if you already use Ubuntu on WSL.
2. **Install Docker Desktop**: install and start [Docker Desktop](https://docs.docker.com/get-started/get-docker/), then turn on Ubuntu under Settings → Resources → WSL integration.
3. **Update your NVIDIA driver**: if you train on your own PC, install the latest NVIDIA driver on Windows.

Do all of the setup and every later step in the Ubuntu terminal. Keep the Kura folder under your Ubuntu home (`~/`), not under `/mnt/c`. Training reads files under `/mnt/c` more slowly.

## Setup

```sh
# 1. Install uv (only if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get Kura and prepare it
git clone https://github.com/nomadoor/Kura.git
cd Kura
uv sync          # install Kura and what it needs
uv run kura init # create the working folders and default settings

# 3. Create the secrets file
cp .env.example .env.local
```

Put only what you use into `.env.local`. The file stays out of Git, and Kura reads it automatically.

| Variable | When you need it |
| --- | --- |
| `RUNPOD_API_KEY` | You train on RunPod |
| `HF_TOKEN` | You use a model that needs access approval (such as FLUX.1-dev) |
| `KURA_NTFY_TOPIC` | You want a finish notification on your phone or PC (optional) |

To check that you are ready:

```sh
uv run kura doctor docker   # Docker and the GPU work (training on your own PC)
uv run kura doctor runpod   # the RunPod settings are correct (using RunPod)
```

## How to use it

### The flow

Start an AI agent (Claude Code, Codex, and so on) in the Kura folder and talk to it. The agent reads Kura's rules (`AGENTS.md`) before it starts.

1. 🧑 Put images and captions (a `.txt` with the same name as each image) in `datasets/<name>/`
2. 🧑 Say what you want. For example: "I want a Krea 2 character LoRA from this dataset." You can also specify details such as the rank or learning rate
3. 🤖 Checks the data and writes the list of files to train on (`dataset.yaml` and `items.jsonl`) and the training settings
4. 🤖 Shows a plan that covers the settings, where to train (your PC or RunPod), and the expected time and cost
5. 🧑 Reviews the plan and approves it, or says what to change
6. 🤖 Runs the training and reports the result when it finishes
7. 🤖 Test-renders in ComfyUI, lining up saved checkpoints and different conditions for comparison
8. 🧑 Looks at the results and decides whether to stop or keep training (continuing from a saved state is supported)

> 💡 For how to put a dataset together, [Training an SDXL (Illustrious) LoRA with AI-Toolkit](https://comfyui.nomadoor.net/en/notes/ai-toolkit-sdxl-lora-training/) may help. It is written for SDXL, but the approach is the same.

### Where to train: your PC or RunPod

- **Your PC**: training runs in Docker. A model is downloaded once and reused afterwards.
- **RunPod**: Kura sends only the files the training needs, trains, collects the outputs, and then **stops the Pod automatically**. If the GPU you want is not available, you choose in the plan whether to use another GPU or wait for it (no charge while waiting). Even if your PC goes down, the Pod has a maximum running time (`--max-lease`, 12 hours by default) as a safety net. Only when you want to inspect results on the Pod itself, `uv run kura run remote <run-id> --hold-for 30m` delays the stop.

The agent suggests where to train based on GPU size and cost, and you decide in the plan.

### Test-rendering in ComfyUI

Test renders use ComfyUI and **API-format** workflows placed in `workflows/`.

- Start ComfyUI at `http://127.0.0.1:8188`.
- Export a workflow with "File → Export (API)" in ComfyUI and put it in `workflows/`. For details, see [Using ComfyUI from an AI agent](https://comfyui.nomadoor.net/en/data-utilities/ai-agent-api/).
- If you have no local GPU, you can also render on a disposable ComfyUI on RunPod.

### Watching progress

You can watch training from another terminal. This screen is only for watching; it does not start or stop training.

```sh
uv run kura monitor             # list every training run
uv run kura run watch <run-id>  # one run in detail
```

## Where files go and cleaning up

| Location | Contents |
| --- | --- |
| `datasets/<name>/` | Your datasets |
| `runs/<run-id>/outputs/` | The LoRAs you trained |
| `artifacts/training-state/` | Saved state for continuing training |
| `cache/huggingface/` | Downloaded models (tens of GB) |

None of these go into Git. If disk space is a concern, you can look first without changing anything:

```sh
uv run kura doctor disk   # what uses how much space (read-only)
uv run kura cleanup all   # what can be removed (add --yes to delete)
uv run kura run prune     # older training runs (add --yes to delete)
```

To free the model cache, delete `cache/huggingface/`. Models are downloaded again when needed.

## Supported models

The main ones are below. How far each was actually verified by training is in the [support table](docs/backend-support.md).

| Training tool | Main models |
| --- | --- |
| sd-scripts | SD 1.5, SDXL, FLUX.1, Anima (LoRA / ControlNet-LLLite) |
| Musubi Tuner | Wan, FLUX.2, Krea 2, Qwen-Image, Z-Image, FLUX.1 Kontext, HiDream-O1, Ideogram 4, HunyuanVideo, FramePack, Kandinsky 5, MiniMax-H3, and more |
| AI-Toolkit | SD 1.5, SDXL, FLUX.1 / Kontext / Flex.2, Chroma, Qwen-Image, FLUX.2, Krea 2, Z-Image, HiDream, Anima, MiniMax-H3, and more |

"Supported" here means training runs and its outputs are saved. It does not promise a good LoRA from any data or settings.

## Updating Kura

```sh
git pull
```

That is all. Anything else needed is set up the next time you run `uv run kura ...`. Kura also manages the training tools' Docker images and pins them to versions it has verified. You never build them or choose versions yourself.

## More

- [Training a Krea 2 LoRA with Kura](https://comfyui.nomadoor.net/en/notes/kura-krea2-lora-training/): a worked example from preparing data to comparing in ComfyUI
- [docs/commands.md](docs/commands.md): command reference
- [docs/backend-support.md](docs/backend-support.md): the support table and what was verified
- [docs/agent-first-cli.md](docs/agent-first-cli.md): what the agent writes and what Kura guarantees
- [AGENTS.md](AGENTS.md): rules for AI agents

## License

MIT
