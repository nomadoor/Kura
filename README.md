# Kura

[![日本語 README](https://img.shields.io/badge/README-日本語-blue)](README.ja.md)
[![Krea 2 LoRA guide](https://img.shields.io/badge/Guide-Krea_2_LoRA-blue)](https://comfyui.nomadoor.net/en/notes/kura-krea2-lora-training/)

Kura is a harness for training LoRAs with an AI agent.

The training itself is done by [AI-Toolkit](https://github.com/ostris/ai-toolkit), [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner), and [sd-scripts](https://github.com/kohya-ss/sd-scripts). Kura starts them safely, watches the training, and keeps a record of it.

A LoRA rarely comes out right the first time. Kura is a tool for running experiment after experiment.

<img width="1905" height="1154" alt="kuramonitor" src="https://github.com/user-attachments/assets/89d09a7e-d5da-4496-86ee-aa14cda30058" />

## Features

- Settings, data, logs, and outputs all stay as files
- The same steps train on your own PC or on RunPod
- A RunPod Pod is deleted once its outputs are collected, and deletes itself if your PC goes down (after 12 hours at most; `--max-lease` changes it)
- Training can resume from a saved state
- A trained LoRA can be test-rendered and compared in ComfyUI right away

## What you need

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An AI agent (Claude Code, Codex, or similar)
- To train on your own PC: Docker and an NVIDIA GPU
- To train on RunPod: a RunPod API key
- To test-render: [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Setup

```sh
uv tool install git+https://github.com/nomadoor/Kura
```

`kura init` makes the current folder a workspace. Datasets, training results, and downloaded models (tens of GB) go there. Do not run it straight in your home folder: make a folder for Kura, move into it, and run it there. Pick a disk with plenty of free space.

```sh
mkdir ~/kura && cd ~/kura
kura init
```

If you need them, register your API keys.

```sh
kura secrets set RUNPOD_API_KEY
kura secrets set HF_TOKEN
```

## Usage

Start your AI agent in the workspace and tell it what LoRA you want. The agent shows a plan, and training starts once you approve it.

You can watch training from another terminal.

```sh
kura monitor
```

## Updating

```sh
uv tool upgrade kura
```

## Windows

Use Kura inside Ubuntu on WSL2.

1. In PowerShell (as administrator), run `wsl --install` and restart
2. Install [Docker Desktop](https://docs.docker.com/get-started/get-docker/) and turn on Ubuntu under Settings → Resources → WSL integration
3. Work in the Ubuntu terminal from then on, and make the workspace under `~/`, not under `/mnt/c`

## More

- [Training a Krea 2 LoRA with Kura](https://comfyui.nomadoor.net/en/notes/kura-krea2-lora-training/)
- [Command reference](docs/commands.md)

## License

MIT
