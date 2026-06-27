# Kura

[![日本語 README](https://img.shields.io/badge/README-日本語-blue)](README.ja.md)

Kura is a file-first experiment workspace for working with AI agents on LoRA training and image-comparison runs.

You can use it manually from the CLI, but the intended workflow is to prepare a dataset, specify the training parameters, and let an AI agent create, check, run, monitor, and compare the runs.

## What Kura does

Kura is not a trainer by itself. It is a thin experiment-management layer around backends such as AI-Toolkit and Musubi Tuner, designed to make those tools safer, more reproducible, and easier for agents to operate.

- Keep training intent in `run.yaml` so runs can be reviewed and reproduced.
- Run AI-Toolkit / Musubi Tuner through Docker locally or disposable RunPod Pods remotely.
- Download RunPod outputs back into the local workspace before cleanup.
- Monitor active and historical runs with `kura monitor`.
- Use ComfyUI API workflows to generate step/strength comparison images.
- Keep datasets, workflows, promptsets, run records, and outputs separated.
- Keep secrets, model weights, checkpoints, generated images, and training artifacts out of git.

## File-first model

Kura treats files as the source of truth.

| Path | Purpose |
| --- | --- |
| `run.yaml` | Human/agent training intent |
| `resolved/` | Compile-time frozen inputs |
| `realizations/` | Append-only launch/reconcile observations |
| `status.json` | Materialized latest state |
| `outputs/` | Downloaded or collected artifacts |

Dataset payloads, model weights, checkpoints, generated images, RunPod downloads, local machine config, and secrets are not tracked by git.

## Requirements

- Python 3.11+
- `uv`
- Docker Desktop or Docker Engine for local training and image builds
- NVIDIA GPU support for practical local training
- RunPod account/API key for remote training
- Hugging Face API token for gated/private models when the selected backend or model requires it
- ComfyUI running at `http://127.0.0.1:8188` when using render runs

On WSL2, enable Docker Desktop integration for the target distribution and keep Docker's disk image on a drive with enough free space for caches and temporary run files.

## Working with an AI agent

The usual flow is:

1. Put the dataset under `datasets/`.
2. Tell the agent the goal.  
   Example: `I want to train a character LoRA with this dataset.`
3. Tell the agent the parameters.  
   Example: `rank 16, alpha 16, lr 5e-5, batch 2, 768px, 1500 steps, save every 100 steps.`
4. Ask the agent to create a local smoke run.
5. If the smoke passes, run locally or on RunPod.
6. Watch progress with `kura monitor`.
7. Pull intermediate checkpoints and generate ComfyUI comparison images.
8. Stop, continue, or extend training based on the comparison.

## Kura monitor

`kura monitor` is a TUI for observing training and render runs.

It shows active runs, history, loss, progress, GPU/RunPod information, and output paths. It is read-only: it does not start or stop training.

```sh
uv run kura monitor
uv run kura run watch <run-id>
```

## Common commands

| Command | Purpose |
| --- | --- |
| `uv sync` | Set up the development environment |
| `uv run kura init` | Initialize a workspace |
| `uv run kura doctor docker` | Check Docker/GPU/cache readiness |
| `uv run kura doctor runpod` | Check RunPod API, Pods, and Network Volumes |
| `uv run kura dataset validate <dataset>` | Validate a dataset manifest |
| `uv run kura run new --experiment <name> --slug <slug>` | Create a train run |
| `uv run kura run compile <run-id>` | Freeze `run.yaml` into resolved inputs |
| `uv run kura run launch <run-id> --executor docker --dry-run` | Preview a local Docker launch |
| `uv run kura run launch <run-id> --executor docker` | Run locally through Docker |
| `uv run kura run remote <run-id>` | Run on RunPod and keep a short review window after download |
| `uv run kura run pull <run-id> --step <step>` | Pull an intermediate checkpoint from a running RunPod run |
| `uv run kura run stop <run-id>` | Stop the associated Pod/container |
| `uv run kura run reconcile <run-id>` | Refresh observed external state |
| `uv run kura run prune --dry-run` | Preview cleanup of old runs |
| `uv run kura monitor` | Open the run monitor TUI |
| `uv run kura run watch <run-id>` | Watch one run in the TUI |
| `uv run kura render new --slug <slug>` | Create a ComfyUI render run |
| `uv run kura render compile <run-id>` | Freeze workflow and promptset inputs |
| `uv run kura render launch <run-id>` | Generate images through ComfyUI |

## RunPod safety

Kura's RunPod path is designed around disposable Pods.

1. Compile the run locally.
2. Upload only the required run inputs.
3. Train on the Pod.
4. Download outputs and logs back to the local workspace.
5. Keep the Pod briefly for review.
6. Stop it automatically.

By default, Kura does not use RunPod Network Volumes. This keeps GPU placement flexible and avoids leaving persistent storage on RunPod.

After a confirmed download, `kura run remote` keeps the Pod for `--hold-for 30m` by default so you can inspect the LoRA and decide whether to stop or continue. A Pod-side `--max-lease 12h` guard is installed as a last billing fuse if the local controller dies.

## Runtime images

Kura does not bake model weights or secrets into images.

| Backend | Default image behavior |
| --- | --- |
| AI-Toolkit | Uses the official `ostris/aitoolkit:latest` image/template on RunPod. A local `kura/ai-toolkit:dev` image can be built when needed. |
| Musubi Tuner | Defaults to `nomadoor/kura-musubi-tuner:dev` for remote runs and `kura/musubi-tuner:dev` locally. Replace these in `workspace.yaml` if you use your own image. |

Image names are set in `workspace.yaml`. Build only when needed.

```sh
uv run kura image build ai-toolkit --ref <ref>
uv run kura image build musubi-tuner --ref <ref>
```

## Notifications

Kura uses desktop notifications when `notify-send` is available. If `KURA_NTFY_TOPIC` is set in `.env.local`, it can also send ntfy notifications.

```env
KURA_NTFY_TOPIC=long-random-topic-name
```

Use a long, unguessable topic and treat it like a secret.

## Agent documentation

Always-loaded agent rules live in [AGENTS.md](AGENTS.md). Task-specific instructions live under `.claude/skills/`. When using an agent, have it read `AGENTS.md` first.

## Check

```sh
uv run python scripts/check_release.py
```

## More documentation

- [README.ja.md](README.ja.md): Japanese overview
- [docs/smoke-test.md](docs/smoke-test.md): smoke test notes

## License

MIT
