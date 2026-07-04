# Command reference

The commands you'll reach for most. In normal use you tell an AI agent what you
want and it runs these for you; this page is for when you want to look one up.

This is a curated subset. Run `uv run kura --help` (or `uv run kura <command> --help`)
for the complete, authoritative, up-to-date list of commands and options.

## Setup

| Command | Purpose |
| --- | --- |
| `uv sync` | Install Kura and its dependencies into `.venv` |
| `uv run kura init` | Create the workspace folders and default config |
| `uv run kura cleanup all` | Preview local cache, run, and Docker cleanup targets |
| `uv run kura cleanup cache --yes` | Delete Kura-managed local model/cache data after previewing it |
| `uv run kura fix-permissions` | Preview root-owned Kura cache/run files that can block cleanup |
| `uv run kura fix-links` | Preview repair for Kura symlinks with container-private targets |
| `uv run kura --version` | Print the installed Kura version |
| `uv run kura doctor docker` | Check Docker / GPU / cache readiness |
| `uv run kura doctor disk` | Report local disk, cache, Docker storage, and permission risks |
| `uv run kura doctor musubi` | Smoke-test Musubi adapter scripts in the configured image |
| `uv run kura doctor runpod` | Check RunPod API, Pods, and Network Volumes |
| `uv run kura doctor comfyui` | Check local ComfyUI endpoint and LoRA staging config |
| `uv run kura doctor workspace` | Show which Kura workspace this command sees |

## Datasets

| Command | Purpose |
| --- | --- |
| `uv run kura dataset validate <dataset>` | Validate a dataset manifest |

## Training runs

| Command | Purpose |
| --- | --- |
| `uv run kura run new --experiment <name> --slug <slug> [--backend ai-toolkit\|musubi-tuner] [--executor docker\|runpod] [--gpu <name>]` | Create a train run |
| `uv run kura run plan <run-id>` | Show training settings, Resources facts, model download estimates, and warnings that will be launched |
| `uv run kura run compile <run-id>` | Freeze `run.yaml` into resolved inputs |
| `uv run kura run launch <run-id> --executor docker --dry-run` | Preview a local Docker launch |
| `uv run kura run launch <run-id> --executor docker` | Run locally through Docker |
| `uv run kura run launch <run-id> --executor docker --wait` | Run locally and wait in the foreground until it finishes (auto-reconciles) |
| `uv run kura run remote <run-id>` | Run on RunPod, download outputs, then auto-stop |
| `uv run kura run pull <run-id> --step <step>` | Pull an intermediate checkpoint from a running RunPod run |
| `uv run kura run stop <run-id>` | Stop the associated Pod/container |
| `uv run kura run reconcile <run-id>` | Refresh observed external state |
| `uv run kura run prune` | Preview cleanup of old runs (add `--yes` to delete) |
| `uv run kura run prune --docker-containers --docker-volumes` | Also clean up Kura-managed stopped containers/volumes (add `--yes` to delete) |

Useful `run remote` flags:

- `--hold-for 30m` keeps a completed Pod briefly after confirmed download so you
  can inspect results. Use `--hold-for 0` to stop immediately.
- `--max-lease 12h` is a best-effort Pod-side billing fuse if the local
  controller dies.

## Monitoring

| Command | Purpose |
| --- | --- |
| `uv run kura monitor` | Open the run monitor TUI |
| `uv run kura run watch <run-id>` | Watch one run in the TUI |

## Render (ComfyUI comparison images)

| Command | Purpose |
| --- | --- |
| `uv run kura render new --slug <slug>` | Create a ComfyUI render run |
| `uv run kura render compile <run-id>` | Freeze workflow and promptset inputs |
| `uv run kura render launch <run-id>` | Generate images through ComfyUI |
| `uv run kura render launch <run-id> --executor runpod` | Generate images through a disposable RunPod ComfyUI Pod |

## Images

Image names are set in `workspace.yaml`. Build only when needed.

| Command | Purpose |
| --- | --- |
| `uv run kura image build ai-toolkit --ref <ref>` | Build the AI-Toolkit image |
| `uv run kura image build musubi-tuner --ref <ref>` | Build the Musubi Tuner image |
| `uv run kura image build comfyui --ref <ref>` | Build the ComfyUI render image |
