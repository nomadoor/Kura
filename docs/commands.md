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
| `uv run kura doctor comfyui --endpoint http://127.0.0.1:8189` | Check a specific ComfyUI endpoint against the configured LoRA staging directory |
| `uv run kura doctor comfyui --endpoint http://127.0.0.1:8189 --probe-stage` | Temporarily stage a probe LoRA and verify the endpoint can see `comfyui.lora_dir` |
| `uv run kura doctor workspace` | Show which Kura workspace this command sees |

## Datasets

| Command | Purpose |
| --- | --- |
| `uv run kura dataset validate <dataset>` | Validate a dataset manifest |
| `uv run kura dataset inspect <dataset-id-or-path>` | Measure dataset facts without pass/fail judgment |
| `uv run kura dataset inspect <dataset-id-or-path> --json` | Print dataset facts for an agent to read |

## Training runs: normal workflow

| Command | Purpose |
| --- | --- |
| `uv run kura run new --experiment <name> --slug <slug> [--backend ai-toolkit\|musubi-tuner] [--executor docker\|runpod] [--gpu <name>]` | Create a train run |
| `uv run kura run plan <run-id>` | Show training settings, Resources facts, model download estimates, and warnings that will be launched |
| `uv run kura run execute <run-id>` | Execute through the Docker or RunPod executor frozen in the compiled run; waits through completion and normal finalization |
| `uv run kura run discard <run-id>` | Preview deletion of a draft or unlaunched compiled run (add `--yes` to delete) |
| `uv run kura run prune` | Preview cleanup of old runs (add `--yes` to delete) |
| `uv run kura run prune --docker-containers --docker-volumes` | Also clean up Kura-managed stopped containers/volumes (add `--yes` to delete) |

Compile after editing `run.yaml`, review `run plan`, obtain the single launch
approval, then use `run execute`. The agent normally performs compile for the
user; it is listed below as a low-level command for inspection and development.

## Diagnosis and recovery

Use these only when a normal execution was interrupted or needs inspection.
They remain separate because stopping or forcing a download is a
situation-dependent decision, not a safe universal `recover` action.

| Command | Purpose |
| --- | --- |
| `uv run kura doctor docker` | Diagnose the local Docker/GPU execution environment |
| `uv run kura doctor runpod` | Diagnose RunPod API access and remaining resources |
| `uv run kura run reconcile <run-id>` | Refresh observed Pod/container state without changing it |
| `uv run kura run pull <run-id> --step <step>` | Recover an intermediate checkpoint from a running RunPod run |
| `uv run kura run download <run-id> --force` | Retry downloading a RunPod snapshot after inspecting remote state |
| `uv run kura run stop <run-id>` | Explicitly stop the associated Pod/container |

## Low-level execution commands

These are retained for diagnosis, recovery, and Kura development. They are not
additional steps in the normal workflow.

| Command | Purpose |
| --- | --- |
| `uv run kura run compile <run-id>` | Freeze `run.yaml` into resolved inputs |
| `uv run kura run launch <run-id> --executor docker --dry-run` | Preview a local Docker launch |
| `uv run kura run launch <run-id> --executor docker --wait` | Launch locally and wait in the foreground |
| `uv run kura run stage <run-id>` | Build the transfer bundle for a remote executor |
| `uv run kura run upload <run-id>` | Upload a staged bundle to an existing RunPod Pod |
| `uv run kura run remote <run-id>` | Invoke the RunPod lifecycle directly with advanced flags |

Useful low-level `run remote` flags:

- `--hold-for 30m` keeps a completed Pod briefly after confirmed download so you
  can inspect results. Use `--hold-for 0` to stop immediately.
- `--max-lease 12h` is a best-effort Pod-side billing fuse if the local
  controller dies.

## Monitoring

| Command | Purpose |
| --- | --- |
| `uv run kura monitor` | Open the run monitor TUI |
| `uv run kura monitor --all` | Include draft runs in the monitor |
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
