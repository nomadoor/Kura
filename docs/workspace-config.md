# Workspace config reference

`workspace.yaml` is local workspace configuration. It is ignored by Git and is
created by `kura init`. Relative host paths are resolved from the workspace root.

This page is intentionally short: it is mostly for AI agents that need to adjust
runtime configuration without guessing.

## Storage

| Key | Purpose | Default |
| --- | --- | --- |
| `storage.host_drive` | Optional override for the Windows drive that backs the WSL2 workspace VHDX, for example `F:`. Kura tries to auto-detect this from the WSL registry first. | `""` |
| `storage.docker_data_drive` | Optional override for the Windows drive that backs Docker Desktop data, if different from `storage.host_drive`. Reserved for Docker backing accounting. | `""` |

On native Linux and macOS, Kura trusts normal filesystem free space. On WSL2,
large local Docker launches need the Windows backing drive as well as the Linux
filesystem. Kura auto-detects the current distro's backing drive when Windows
interop is available; use `storage.host_drive` only when that detection is
wrong or unavailable.

## Docker

| Key | Purpose | Default |
| --- | --- | --- |
| `docker.images.ai-toolkit.local` | Local Docker image used for AI-Toolkit runs | `nomadoor/kura-ai-toolkit:dev` |
| `docker.images.ai-toolkit.remote` | Image name used when publishing your own AI-Toolkit image | `nomadoor/kura-ai-toolkit:dev` |
| `docker.images.musubi-tuner.local` | Local Docker image used for Musubi Tuner runs | `nomadoor/kura-musubi-tuner:dev` |
| `docker.images.musubi-tuner.remote` | Image name used for RunPod when not using the default image override | `nomadoor/kura-musubi-tuner:dev` |
| `docker.workspace_target` | Container path for the mounted workspace | `/workspace` |
| `docker.gpu` | Add `--gpus all` for local Docker training | `true` |
| `docker.mounts[]` | Extra host mounts for local Docker runs | HF cache mount |
| `docker.min_free_gb` | Minimum free space Kura keeps after estimated local writes before Docker launch | `100` |
| `docker.build_cache_limit_gb` | Docker build cache limit checked before local Docker launch | `30` |

Default Hugging Face cache mount:

```yaml
docker:
  mounts:
    - source: ./cache/huggingface
      target: /root/.cache/huggingface
      mode: rw
```

`./cache/huggingface` stays outside Git and is reused across local Docker runs
inside the same workspace. Advanced users can point `source` at a shared absolute
path.

For Musubi runs with automatic Hugging Face downloads, Kura tries to estimate
the referenced file sizes before local launch. The estimate is added on top of
`docker.min_free_gb`, so the configured value remains a safety margin instead
of being consumed by the download.

Musubi automatic downloads store provenance in
`resolved/musubi/model-bundle.lock.yaml`. The `cache/models/` tree is a
Kura-managed convenience layer for container paths and may contain symlinks; the
lock file is the reproducible source of truth for which Hugging Face repo/files
were selected.

Path namespace depends on the consumer. Container command specs may use
`/workspace/...`, but host-consumed workspace artifacts should be
workspace-relative or host-resolvable. `kura doctor disk` reports Kura symlinks
that point at container-private paths such as `/root/...`; `kura fix-links`
previews and can repair links whose targets are covered by the effective
workspace mount table.

## ComfyUI

| Key | Purpose | Default |
| --- | --- | --- |
| `comfyui.endpoint` | Local ComfyUI API endpoint | `http://127.0.0.1:8188` |
| `comfyui.lora_dir` | Host path to ComfyUI `models/loras`; empty means no automatic LoRA staging | `""` |
| `comfyui.lora_stage_subdir` | Temporary subdirectory under `lora_dir` | `Kura_tmp` |
| `comfyui.lora_stage_mode` | How render runs expose a local LoRA to ComfyUI | `symlink` |
| `comfyui.lora_stage_cleanup` | Whether temporary staged LoRAs are removed after render | `remove_after_render` |
| `comfyui.model_registry` | Explicit ComfyUI model name to Hugging Face repo/file mappings for RunPod render | `{}` |
| `comfyui.runpod` | Optional RunPod overrides for ComfyUI render Pods | created by `kura init` |

If `comfyui.lora_dir` is changed after a render run was compiled, re-run:

```sh
uv run kura render compile <run-id>
```

Render compile freezes these settings into `resolved/manifest.lock.yaml`.

## RunPod

| Key | Purpose | Default |
| --- | --- | --- |
| `runpod.default_image.ai-toolkit` | Default AI-Toolkit remote image/template image | `ostris/aitoolkit:latest` |
| `runpod.default_image.musubi-tuner` | Default Musubi remote image | `nomadoor/kura-musubi-tuner:dev` |
| `runpod.default_image.comfyui` | Default ComfyUI remote render image | `nomadoor/kura-comfyui:dev` |
| `runpod.template_id` | Optional RunPod template ID; used for AI-Toolkit-compatible official template startup | `0fqzfjy6f3` |
| `runpod.api_key_env` | Environment variable that holds the RunPod API key | `RUNPOD_API_KEY` |
| `runpod.storage_mode` | Remote staging mode | `upload` |
| `runpod.gpu_type_ids` | Ordered RunPod GPU candidates. The first available candidate is tried first. | `["NVIDIA RTX A5000", "NVIDIA A40"]` |
| `runpod.gpu_count` | Number of GPUs | `1` |
| `runpod.container_disk_gb` | Disposable Pod container disk size | `150` |
| `runpod.download_min_free_gb` | Minimum local free space required before RunPod download | `50` |
| `runpod.volume_in_gb` | Network Volume size; Kura defaults to none | `0` |
| `runpod.workspace_path` | Workspace path inside the Pod | `/workspace` |
| `runpod.cloud_type` / `runpod.cloud_types` | RunPod cloud preference; `ANY` tries community then secure | `ANY` |
| `runpod.gpu_type_priority` | RunPod GPU selection priority | `custom` |
| `runpod.interruptible` | Whether to allow interruptible Pods | `false` |

`--hold-for` and `--max-lease` are not `workspace.yaml` keys. They are
`kura run remote` flags; see [commands.md](commands.md).

If a run needs a specific GPU, set `compute.gpu` in that run. Kura will use that
GPU before the workspace-level candidates.

Training RunPod Pods are disposable. In `upload` mode, local model caches are
not uploaded with the run bundle, so `kura run plan` reports model downloads as
remote writes for RunPod even when the same files are cached locally. Before
launch, Kura compares estimated remote model downloads plus the configured
checkpoint estimate against `runpod.container_disk_gb`.

## Useful checks

```sh
uv run kura doctor workspace
uv run kura doctor docker
uv run kura doctor comfyui
uv run kura doctor runpod
```
