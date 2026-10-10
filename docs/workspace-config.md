# Workspace config reference

`workspace.yaml` is local workspace configuration. It is ignored by Git and is
created by `kura init`. Relative host paths are resolved from the workspace root.

This page is intentionally short: it is mostly for AI agents that need to adjust
runtime configuration without guessing.

The file is a closed contract. Every section and key below is one Kura reads, and
anything else is refused when the workspace is loaded — a misspelled
`comfyui.input_stage_mod` would otherwise leave the staging mode on its default
while the file recorded the intended one. Settings an older Kura wrote but no
longer reads are reported as obsolete and should be deleted rather than
corrected. Run `kura doctor workspace` to print the accepted settings instead of
reading the source; its recursive `settings` field lists fixed names and marks
dynamic names as `<name>`. A dynamic name such as `comfyui.model_registry.<name>`
is your vocabulary, but the fields beneath it are still checked because Kura
reads them.

The file carries `schema_version: 2`. A workspace written by an older Kura is
refused with the command that migrates it: `kura workspace migrate` shows the
change, applies it only when confirmed, and keeps the previous file as
`workspace.yaml.<timestamp>.bak`.

## Storage

| Key | Purpose | Default |
| --- | --- | --- |
| `storage.host_drive` | Optional override for the Windows drive that backs the WSL2 workspace VHDX, for example `F:`. Kura tries to auto-detect this from the WSL registry first. | `""` |

On native Linux and macOS, Kura trusts normal filesystem free space. On WSL2,
large local Docker launches need the Windows backing drive as well as the Linux
filesystem. Kura auto-detects the current distro's backing drive when Windows
interop is available; use `storage.host_drive` only when that detection is
wrong or unavailable.

## Images

Kura pins one image per backend by digest in its own code: the digest each
backend's smoke evidence ran with. Local and RunPod runs use the same image,
and Docker pulls it the first time a run needs it; users never build images.
The plan names the image a run uses and says whether it is pinned or
overridden.

| Key | Purpose | Default |
| --- | --- | --- |
| `images.ai-toolkit` | Image for AI-Toolkit runs, locally and on RunPod | the pinned digest |
| `images.musubi-tuner` | Image for Musubi Tuner runs | the pinned digest |
| `images.sd-scripts` | Image for sd-scripts runs | the pinned digest |
| `images.comfyui` | Image for RunPod render sessions | the pinned digest |

An override is for deliberately running another image, such as one built while
developing Kura. Prefer a digest; the plan and `kura doctor workspace` warn
about a mutable tag. AI-Toolkit extends a versioned upstream image, while the
Musubi Tuner and sd-scripts images are paired with Kura's adapters; a pinned
digest moves only after compatibility checks.

## Docker

| Key | Purpose | Default |
| --- | --- | --- |
| `docker.hf_cache` | Host directory for the Hugging Face cache of local Docker runs; a relative path is relative to the workspace | `./cache/huggingface` |
| `docker.mounts[]` | Extra host mounts for local Docker runs; a mount over the Hugging Face cache is refused (use `docker.hf_cache`) | none |
| `docker.min_free_gb` | Minimum free space Kura keeps after estimated local writes before Docker launch; `kura doctor disk` warns below it | `100` |
| `docker.build_cache_limit_gb` | Docker build cache size above which `kura doctor disk` warns and `kura image build` stops (unless `--allow-large-build-cache`); it does not stop a launch. Outside a workspace, `kura image build` uses the default | `30` |

## Agents

| Key | Purpose | Default |
| --- | --- | --- |
| `agents.view_images` | Whether agents may open user images (datasets, samples, render inputs) in this workspace, for example to write captions or judge samples. Even when true, an agent stops at the first image its own service's policy does not allow (`docs/adr/agents-and-user-images.md`) | `false` |

## Job runner

| Key | Purpose | Default |
| --- | --- | --- |
| `runner.local_slots` | How many local Docker training runs the job runner runs at once; later launches wait in the order they were requested. Values below 1 count as 1 | `1` |

Model downloads for local Docker runs go to the Hugging Face cache, by default
`./cache/huggingface`, which stays outside Git and is reused across runs. To keep
these large files on another drive, name a host directory:

```yaml
docker:
  hf_cache: /mnt/e/hf-cache
```

Kura mounts that directory at `/workspace/cache/huggingface` in every local
container, measures free space on its drive before launch, and counts the models
already in it. `kura cleanup cache` reports a cache outside the workspace but
never deletes it, since other workspaces may share it.

Local Docker training always requests the GPU (`--gpus all`) and sees the
workspace paths it uses under `/workspace` (only the selected ones for a
manifest-v2 dataset); the RunPod API key is always read from
`RUNPOD_API_KEY`. Older workspaces that still set `docker.gpu`,
`docker.workspace_target`, `storage.docker_data_drive`, `runpod.api_key_env`, or
`comfyui.runpod.api_key_env` are refused with a pointer to `kura workspace migrate`, which drops them.

Kura starts every RunPod Pod from the configured image and its own start
script, which arms the maximum lease before anything else runs. RunPod
templates are not used, so `runpod.template_id` is refused the same way and
`kura workspace migrate` drops it.

Workspaces created before `docker.hf_cache` mounted the cache through
`docker.mounts` (target `/workspace/cache/huggingface` or the older
`/root/.cache/huggingface`). Kura now refuses that entry; `kura workspace migrate`
moves it to `docker.hf_cache`, or removes it when it named the default location.
Links that containers wrote through the older target still resolve.

Executors set `HF_HOME=/workspace/cache/huggingface` and
`HF_HUB_CACHE=/workspace/cache/huggingface/hub`. AI-Toolkit, Kura-managed
Musubi downloads, and remote ComfyUI preparation therefore reuse the same
repository snapshot and blob namespace.

For Musubi runs with automatic Hugging Face downloads, Kura tries to estimate
the referenced file sizes before local launch. The estimate is added on top of
`docker.min_free_gb`, so the configured value remains a safety margin instead
of being consumed by the download.

Musubi automatic downloads store provenance in
`resolved/musubi/model-bundle.lock.yaml`. The `cache/models/` tree is a
Kura-managed convenience layer for container paths and may contain symlinks; the
lock file is the reproducible source of truth for which Hugging Face repo/files
were selected.

sd-scripts downloads and explicit paths are frozen in
`resolved/sd-scripts/model-bundle.lock.yaml`. If
`cache_latents_to_disk` or `cache_text_encoder_outputs_to_disk` is enabled,
set `backend.config.disk_cache_estimate_gb` to a measured positive estimate.
An unknown estimate blocks launch unless the reviewed run explicitly records
`safety.allow_unknown_disk_cache: true`. Cache files stay below the individual
run; the shared dataset remains unchanged.

All registered training adapters reject unknown top-level `backend.config`
keys. Use `uv run kura run capabilities <backend>` (or `--json`) to inspect the
always-applicable fields, architecture/mode-conditional fields, and explicitly
unverified escape hatches. A conditional field used with the wrong selector is
rejected before compilation rather than silently ignored. FLUX and
Anima flow-matching controls (`timestep_sampling`, `discrete_flow_shift`, and
`sigmoid_scale`), FLUX `guidance_scale` / `model_prediction_type`, Anima
`qwen_image_vae_2d` / `vae_chunk_size`, and SDXL `unet_lr` /
`text_encoder_lr1` / `text_encoder_lr2` are validated native fields and appear
in the run plan. Use `extra_args` only for an audited upstream option not owned
by Kura. Exact and argparse-abbreviated spellings of adapter-owned flags are
rejected by the built-in selector; adapter-owned flags cannot be duplicated
there.

Path namespace depends on the consumer. Container command specs may use
`/workspace/...`, but host-consumed workspace artifacts should be
workspace-relative or host-resolvable. `kura doctor disk` reports Kura symlinks
that point at container-private paths such as `/root/...`; `kura fix-links`
previews and can repair links whose targets are covered by the effective
workspace mount table.

## ComfyUI

| Key | Purpose | Default |
| --- | --- | --- |
| `comfyui.endpoint` | Local ComfyUI API endpoint; `kura render new` starts each render from it, and `kura doctor comfyui` checks it | `http://127.0.0.1:8188` |
| `comfyui.lora_dir` | Host path to ComfyUI `models/loras`; empty means no automatic LoRA staging | `""` |
| `comfyui.lora_stage_subdir` | Temporary subdirectory under `lora_dir` | `Kura_tmp` |
| `comfyui.lora_stage_mode` | How render runs expose a local LoRA to ComfyUI: `auto` links when the endpoint's `/system_stats` reports Linux or WSL, and copies when it reports Windows, does not answer, or Kura itself runs on Windows; `symlink` or `copy` forces one | `auto` |
| `comfyui.lora_stage_cleanup` | Whether temporary staged LoRAs are removed after render | `remove_after_render` |
| `comfyui.model_patches_dir` | Host path to ComfyUI `models/model_patches`; required and non-empty when a render workflow declares a `model_patch` patch | `""` |
| `comfyui.model_patch_stage_subdir` | Temporary subdirectory under `model_patches_dir` | `Kura_tmp` |
| `comfyui.model_patch_stage_mode` | How render runs expose a local model patch to ComfyUI; same choices as `lora_stage_mode` | `auto` |
| `comfyui.model_patch_stage_cleanup` | Whether temporary staged model patches are removed after render | `remove_after_render` |
| `comfyui.input_dir` | Host path to ComfyUI `input`; required and non-empty for local renders with a `type: image` patch binding. RunPod uses its managed Pod input directory instead | `""` |
| `comfyui.input_stage_subdir` | Temporary subdirectory under `input_dir` | `Kura_tmp` |
| `comfyui.input_stage_mode` | How render runs expose a promptset image to ComfyUI; ComfyUI rejects symlinked `LoadImage` inputs, so this defaults to copying | `copy` |
| `comfyui.input_stage_cleanup` | Whether temporary staged images are removed after render | `remove_after_render` |
| `comfyui.model_registry` | Explicit ComfyUI model name to Hugging Face repo/file mappings for RunPod render | `{}` |
| `comfyui.runpod` | Optional RunPod overrides for ComfyUI render Pods | created by `kura init` |

The local executor treats `comfyui.endpoint` as an external, user-managed
service. Kura submits HTTP requests to that exact endpoint; it does not start or
restart ComfyUI, start Docker, install ComfyUI, or download missing models.
`lora_dir`, `model_patches_dir`, and `input_dir` must be directories scanned by that same
instance and should normally live outside the Kura workspace. Use
`kura doctor comfyui --workflow <api-workflow.json>` to verify the endpoint and
the workflow's required models before launch.

`comfyui.model_registry` and `comfyui.runpod` are RunPod-only configuration.
They are not frozen into local render manifests and cannot authorize local
downloads. An unreachable local endpoint or a missing model is a stop-and-ask
condition, not permission to create a replacement service. Any dedicated smoke
instance requires separate approval, isolated model paths, explicit ownership,
and teardown without changing the normal workspace endpoint.

If `comfyui.lora_dir` is changed after a render run was compiled, re-run:

```sh
uv run kura render compile <run-id>
```

Render compile freezes these settings into `resolved/manifest.lock.yaml`.

## RunPod

| Key | Purpose | Default |
| --- | --- | --- |
| `runpod.storage_mode` | Remote staging mode | `upload` |
| `runpod.gpu_type_ids` | Ordered RunPod GPU candidates. The first available candidate is tried first. | `["NVIDIA RTX A5000", "NVIDIA A40"]` |
| `runpod.gpu_count` | Number of GPUs | `1` |
| `runpod.container_disk_gb` | Disposable Pod container disk size | `150` |
| `runpod.download_min_free_gb` | Minimum local free space required before RunPod download | `50` |
| `runpod.volume_in_gb` | Network Volume size; Kura defaults to none | `0` |
| `runpod.workspace_path` | Workspace path inside the Pod | `/workspace` |
| `runpod.cloud_type` / `runpod.cloud_types` | RunPod cloud preference; `ANY` tries community then secure | `ANY` |
| `runpod.gpu_type_priority` | GPU candidate ordering: `custom` uses the listed order; `availability` is accepted only for one GPU because the GraphQL control plane cannot preserve availability ordering across fallback attempts | `custom` |
| `runpod.data_center_ids` | Ordered RunPod data-center candidates; Kura tries each configured data center in order | unset |
| `runpod.data_center_priority` | Data-center ordering: `custom` uses the listed order; `availability` is accepted only for one data center because the GraphQL control plane cannot preserve availability ordering across fallback attempts | unset |
| `runpod.country_codes` | Ordered RunPod country candidates; Kura tries each configured country within each data-center attempt | unset |
| `runpod.interruptible` | Whether to allow interruptible Pods | `false` |

`--max-lease` and `--unattended-wait` are not `workspace.yaml` keys. They are
`kura run execute` flags; see [commands.md](commands.md).

If a run needs a specific GPU, set `compute.gpu` in that run. Kura will use that
GPU before the workspace-level candidates.

RunPod capacity behavior belongs to the run intent, not `workspace.yaml`:

```yaml
compute:
  executor: runpod
  gpu: NVIDIA RTX A5000
  capacity:          # optional; this is the default
    mode: wait         # or immediate: fail at once when the GPU is taken
    timeout: 24h       # for wait
    poll_interval: 30s
```

RunPod GPUs are often taken, so a run waits for one by default. While it waits
there is no Pod and no billing.

`kura run plan` measures live stock and price before approval. `run execute`
uses the compiled capacity policy without asking again.

Training RunPod Pods are disposable. In `upload` mode, local model caches are
not uploaded with the run bundle, so `kura run plan` reports model downloads as
remote writes for RunPod even when the same files are cached locally. Before
launch, Kura compares estimated remote model downloads plus the configured
checkpoint estimate against `runpod.container_disk_gb`.

## Useful checks

```sh
uv run kura doctor workspace
uv run kura doctor docker
uv run kura doctor sd-scripts
uv run kura doctor comfyui
uv run kura doctor runpod
```
