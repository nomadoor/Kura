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
reading the source; its `settings` field lists every section and key, and
`uninterpreted_subtrees` lists the places whose inner names are your data rather
than Kura vocabulary (`docker.images`, `comfyui.model_registry`, and the like).

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

Image references are Kura-managed defaults: images are pulled automatically
when needed, move together with Kura releases, and users normally never build
or change them. Overriding them (e.g. pointing at your own registry after
`kura image build` / `kura image publish`) is an escape hatch for developing
Kura itself, not part of normal use. Trainer freshness works differently per
backend by design. AI-Toolkit extends a versioned upstream official image,
while the Musubi Tuner and sd-scripts images are paired with Kura's adapters. All
defaults move only after compatibility checks; a mutable upstream tag is not a
reproducible run contract.

| Key | Purpose | Default |
| --- | --- | --- |
| `docker.images.ai-toolkit.local` | Local Docker image used for AI-Toolkit runs | `nomadoor/kura-ai-toolkit:dev` |
| `docker.images.ai-toolkit.remote` | Image name used when publishing your own AI-Toolkit image | `nomadoor/kura-ai-toolkit:dev` |
| `docker.images.musubi-tuner.local` | Local Docker image used for Musubi Tuner runs | `nomadoor/kura-musubi-tuner:dev` |
| `docker.images.musubi-tuner.remote` | Image name used for RunPod when not using the default image override | `nomadoor/kura-musubi-tuner:dev` |
| `docker.images.sd-scripts.local` | Local Docker image used for sd-scripts runs | `nomadoor/kura-sd-scripts:dev` |
| `docker.images.sd-scripts.remote` | Image name used when publishing your own sd-scripts image | `nomadoor/kura-sd-scripts:dev` |
| `docker.workspace_target` | Container path for the mounted workspace. Kura currently supports only `/workspace`; other values are rejected at launch because backend artifacts compile `/workspace/...` paths. | `/workspace` |
| `docker.gpu` | Add `--gpus all` for local Docker training | `true` |
| `docker.mounts[]` | Extra host mounts for local Docker runs | HF cache mount |
| `docker.min_free_gb` | Minimum free space Kura keeps after estimated local writes before Docker launch | `100` |
| `docker.build_cache_limit_gb` | Docker build cache limit checked before local Docker launch | `30` |

Default Hugging Face cache mount:

```yaml
docker:
  mounts:
    - source: ./cache/huggingface
      target: /workspace/cache/huggingface
      mode: rw
```

`./cache/huggingface` stays outside Git and is reused across local Docker runs
inside the same workspace. Advanced users can point `source` at a shared absolute
path. Kura also maps the legacy `/root/.cache/huggingface` target into this
workspace path so existing workspaces do not keep creating root-owned files.

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
by the built-in selector; adapter-owned flags cannot be duplicated there.

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
| `comfyui.model_patches_dir` | Host path to ComfyUI `models/model_patches`; required and non-empty when a render workflow declares a `model_patch` patch | `""` |
| `comfyui.model_patch_stage_subdir` | Temporary subdirectory under `model_patches_dir` | `Kura_tmp` |
| `comfyui.model_patch_stage_mode` | How render runs expose a local model patch to ComfyUI | `symlink` |
| `comfyui.model_patch_stage_cleanup` | Whether temporary staged model patches are removed after render | `remove_after_render` |
| `comfyui.input_dir` | Host path to ComfyUI `input`; required and non-empty when a render promptset supplies images through a `type: image` patch binding | `""` |
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
| `runpod.default_image.ai-toolkit` | Default AI-Toolkit remote image/template image | `ostris/aitoolkit:0.10.22` |
| `runpod.default_image.musubi-tuner` | Default Musubi remote image | `nomadoor/kura-musubi-tuner:dev` |
| `runpod.default_image.sd-scripts` | Default sd-scripts remote image | `nomadoor/kura-sd-scripts:dev` |
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

RunPod capacity behavior belongs to the run intent, not `workspace.yaml`:

```yaml
compute:
  executor: runpod
  gpu: NVIDIA RTX A5000
  capacity:
    mode: immediate  # or wait
    timeout: 6h      # required only for wait; defaults to 24h
    poll_interval: 30s
```

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
