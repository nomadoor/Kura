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
| `storage.host_drive` | Optional override for the Windows drive that backs the WSL2 workspace VHDX, written as a drive letter and colon such as `F:`. Kura tries to auto-detect this from the WSL registry first. | unset |

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
| `images.comfyui` | Image for RunPod render sessions; it must carry the `kura_comfy_prepare.py` of this Kura (read the compiled model list), so use a build from the same checkout | the pinned digest |

An override is for deliberately running another image, such as one built while
developing Kura. Prefer a digest; the plan and `kura doctor workspace` warn
about a mutable tag. AI-Toolkit extends a versioned upstream image, while the
Musubi Tuner and sd-scripts images are paired with Kura's adapters; a pinned
digest moves only after compatibility checks.

## Docker

| Key | Purpose | Default |
| --- | --- | --- |
| `docker.hf_cache` | Host directory for the Hugging Face cache of local Docker runs; a relative path is relative to the workspace | `./cache/huggingface` |
| `docker.mounts[]` | Extra host mounts for local Docker runs, each with a `source`, a `target`, and an optional `mode` (`ro` or `rw`); a mount over the Hugging Face cache is refused (use `docker.hf_cache`) | none |
| `docker.min_free_gb` | The reserve, in GiB, Kura keeps free on every store a local Docker run writes to (the workspace, the Hugging Face cache, writable mounts, and where Docker keeps its data when this host can measure it): the launch keeps it after estimated writes, the job runner stops a running local run whose store falls below it (checked about every 30 seconds; the stop and its reason are recorded first), and `kura image build` keeps it where Docker keeps its data (when this host can measure it); `kura doctor disk` warns below it. A run may raise its own reserve with `safety.max_run_disk_gb` in run.yaml; the launch and the watch keep the larger of the two. RunPod downloads and pulls keep the same reserve on the local disk, plus what they write (`runpod.download_min_free_gb` is retired and refused) | `20` |

## Agents

| Key | Purpose | Default |
| --- | --- | --- |
| `agents.view_images` | Whether agents may open user images (datasets, samples, render inputs) in this workspace, for example to write captions or judge samples. Even when true, an agent stops at the first image its own service's policy does not allow (`docs/adr/agents-and-user-images.md`) | `false` |

## Job runner

| Key | Purpose | Default |
| --- | --- | --- |
| `runner.local_slots` | How many local Docker training runs the job runner runs at once; later launches wait in the order they were requested. At least 1 | `1` |

Model downloads for local Docker runs go to the Hugging Face cache, by default
`./cache/huggingface`, which stays outside Git and is reused across runs. To keep
these large files on another drive, name a host directory:

```yaml
docker:
  hf_cache: /mnt/e/hf-cache
```

Kura mounts that directory at `/workspace/cache/huggingface` in every local
container, measures free space on its drive before launch, and counts the models
already in it. `kura cleanup plan` lists the models in a cache outside the workspace
too, and `kura cleanup --models <id>... --yes` removes only the ones named,
since other workspaces may share it; `--models unreferenced` is refused for it, because only this workspace's records are read. Point several workspaces at a cache outside all of them, never into one workspace's own cache, which that workspace treats as its own. AI-Toolkit's own model
downloads go into this cache too: diffusers repositories in `hub/`, single files
in `ai-toolkit-models/`. Two local runs that
fetch the same AI-Toolkit model at the same time can collide in the trainer's
partial-download directories, which a shared cache widens to every workspace
using it; with the default `runner.local_slots: 1` runs do not overlap.

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

Before a local launch, Kura adds one bounded estimate of the run's writes to
`docker.min_free_gb` on each store (`docs/adr/local-disk-protection.md`): the
model files it downloads for sd-scripts and Musubi Tuner, sized from Hugging
Face metadata (cached files count 0), on the Hugging Face cache's store, and
the run's own weights-sized files on the workspace's store, each bounded by
`rank × 64 MiB`, or by the rank times the measured size per rank for a model
a recorded run measured (the most checkpoints at once, the trainer's training
states, and Kura's copies of them, a state counting as five such files). The reserve
therefore stays a margin instead of being consumed by the run, and the job
runner's disk watch stops a run that reaches it anyway.

Musubi automatic downloads store provenance in
`resolved/musubi/model-bundle.lock.yaml`. The `cache/models/` tree is a
Kura-managed convenience layer for container paths and may contain symlinks; the
lock file is the reproducible source of truth for which Hugging Face repo/files
were selected.

sd-scripts downloads and explicit paths are frozen in
`resolved/sd-scripts/model-bundle.lock.yaml`.

A run whose trainer writes a disk cache (Musubi Tuner always caches latents and
text-encoder outputs; sd-scripts with `cache_latents_to_disk` or
`cache_text_encoder_outputs_to_disk`; AI-Toolkit when its pinned baseline caches
latents to disk or `train.cache_text_embeddings` is on, authored or from that baseline) gets a plan
line that the cache is not counted (image caches are a few MB per item). Cache
files stay below the individual run; the shared dataset remains unchanged.

AI-Toolkit downloads its own models, so Kura does not size them. The plan warns
that the download is not counted (often 10–30 GB); locally the disk watch
covers it. `safety.checkpoint_estimate_gb`, `safety.disk_cache_estimate_gb`,
and `safety.model_download_estimate_gb` are retired and refused.

All registered training adapters reject unknown top-level `backend.config`
keys. Use `uv run kura run capabilities <backend>` (or `--json`) to inspect the
always-applicable fields, architecture/mode-conditional fields, and explicitly
unverified escape hatches. A conditional field used with the wrong selector is
rejected before compilation rather than silently ignored. FLUX and
Anima flow-matching controls (`timestep_sampling`, `discrete_flow_shift`, and
`sigmoid_scale`), FLUX `guidance_scale` / `model_prediction_type`, Anima
`qwen_image_vae_2d` / `vae_chunk_size`, and SDXL `unet_lr` /
`text_encoder_lr` (one number, or a list with one per text encoder, as sd-scripts takes it) are declared native fields and appear
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
| `comfyui.endpoint` | Local ComfyUI API endpoint: scheme, host, port, and an optional path, with no user name, password, query, or fragment (they are refused, since the endpoint is kept in run records); `kura render new` starts each render from it, and `kura doctor comfyui` checks it | `http://127.0.0.1:8188` |
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
| `comfyui.input_stage_mode` | How render runs expose a case image to ComfyUI; ComfyUI rejects symlinked `LoadImage` inputs, so this defaults to copying | `copy` |
| `comfyui.input_stage_cleanup` | Whether temporary staged images are removed after render | `remove_after_render` |
| `comfyui.model_registry` | Explicit ComfyUI model name to Hugging Face repo/file mappings for RunPod render | `{}` |
| `comfyui.runpod` | Optional RunPod overrides for ComfyUI render Pods: any `runpod` key except `storage_mode`, laid over `runpod` when a render launches | created by `kura init` |

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
| `runpod.storage_mode` | Remote staging mode. `upload` is the only mode: the Pod waits on SSH, Kura uploads the staged inputs, runs the trainer, and collects the outputs before the Pod stops. `container_disk` and `object_staging` were removed; `kura workspace migrate` drops them. | `upload` |
| `runpod.gpu_type_ids` | Ordered RunPod GPU candidates. The first available candidate is tried first. | `["NVIDIA RTX A5000", "NVIDIA A40"]` |
| `runpod.gpu_count` | Number of GPUs | `1` |
| `runpod.container_disk_gb` | Disposable Pod container disk size | `150` |
| `runpod.workspace_path` | Workspace path inside the Pod | `/workspace` |
| `runpod.cloud_types` | RunPod clouds to try, a list of `COMMUNITY` and `SECURE` in order; omit it for both | `[COMMUNITY, SECURE]` |
| `runpod.gpu_type_priority` | GPU candidate ordering: `custom` uses the listed order; `availability` is accepted only with one GPU in the list (refused when workspace.yaml loads), because Kura tries each candidate in turn and cannot keep RunPod's availability order | `custom` |
| `runpod.data_center_ids` | Ordered RunPod data-center candidates; Kura tries each configured data center in order | unset |
| `runpod.data_center_priority` | Data-center ordering: `custom` uses the listed order; `availability` is accepted only with one data center in the list (refused when workspace.yaml loads), because Kura tries each candidate in turn and cannot keep RunPod's availability order | unset |
| `runpod.country_codes` | Ordered RunPod country candidates; Kura tries each configured country within each data-center attempt | unset |
| `runpod.ports` | Ports the Pod opens, each `<port>/http` or `<port>/tcp`. Kura reaches every Pod over SSH, so keep `22/tcp` | the image's own: `8675/http` and `22/tcp` for AI-Toolkit, `22/tcp` for the other trainers and ComfyUI |
| `runpod.backend_ports` | Ports for one image, by image name (`ai-toolkit`, `musubi-tuner`, `sd-scripts`, `comfyui`); they replace `runpod.ports` for that image | unset |
| `runpod.interruptible` | Only `false` is accepted: Kura creates on-demand Pods, since an interrupted Pod would end a run before its outputs are collected; `true` is refused when workspace.yaml loads | `false` |

`--max-lease` and `--unattended-wait` are not `workspace.yaml` keys. They are
`kura run execute` flags; see [commands.md](commands.md).

If a run needs a specific GPU, set `compute.gpu` in that run. Kura will use that
GPU before the workspace-level candidates.

RunPod capacity behavior belongs to the run intent, not `workspace.yaml`:

```yaml
compute:
  executor: runpod
  gpu: [NVIDIA RTX A5000]   # a list of RunPod GPU type IDs; only on RunPod runs
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
launch, Kura shows the same bounded estimate (remote model downloads, the input
transfer, and the run's own weights-sized files) against
`runpod.container_disk_gb`, and refuses only when the downloads and the
transfer, which it measures, alone do not fit. A full Pod disk fails the run;
the maximum lease bounds the bill.

## Useful checks

```sh
uv run kura doctor workspace
uv run kura doctor docker
uv run kura doctor sd-scripts
uv run kura doctor comfyui
uv run kura doctor runpod
```
