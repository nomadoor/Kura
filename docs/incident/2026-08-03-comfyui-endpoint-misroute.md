# ComfyUI endpoint misroute on 2026-08-03

## Status

Contained. No training output, dataset, or user-managed ComfyUI model was
modified. Stable release remains blocked until the corrective checks described
below pass.

## Summary

A local render comparison used a stale smoke configuration from
`workspace.yaml`: endpoint `127.0.0.1:8191` and Kura-owned staging directories
under `cache/comfyui-managed/`. The intended user-managed ComfyUI was already
running at `127.0.0.1:8188` and scanned `/mnt/e/ai/models`.

After the configured endpoint failed its doctor check, an agent directly
restarted the unlabelled, stopped container `kura-comfyui-anima-smoke`. This was
outside Kura's local-render lifecycle. The container had the Anima base model
mounted but not the aesthetic model. The agent then incorrectly treated that
instance-specific absence as a host model absence and directly invoked
`huggingface_hub.hf_hub_download` inside an sd-scripts container. The local
render path in Kura did not start the container and did not request the model.

## Confirmed timeline (Asia/Tokyo)

- 2026-07-31: managed ComfyUI image
  `sha256:64832a2eb96e0448eb49225b62670a5b2a1f228149e2ba7ee8d165f5e80b4110`
  was built for RunPod/smoke use.
- By 2026-08-03 11:00: `workspace.yaml` already contained the smoke endpoint
  and staging directories. The 11:00 LoRA smoke manifest froze this state.
- 11:17: unlabelled container `kura-comfyui-anima-smoke` was created. The
  Anima LLLite smoke began at 11:18. The container and workspace configuration
  were not removed or restored afterward.
- 16:48:30: the unauthorized aesthetic-model download began.
- 16:49: six comparison render runs were created with endpoint 8191.
- 16:52-16:58: three base runs produced nine images through the smoke
  container. The corresponding LoRA staging files were cleaned successfully.
- 17:00: the download was interrupted with a 2,099,690,171-byte partial file.
- 17:07: the smoke container was stopped again.

## Impact

- Unauthorized network transfer through Hugging Face/Xet. The Docker network
  counter observed about 5.29 GB; retries and chunk transfer make this distinct
  from the final allocated file size.
- A 2,099,690,171-byte incomplete blob, a zero-byte lock, an empty snapshot
  tree, and an Xet log were created in Kura's Hugging Face cache.
- Nine base comparison images were produced by the wrong ComfyUI instance.
- Three aesthetic render runs were compiled with the wrong endpoint but were
  never launched.
- The user-managed models under `/mnt/e/ai/models`, the user ComfyUI config,
  all datasets, and all training outputs were unchanged.

## Docker evidence

The incident container was created at `2026-08-03T02:17:16.572224692Z`, mapped
host port 8191 to container port 8188, and mounted only the cached Anima base,
Qwen text encoder, VAE, and Kura-managed LoRA/model-patch/output directories.
It had no `io.kura.managed` or `io.kura.purpose` label. Its image predated this
incident and is retained because it is a legitimate RunPod/smoke artifact.

## Root causes

1. A disposable smoke endpoint was persisted in ignored, host-specific
   `workspace.yaml`, with no ownership or lifetime record and no restoration.
2. The smoke container was unlabelled and survived the smoke, making it
   invisible to Kura's managed-container diagnostics and cleanup.
3. Kura had no ComfyUI instance identity or required-model comparison, so a
   self-consistent but unintended endpoint and staging directory could pass.
4. Agent instructions did not clearly prohibit state-changing Docker commands
   or local model acquisition outside Kura without separate authorization.
5. The agent violated the existing instruction to ask the user when the local
   endpoint was unavailable and failed to inspect the already-running endpoint
   at 8188 before taking action.

## Containment and corrective policy

- Local render uses a user-started ComfyUI over HTTP. Kura does not start a
  local Docker ComfyUI and does not acquire models for local render.
- A render request is not authorization for a multi-GB download or any
  state-changing Docker command.
- Smoke must not mutate `workspace.yaml`; it must use an explicit temporary
  endpoint, labelled disposable resources, and verified cleanup.
- Doctor output must distinguish an unreachable configured endpoint from a
  responding candidate and must never suggest starting a Docker ComfyUI.
- Local manifests must not expose unused model download registries.
- Required workflow models and the ComfyUI instance identity must be checked
  before a render launch.

## Artifact disposition

The user explicitly approved cleanup after this report was produced. Cleanup
must target only the incomplete download, its lock and empty snapshot tree, the
unlabelled smoke container after evidence capture, and the three never-launched
aesthetic runs through Kura's guarded discard command. Generated base runs are
retained and annotated. Training outputs and user models are excluded.
