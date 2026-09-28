# Training workspace path inventory

Status: implemented. Local Docker training no longer binds the whole workspace;
it mounts only the paths below, built by `local_training_mounts` in
`src/kura/dataset_handoff.py`. Real-container evidence is recorded in
[smoke-evidence/2026-09-29-dataset-handoff-final.yaml](smoke-evidence/2026-09-29-dataset-handoff-final.yaml).

The inventory covers literal `/workspace` paths, generated command/config
paths, container helpers, executor environment and mounts, and user-selectable
native paths in the three built-in training backends. New runtime-generated
paths must be added here together with their mount.

| Container path or family | Consumer | Access | Mount |
| --- | --- | --- | --- |
| `/workspace/datasets/<id>/...` | Targets of the run-owned view links for all three backends | read | Selected dataset physical root, read-only bind; backend views link to it. |
| `/workspace/runs/<id>/resolved/...` | All backend configs and locks, container preflights, and the Resume verifier | read | Current run's `resolved/`, read-only overlay. |
| `/workspace/runs/<id>/cache/...` | Run-owned dataset views (`cache/dataset-view/`), Musubi dataset and guidance caches, and trainer-adjacent caches | read-write | Current run directory, read-write bind; the view is a dedicated child removed after publication. |
| `/workspace/runs/<id>/{outputs,checkpoints,samples,metrics,logs,realizations}/...` | Trainers, wrappers and executor logs/exit records | read-write | Current run directory, read-write bind. Output and Resume state must never live only in the disposable view. |
| `/workspace/cache/huggingface/...` and `/workspace/cache/models/...` | `HF_HOME`, `HF_HUB_CACHE`, Kura download helpers, Musubi/sd-scripts model links | read-write during acquisition | Workspace `cache/`, read-write bind. An explicitly selected local-path model under it needs a read-only overlay at its frozen runtime path. |
| `/workspace/cache/ai-toolkit/models/...` | AI-Toolkit `MODELS_PATH` | read-write | Workspace `cache/`, read-write bind. |
| `/workspace/artifacts/training-state/<id>/payload` | All three backend Resume commands and the verifier | read | Only the selected workspace `artifacts/training-state/<id>/`, read-only bind when resuming. |
| Selected workspace local-path model file or directory under `/workspace/...` | AI-Toolkit base path, Musubi/sd-scripts explicit role paths | read | Exact frozen file/directory, read-only bind, with no writable alias. The source may need its own physical-root mapping. |
| `/workspace` itself | Helper path-resolution root and the RunPod Pod workspace | path traversal; no implicit write | Docker: namespace of explicit mounts, not a broad bind. RunPod: disposable Pod workspace populated only by the verified selected-file transfer. |

The Docker mount table does not change ComfyUI render execution. Its
`/workspace` usage in `render_runpod.py`, the ComfyUI image, and authored
workspace templates is inventoried separately from the three training
backends; training-only mounts are not applied to render.
An explicit custom native command can name a path not known at compile time;
it remains an unverified escape hatch and cannot inherit a first-class
dataset-handoff claim. Its declared paths and mount needs must be shown and
checked rather than assumed covered by the removed workspace bind.

## Gaps closed when the broad bind was removed

1. Musubi's unconditional guidance cache moved from `resolved/musubi/` to the
   run-owned writable `/workspace/runs/<id>/cache/musubi/`, declared as a
   backend-cache write root; the compiled command creates it before
   precaching.
2. `docker_command` uses the effective explicit mount table instead of a
   read-write workspace root, and uncovered source, model, or Resume paths and
   writable aliases are rejected before launch.
3. Workspace-config extra mounts and explicit native model paths are resolved
   to physical sources and checked against protected roots; the writable
   `cache/` parent cannot re-expose a read-only local-path model.
4. Generated native commands, configs, env values, and helper arguments are
   checked as a set against the effective mounts by compile and launch tests.
5. RunPod transfers only the selected files with per-file SHA-256 proof on the
   Pod (see the selected-file transfer section of the implementation spec);
   this is a transport, not a bind mount.

## Real-container checks

1. Resume with each built-in backend reads
   `/workspace/artifacts/training-state/<id>/` from a read-only mount of only
   the selected artifact: AI-Toolkit, sd-scripts, and Musubi passed.
2. MiniMax-H3 guidance precaching writes the unconditional cache under
   `runs/<id>/cache/musubi/`, and training reads it from there; `resolved/` is
   not a write target. Proven on an A40 RunPod smoke; the read-only
   `resolved/` mount itself is proven by the local Docker smokes.
3. Each backend with a workspace local-path model mounts the exact model input
   read-only with no workspace bind: AI-Toolkit, sd-scripts, and Musubi
   passed. A missing or uncovered local path stops before Docker starts.

Source survey: `src/kura/backends/{ai_toolkit,musubi_command,musubi_datasets,
musubi_models,sd_scripts,sd_scripts_datasets,sd_scripts_models}.py`,
`src/kura/container_scripts/{training_state_verify,hf_download,
ai_toolkit_video_assert,musubi_dataset_assert,runpod_input_verify}.py`, `src/kura/executors/{docker,runpod}.py`,
`src/kura/run_commands/{launch,plan,runpod_ssh,render_runpod}.py`,
`src/kura/{paths,runtime_io,training_artifacts,init_templates}.py`, and the
training and ComfyUI Docker skeletons. This list is a review aid, not a
substitute for checking the generated runtime command.
