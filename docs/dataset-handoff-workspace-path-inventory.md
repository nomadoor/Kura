# Training workspace path inventory

Status: implementation audit in progress. This records the pre-implementation
survey required by `dataset-handoff-implementation-spec.md`; it is not proof
that the new mount layout is active or that every trainer mode has been smoked.

The survey covers literal `/workspace` paths, generated command/config paths,
container helpers, executor environment and mounts, and user-selectable native
paths in the current three built-in training backends. Docker currently binds
the entire workspace read-write in `executors/docker.py`. The table describes
the replacement mount coverage, not current behavior. Runtime-generated paths
must be checked against this table before that broad bind is removed.

| Container path or family | Current consumer | Access | Planned coverage |
| --- | --- | --- | --- |
| `/workspace/datasets/<id>/...` | AI-Toolkit, Musubi, sd-scripts native dataset inputs and container staging helpers | read | Selected dataset physical root, read-only bind; backend views link to it. |
| `/workspace/runs/<id>/resolved/...` | All backend configs/locks, stage and Resume verifiers | read | Current run's `resolved/`, read-only overlay. |
| `/workspace/runs/<id>/cache/...` | Existing stage helpers, Musubi dataset caches, sd-scripts native output and future disposable views | read-write | Current run directory, read-write bind; the future view is a dedicated deletable child. |
| `/workspace/runs/<id>/{outputs,checkpoints,samples,metrics,logs,realizations}/...` | Trainers, wrappers and executor logs/exit records | read-write | Current run directory, read-write bind. Output and Resume state must never live only in the disposable view. |
| `/workspace/cache/huggingface/...` and `/workspace/cache/models/...` | `HF_HOME`, `HF_HUB_CACHE`, Kura download helpers, Musubi/sd-scripts model links | read-write during acquisition | Workspace `cache/`, read-write bind. An explicitly selected local-path model under it needs a read-only overlay at its frozen runtime path. |
| `/workspace/cache/ai-toolkit/models/...` | AI-Toolkit `MODELS_PATH` | read-write | Workspace `cache/`, read-write bind. |
| `/workspace/artifacts/training-state/<id>/payload` | All three backend Resume commands and the verifier | read | Only the selected workspace `artifacts/training-state/<id>/`, read-only bind when resuming. |
| Selected workspace local-path model file or directory under `/workspace/...` | AI-Toolkit base path, Musubi/sd-scripts explicit role paths | read | Exact frozen file/directory, read-only bind, with no writable alias. The source may need its own physical-root mapping. |
| `/workspace` itself | Helper path-resolution root and RunPod archive extraction/workdir | path traversal; no implicit write | Docker: namespace of explicit mounts, not a broad bind. RunPod: disposable Pod workspace with separate transfer validation. |

The Docker mount refactor does not change ComfyUI render execution. Its
`/workspace` usage in `render_runpod.py`, the ComfyUI image, and authored
workspace templates is inventoried separately from the three training
backends; the refactor must not silently apply training-only mounts to render.
An explicit custom native command can name a path not known at compile time;
it remains an unverified escape hatch and cannot inherit a first-class
dataset-handoff claim. Its declared paths and mount needs must be shown and
checked rather than assumed covered by the removed workspace bind.

## Gaps to close before switching Docker mounts

1. Musubi's unconditional guidance cache previously targeted
   `/workspace/runs/<id>/resolved/musubi/minimax-h3-uncond.safetensors`.
   Its producer and consumer now target the run-owned writable
   `/workspace/runs/<id>/cache/musubi/` directory. The compiled command
   creates that directory before precaching; real-container confirmation
   remains part of the mount migration.
2. `docker_command` currently unconditionally mounts the workspace root
   read-write, and `workspace_mount_mappings` assumes that root mapping.
   Replace both with the effective explicit mount table; reject uncovered
   source/model/Resume paths and writable aliases before launch.
3. Workspace-config extra Docker mounts and explicit native model paths can
   overlap protected roots or point outside the workspace. Resolve physical
   sources and effective target precedence before accepting them. Do not let
   the writable `cache/` parent re-expose a read-only local-path model.
4. Generated native commands, configs, env values, and helper arguments must
   be checked as a set against the effective mounts for each backend path.
   Add public compile/launch tests for dataset, `resolved/`, run writes,
   model acquisition, Resume and custom-path rejection. A literal search
   alone cannot prove dynamic values are covered.
5. RunPod currently stages broad workspace archive content. Replace it with
   the selected-file transfer and Pod-side SHA-256 proof specified in the
   handoff contract. This remote copy is not a Docker bind-mount problem.

## Real-container checks still required

The closed mount table has unit coverage, but these checks still require the
owner-approved local Docker smoke before the branch can merge:

1. Resume once with each built-in backend and prove that the container reads
   `/workspace/artifacts/training-state/<id>/payload` from the read-only mount.
2. Run MiniMax-H3 guidance precaching and prove that the unconditional cache is
   created under `runs/<id>/cache/musubi/` while `resolved/` remains read-only.
3. Launch each backend with a workspace local-path model and prove that the
   exact model input is mounted read-only. A missing or uncovered local path
   must stop before Docker starts and therefore before any model acquisition.

Source survey: `src/kura/backends/{ai_toolkit,musubi_command,musubi_datasets,
musubi_models,sd_scripts,sd_scripts_datasets,sd_scripts_models}.py`,
`src/kura/container_scripts/{dataset_stage,
training_state_verify,hf_download}.py`, `src/kura/executors/{docker,runpod}.py`,
`src/kura/run_commands/{launch,plan,runpod_ssh,render_runpod}.py`,
`src/kura/{paths,runtime_io,training_artifacts,init_templates}.py`, and the
training and ComfyUI Docker skeletons. This list is a review aid, not a
substitute for checking the generated runtime command.
