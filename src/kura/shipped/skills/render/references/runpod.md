# Rendering on RunPod

A RunPod render starts a rented GPU (a Pod) with Kura's ComfyUI image, sends the LoRAs and
input images, renders, brings the images back, and deletes the Pod. It bills by the hour, so
it is the user's decision. Cost, the lease, and recovery work as for training: see
references/runpod.md of the `train` skill.

## Writing the run

- `executor.name: runpod` in `run.yaml`.
- The GPU is `compute.gpu` in the render's `run.yaml` (a list, in order of preference). Without
  it the workspace's list is used (`comfyui.runpod.gpu_type_ids` in `workspace.yaml`, else
  `runpod.gpu_type_ids`). Choose it with the user.
- The workflow's base models must be known by name: from the sample's sidecar, or from
  `comfyui.model_registry` in `workspace.yaml`. Compile refuses a model it cannot place before
  any Pod exists; propose its source to the user and, with their yes, add it to the registry.
  Never guess a repository from a file name.
- LoRAs (from cases, through the sidecar's `lora_insert` or a `lora` patch) and `type: image`
  inputs are sent to the Pod. Other model files are not.

## Running it

- `kura render launch <run-id> --dry-run` shows the GPU, the hourly price, and the cost
  ceiling, not the cases. Show them to the user together with the cases from your cases file
  (step 5 of the skill).
- After their yes, `kura render launch <run-id> --yes`.
- `kura render launch` has no `--max-lease`: a render's Pod gets the default lease (12h), and
  the dry run's cost ceiling is for that lease. `kura run lease <run-id> <duration>` changes it
  while the Pod runs; a longer lease is the user's billing decision, as for training.

## A local render that failed

Moving it to RunPod is a new cost decision, never an automatic fix. Make a new render with
`kura render new`, copy the settings from the failed run's `run.yaml`, set
`executor.name: runpod`, compile, and show the dry run for a yes.
