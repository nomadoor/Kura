# Rendering on RunPod

A RunPod render starts a rented GPU (a Pod) with Kura's ComfyUI image, sends the LoRAs and
input images, renders, brings the images back, and deletes the Pod. It bills by the hour, so
it is the user's decision. Cost, the lease, and recovery work as for training: see
references/runpod.md of the `train` skill.

## Writing the run

- `executor.name: runpod` in `run.yaml`.
- The workflow's base models must be known by name: from the sample's sidecar, or from
  `comfyui.model_registry` in `workspace.yaml`. Compile refuses a model it cannot place before
  any Pod exists; propose its source to the user and, with their yes, add it to the registry.
  Never guess a repository from a file name.
- LoRAs (from cases, through the sidecar's `lora_insert` or a `lora` patch) and `type: image`
  inputs are sent to the Pod. Other model files are not.

## Running it

- `kura render launch <run-id> --dry-run` shows the GPU, the hourly price, and the cost
  ceiling. Show them to the user.
- After their yes, `kura render launch <run-id> --yes`.

## A local render that failed

Moving it to RunPod is a new cost decision, never an automatic fix. Make a new render with
`kura render new`, copy the settings from the failed run's `run.yaml`, set
`executor.name: runpod`, compile, and show the dry run for a yes.
