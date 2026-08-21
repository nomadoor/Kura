# Local ComfyUI execution

## Endpoint and model verification

Local render authorizes HTTP calls only to the endpoint recorded in the Kura
run. It does not authorize starting or restarting ComfyUI, using Docker as a
fallback, installing nodes, or downloading models.

1. Confirm reachability at the configured endpoint.
2. Run workflow-aware doctor and verify every loader model is visible.
3. Before staging a Kura LoRA, run doctor with `--probe-stage` and verify the
   configured `comfyui.lora_dir` is visible to that exact process.
4. Compile only after the endpoint and staging configuration are correct so
   their facts are frozen in the manifest.

If `comfyui.lora_dir` is absent, ask once for the user's actual
`models/loras` directory and record it in ignored `workspace.yaml`. Never infer
it from the ComfyUI process or silently retarget to another directory.

If the endpoint is unreachable, ask the user to start it or identify the
correct endpoint. A reachable conventional port is only a diagnostic hint.

## Staging

Kura outputs remain the source of truth. Launch may temporarily expose them to
ComfyUI:

- LoRA: stage under configured `lora_dir/Kura_tmp`, patch the visible name, and
  remove the staged file or link afterward.
- Model patch: use the configured model-patch directory and binding.
- Input image: copy a compile-frozen image under configured `input_dir/Kura_tmp`
  and remove it afterward. Do not symlink input images; ComfyUI rejects a
  `LoadImage` path that escapes its input directory.

Local launch deduplicates repeated references to the same staging target.
Distinct required artifacts are staged before the finite queue starts and
cleaned up afterward.

If a probe is not visible, explain that this endpoint does not scan the
configured directory. With user approval, inspect their ComfyUI configuration
and propose the correct workspace value. Do not ask for manual file copying as
the first workaround.

## Dedicated smoke instances

Starting a smoke ComfyUI is a separate environment mutation and requires
explicit approval. Label it `io.kura.purpose=smoke`, use a separate endpoint
and model-path configuration, record ownership in the smoke evidence, and
remove it before handoff. Never turn it into the endpoint for a normal render.
