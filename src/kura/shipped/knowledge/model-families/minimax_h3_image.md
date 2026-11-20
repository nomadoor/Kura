# minimax_h3_image

## Training

- Recorded: AI-Toolkit `model_arch: minimax_h3_image` completed one real
  optimizer step of text-to-image on RunPod (A40, 48 GB) on 2026-11-20, with
  the baseline load and the training adapter, at 256×256 on one image. The
  container's memory reached its 50 GB limit while loading, without being
  killed. Not recorded yet: a 24 GB GPU, output quality, or settings for a
  real dataset.
  source: evidence ai-toolkit-minimax-h3-image-t2i-runpod-2026-11-20
- That run predates the module filter below, so its 50 `adaln_proj` layers
  got LoRA weights that stayed zero.
  source: evidence ai-toolkit-minimax-h3-image-t2i-runpod-2026-11-20
- Backend and model: AI-Toolkit `model_arch: minimax_h3_image` with
  `model.base: Comfy-Org/MiniMax-H3-Image`, from image and caption datasets.
  The baseline fills the upstream UI's pre-quantized load (`qtype: convrot8`,
  `qtype_te: nvfp4`) and its text-embedding and latent caches.
  source: AI-Toolkit baseline entry `minimax_h3_image` (pinned commit bdd55603)
- Editing: a dataset whose samples each have a target, one or more `control`
  images, and a caption trains it with references, passed in the item's
  order. The trainer keeps each reference at its own size and scales it to a
  short edge between 1024 and 2048 (`native_config.model.model_kwargs`
  `reference_min_short_edge` and `reference_max_short_edge` change the
  bounds); the text encoder sees it at a 1024 short edge. Samples may have
  different numbers of references at batch size 1; at a larger batch the
  trainer refuses a batch whose counts or reference sizes differ. Untested:
  no editing run is recorded yet.
  source: AI-Toolkit commit bdd55603 (minimax_h3_image.py reference sizing, minimax_h3.py `get_noise_prediction`); Kura profile `multi-control-image`
- The trainer labels the references `Picture 1`, `Picture 2`, and so on for
  the text encoder, so a caption can name them as `<Picture 1>`; upstream
  says to describe the whole image in detail. A caption that names no
  picture still trains.
  source: AI-Toolkit commit bdd55603 (minimax_h3_image.py, text_encoder.py; ui.tsx model notes)
- The model is step and guidance distilled (6 steps, no CFG), and upstream
  trains it with its training adapter live. A run compiled by this Kura
  loads that adapter by default
  (`ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors`,
  from the baseline entry): `kura run plan` lists it as the model input
  `training_adapter`, and the trainer downloads it at the start of the run.
  A run compiled by an earlier Kura keeps what it froze, so its plan lists no
  adapter and it loads none unless its `native_config` named one.
  `backend.config.assistant_lora_path` names another adapter: a Hugging Face
  `user/repo/file` path, or a local file on local Docker only (RunPod does not
  upload it). `assistant_lora_path: none` trains without one. It cannot be
  written in `native_config`.
  source: AI-Toolkit commit 5b1670a (ui.tsx entry `minimax_h3_image`, minimax_h3_image.py)
- The baseline leaves the `adaln_proj` layers out of the LoRA, as the
  upstream UI does (`network.network_kwargs.ignore_if_contains: [adaln_proj]`
  in `kura run plan`). A LoRA on those layers never trains: the model reads
  their weights directly, so the saved weights stay zero.
  source: AI-Toolkit baseline entry `minimax_h3_image` (pinned commit bdd55603); minimax_h3 transformer.py `MiniMaxH3AdalnProj.forward`
- Training draws only the model's 6 sampling timesteps, each through its own
  output head, so `timestep_type` has no effect.
  source: AI-Toolkit commit 5b1670a (minimax_h3_image.py `get_train_scheduler`)
- Upstream says the model was trained for 4 to 20 megapixel output (2048×2048
  and up) and shows more artifacts at lower resolutions. The default download
  is about 38 GB.
  source: AI-Toolkit commit 5b1670a (ui.tsx model notes for `minimax_h3_image`)
- Gradient checkpointing is refused, as for the other MiniMax-H3 archs.
  source: Kura `MINIMAX_H3_ARCHS` (ai_toolkit.py)

## character

- no entry yet.

## style

- no entry yet.
