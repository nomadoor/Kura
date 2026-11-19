# minimax_h3_image

## Training

- Untested: Kura only compiles it. No run has trained MiniMax-H3-Image
  through Kura yet.
- Backend and model: AI-Toolkit `model_arch: minimax_h3_image` with
  `model.base: Comfy-Org/MiniMax-H3-Image`, from image and caption datasets.
  The baseline fills the upstream UI's pre-quantized load (`qtype: convrot8`,
  `qtype_te: nvfp4`) and its text-embedding and latent caches. Reference
  images (editing) are not accepted yet.
  source: AI-Toolkit baseline entry `minimax_h3_image` (pinned commit bdd55603)
- The model is step and guidance distilled (6 steps, no CFG), and upstream
  trains it with its training adapter live. Kura does not add the adapter by
  itself yet: write it in `backend.config.native_config.model`:
  `assistant_lora_path: ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors`.
  The trainer downloads it at the start of the run.
  source: AI-Toolkit commit 5b1670a (ui.tsx entry `minimax_h3_image`, minimax_h3_image.py)
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
