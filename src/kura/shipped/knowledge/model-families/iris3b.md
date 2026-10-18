# iris3b

## Training

- Backend and model: AI-Toolkit `model_arch: iris3b`; its baseline entry trains
  from `speridlabs/iris-3b` with `convrot8` quantization of the model and the
  text encoder and `low_vram` on.
  source: AI-Toolkit baseline entry `iris3b` (pinned commit dcebc744)
- VRAM class: on an RTX 4070 Ti with 12 GB at 768 px, with the text encoder
  resident (the baseline), about 2.0 to 3.3 s per step at 11.8 to 12.0 GB; with
  `native_config.train.cache_text_embeddings: true`, 0.9 to 1.4 s per step at
  7.4 GB.
  source: acceptance run 2026-10-18 (local RTX 4070 Ti 12 GB)
- Recorded: in that acceptance test an AI-Toolkit `iris3b` LoRA ran 650 steps
  in 15 minutes on AI-Toolkit 0.13.30. The run folder was not kept.
  source: acceptance test 2026-10-17/18 (local RTX 4070 Ti 12 GB, pins of Kura 0.6.0)

## character

- no entry yet.

## style

- no entry yet.
