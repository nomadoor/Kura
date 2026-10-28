# Naming the model

Each backend takes its weights from a different place. `kura run capabilities <backend>`
lists the model selector values it accepts; `--arch <value>` prints that value's facts.

- **ai-toolkit** trains from `model.base`: a Hugging Face repository id (`owner/name`), which
  the trainer downloads, or, for a local Docker run, an absolute path as the container sees
  it (`/workspace/...` inside the workspace). `backend.config.model_arch` says what the
  weights are.
- **sd-scripts** and **musubi-tuner** take their weights from `backend.config.model_paths`
  and `backend.config.model_downloads`; musubi-tuner can instead name a `model_bundle`, which
  downloads its own files and is never combined with `model_downloads`.
  `backend.config.architecture` says what the weights are, and `model.base` is only a label.
  For musubi-tuner `architecture: flux2`, `model_version` picks the variant, and its weights
  are still named (`model_bundle: flux2-<model_version>`, or `model_paths`/`model_downloads`).

Kura keeps no list of default weights. A card's **Model files** section lists the files a
recorded run trained from; a refusal for a missing model points there.

Card names are family names, not selector values (ai-toolkit `sd1` is the `sd15` card;
`flux2_klein_4b` is the `flux2` card). When unsure which card covers a selector, read the
cards' headings rather than guess.

The trainer saves LoRA weights in its own precision unless the run sets one: Musubi Tuner and
sd-scripts in fp32 (`save_precision`; for sd-scripts only through `extra_args`), AI-Toolkit
in fp16 (`native_config.save.dtype`). An fp32 file is about twice the size.
