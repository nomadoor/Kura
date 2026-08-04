# LoRA evaluation examples

These authored examples show evaluation intent and provenance. They are not
launch-ready model recipes. Copy the relevant block into a Kura render run,
replace placeholder inputs, inspect the family knowledge, present the complete
plan to the user, then compile.

- `run-reconstruction.yaml` asks whether learned examples can be reproduced;
  it explicitly does not claim generalization.
- `run-outfit-transfer.yaml` changes only outfit content while keeping the
  checkpoint, seed, workflow, strength, and prompt policy fixed.
