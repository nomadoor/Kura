# User preferences

This user's own tested preferences and tendencies. These outrank baseline
cards when they cover the case at hand. Entries move into a baseline card
only when the owner says they should apply generally.

Format example (keep entries in this shape):

```text
- lr starting point: prefer 7e-5 over the common 1e-4 for <scope>
  source: owner (<date>)
  note: <why / observed behavior>
```

Confirmed entries:

- LoRA learning-rate starting point: use 7e-5 as the owner's provisional Kura
  default when no evaluated run or architecture-specific reason says
  otherwise. Treat 1e-4 as a common stronger option, not the default.
  source: owner (2026-07-02)
- character LoRA resolution: 768 is usually sufficient as a practical starting
  point; raise toward 1024 only when the model/task benefits and hardware has
  headroom.
  source: owner (2026-07-02)

Known signals not yet confirmed as preferences:

- The owner uses gradient checkpointing sparingly and dislikes it being
  enabled without need. Already encoded as skill procedure (headroom rule),
  noted here for context. source: owner (2026-07-02)
