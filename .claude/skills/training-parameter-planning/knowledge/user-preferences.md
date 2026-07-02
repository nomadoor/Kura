# User preferences

This user's own tested preferences and tendencies. These outrank baseline
cards when they cover the case at hand. Entries move into a baseline card
only when the owner says they should apply generally.

Format example (keep entries in this shape):

```
- lr starting point: prefer 5e-5 over the common 1e-4 for <scope>
  source: owner (<date>)
  note: <why / observed behavior>
```

Confirmed entries:

- character LoRA resolution: 768 is usually sufficient as a practical starting
  point; raise toward 1024 only when the model/task benefits and hardware has
  headroom.
  source: owner (2026-07-02)

Known signals not yet confirmed as preferences:

- The owner has hinted at preferring lower starting learning rates (7e-5 /
  5e-5) over the common 1e-4, but has not confirmed this as a general
  preference — ask when it next becomes relevant, then record the answer
  here with `source: owner`.
- The owner uses gradient checkpointing sparingly and dislikes it being
  enabled without need. Already encoded as skill procedure (headroom rule),
  noted here for context. source: owner (2026-07-02)
