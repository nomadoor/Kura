# Evaluation review template

Use this when the user requests a review, comparison, contact sheet, or XY plot
without already fixing every execution detail. Inspect available evidence first;
recommend concrete values instead of returning a questionnaire.

Respond in the user's language with this structure:

```text
I will treat this as <category>.

Varied:
- <axis and ordered values>

Fixed:
- workflow/model variant: <value>
- prompts: <roles or full list>
- negative prompt: <full text>
- seeds: <values>
- adapter strength and application: <value>

Plan:
- <axis cardinalities> = <N> logical cases in one Kura render queue
- Kura generates raw images and complete per-case metadata
- after completion, the agent assembles <requested presentation artifacts>

This can establish <claim>. It cannot establish <limits/confounds>.

Complete positive prompts:
1. <prompt>

May I launch this plan?
```

Do not require the user to know Kura's case schema. Translate a vague request
such as "compare the steps" into a recommended checkpoint set, prompt roles,
seeds, strength, expected case count, and presentation layout from the training
run, dataset, workflow, family card, and prior evidence.

If the user already supplied all material choices and explicitly said to run
them, treat that as approval; summarize the frozen plan before launch without
asking the same question again.

Do not say Kura needs a new sweep feature when the requested combinations can
be represented as explicit cases. Do not promise an automatic Kura contact
sheet: Kura owns raw generation and metadata, while the agent owns presentation
from completed images under `AGENTS.md`.
