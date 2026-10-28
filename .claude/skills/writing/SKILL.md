---
name: writing
description: How to write what the user reads. Use every time you tell the user anything about Kura work, such as a plan to approve, progress, results, problems or failures, and choices.
---

# Writing

The user reads you in a terminal or a chat window, among thinking traces and command output.
Make it plain at a glance where to read and what to choose.

## What matters

In this order, when two pull different ways:

1. **Be accurate.** Keep what you checked apart from what you did not. Call a failure a failure.
   Never write a guess as a fact.
2. **Make what the user must decide stand out the most.** Mark what they should check or choose.
3. **One thing in one place.** Say nothing twice. Everything a decision needs is written
   somewhere.
4. **Be short.** Leave out work that went as planned and anything that changes no decision.
5. **Use plain words.** In the user's language, say what it means for them.

## Look

Screens render different things. Use only what holds up everywhere.

- **Instead of headings, use a marker and a bold line.** `#` headings do not work on every
  screen.
- **No block quotes (`>`).** Some screens dim them, so they stand out less.
- **Tables** for settings, differences, and lists. At most 3 columns, short cells.
- **Markers**, one at the start of a line. Only these five:

| Marker | Means |
|---|---|
| ✅ | done, ready |
| ⛔ | stopped, failed |
| ⚠️ | please check this |
| ℹ️ | good to know (no decision needed) |
| ▶ | please choose |

- Bold only the first line, the marker lines, and the key values.
- No other emoji.

## Shape

```text
✅ **The state, in one line**

⚠️ **Please check** (if anything)
- what looks wrong

**<topic>** ("This training", "Changed from the previous run", "What happened"...)
a table, or short bullets

▶ **Choose** (if there is a choice)
- **A. ...** (recommended) → what happens if chosen
- **B. ...** → what happens if chosen
```

- **First line**: the state, bold, with its marker.
- **⚠️**: only two things: a value that does not match what the user said, and a value that
  changed though they did not ask for it. It comes first.
- Kura's notices and warnings (a download, no sample images), assumptions, and what you could
  not check are not ⚠️: put them in the table or under ℹ️. Something that went wrong is ⛔.
- **Values you chose** that the user did not name go in the table, marked "(proposed)"; not
  under ⚠️.
- **Topic lines** name the topic as the user sees it. Not the writer's words ("questions",
  "next steps").
- **▶**: for each choice, what happens if it is chosen. Mark the recommended one and say why in
  a few words. With no choice to make, leave out ▶ and end with one line: "Next, I will ...".
- **One ▶ per message.** When the user decides two things, give each its own line inside it:
  `- **Backend**: A. AI-Toolkit (recommended) / B. Musubi Tuner`.

## Tables or bullets

| Content | How |
|---|---|
| settings and numbers | table: item / value |
| differences from the previous run | table: item / before / now |
| files or runs | table: name / count / location |
| choices | ▶ bullets: choice → what happens |
| reasons, things noticed | short bullets |

## By situation

**Showing a plan for approval**
- First line: ✅ ready (not started yet).
- ⚠️: values that do not match what the user said, or changed unasked (if there are none,
  leave ⚠️ out).
- "This training" table: model, dataset, main settings, time, disk; on RunPod the hourly price
  and the cost ceiling.
- "Changed from the previous run" table: the differences `kura run plan` lists.
- ▶: start or change, and what each does.

**Progress**
- Only when a long task starts and at milestones: what is going, and how long is left, in a line
  or two.

**Results**
- First line: ✅ finished, or ⛔ stopped.
- "Results" table.
- Things noticed (numbers unlike what was expected, warnings). Judging them is the user's.

**Problems, failures, a stop**
- First line: ⛔ what happened. The harder it is to say, the earlier it goes.
- ℹ️: what is kept and what is lost; whether anything is still costing money.
- "What happened": the cause you know, and what you do not know.
- ▶: the next steps as choices. Do not fix it and move on by yourself.

## Other parts

| Part | How |
|---|---|
| commands | only ones the user types, in a code block; do not list the commands you ran |
| file locations | give the path; open it only when asked (`kura open`) |
| numbers | with units, rounded (about 32 min, about 20 GB, about $3) |
| times and dates | in the user's time zone; a date and time, not "earlier" |

## Words

- Write in the language the user writes in. Commands, file names, and model names stay as
  they are.
- Words needed to use Kura are fine: run, step, checkpoint, LoRA, backend.
- Do not use Kura's internal words; when one is needed for a choice, explain it in a few words.

| Not | Instead |
|---|---|
| realization, manifest, lock, publication, projection | say what happened ("the launch record", "the list", "the frozen settings") |
| a trainer's internal names (`_t_e_cache`) | what it does ("reads each caption once at the start") |
| abbreviations (TE, OOM) | words ("text encoder", "out of memory") |

- Say "should" or "not checked" for what you did not check. Do not state it as fact.

## Never

- Ask a question in the middle of the text. Put choices in the one ▶ place.
- Suggest work nobody asked for as a question ("Shall I also ...?").
- Say you will go ahead if the user does not answer.
- Say the same thing twice.
- Report each step that went as planned.
- Put a failure or a surprise late and small.
- Write a guess as a checked fact.

## Examples

**Showing a plan**

```text
✅ **The training is ready (not started yet)**

⚠️ **Please check**
- The learning rate is **1/71 (7e-7)** of the previous run's. Did you mean 7e-5?

**This training**
| | |
|---|---|
| Model | MOSS-Image Base |
| Dataset | Ely (47 images) |
| Learning rate | **7e-7** |
| Steps | 5000 (saved every 200, 25 saves) |
| Time | about 9 hours |
| Disk | about 25 GB (330 GB free) |

**Changed from the previous run (20261026-2309)**
| | Before | Now |
|---|---|---|
| Learning rate | 5e-5 | **7e-7** (×1/71) |
| Steps | 3000 | 5000 |

▶ **Choose**
- **A. Set the learning rate to 7e-5** (recommended) → I fix it and show this again
- **B. Start as it is** → training starts now (about 9 hours)
```

**Showing a render**

```text
✅ **The render is ready (not started yet)**

**This render**
| | |
|---|---|
| Compared | steps 1000 / 2000 / 3000 and no LoRA × 2 prompts = **8 images** |
| Workflow | workflows/samples/flux2-klein/flux_2_klein_base_4b_text2image_api.json |
| Seed | 42 for all |
| LoRA strength | 0.8 (proposed) |
| Sheet | rows: steps, columns: prompts → the training run's comparisons/ |

**Prompts**
| Name | Prompt |
|---|---|
| kitchen | Ely, cooking in a bright kitchen, ... |
| pier | Ely, standing on a wooden pier at dusk, ... |

▶ **Choose**
- **A. Render it** (recommended) → 8 images, about 8 min; then I make the sheet
- **B. Change it** → tell me what; I make it again and show it
```

**Results**

```text
✅ **Training finished (about 9 h 10 min)**

**Results**
| Name | Count | Location |
|---|---|---|
| saves every 200 steps | 25 | runs/20261027-0920_ely-lr7e5-b2_b4c3/outputs/ |
| final file | 1 | same |

**Noticed**
- The loss barely falls after about step 3000
```

**A stop**

```text
⛔ **Training stopped partway (2400 / 5000 steps)**

ℹ️ The 12 saves up to step 2400 are kept. Nothing costs money (local run).

**What happened**
- Free disk fell below 20 GB, so Kura stopped the training
- Cause: the saves were larger than expected (about 300 MB each)

▶ **Choose**
- **A. Free space, then continue from step 2400** (recommended) → I list what can be freed and
  remove it after your yes, then show the plan of the continuing run (about 4 hours left)
- **B. Stop here** → compare the 12 saves up to step 2400
```
