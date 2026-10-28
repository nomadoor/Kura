---
name: cleanup
description: Free disk space in a Kura workspace. Use when the disk is short, when Kura refused or stopped a run for lack of space, when the user wants to remove caches, old models, or training states, or when files a container left cannot be removed.
---

# Cleanup

## Flow

```text
kura doctor disk          free space now, and how much can be freed
kura cleanup plan         what can be freed, grouped by what removing it costs
   ↓ explain it to the user by effect ─→ one yes
kura cleanup ... --yes    remove exactly what was agreed
kura doctor disk          report the space freed
```

## Who decides

| | Decides |
|---|---|
| Kura | what can be removed, what removing each costs, and that nothing in use is removed |
| You | explaining the list by effect, as one proposal |
| The user | what to remove; outputs and datasets above all |

## What can be freed

`kura cleanup plan` groups everything by what removing it costs:

| Class | Removing it costs | Examples |
|---|---|---|
| free | nothing | finished runs' temporary files, RunPod upload archives, Docker build cache, stopped containers |
| download | a download the next time it is needed | models in the Hugging Face cache, old Kura images |
| resume data | that run can no longer be continued (its LoRAs stay) | finished runs' training states |
| outputs | the trained results | checkpoints and LoRAs; listed only with `--outputs` |

What is in use (a model an unfinished run needs, say) is left out or refused.

## Steps

1. Run `kura cleanup plan` (add `--outputs` only when the user asks about their results).
2. **Make one proposal, by effect**: how much space, what it costs to get back, whether it can
   be undone, what stays. Do not list commands.
3. Get one yes for exactly what you proposed.
4. Run each command first without `--yes` to see what it removes. If it lists anything the
   proposal did not, stop and show it. Add `--yes` only when they match. One
   command per class, all under the same yes:
   ```sh
   kura cleanup --yes                          # the free class
   kura cleanup --models <id>... --yes         # these models
   kura cleanup --images old --yes             # old Kura images
   kura cleanup --resume-data <run>... --yes   # these runs' training states
   ```
5. Run `kura doctor disk` again and report the space freed.

- Docker's build cache, and a model cache outside the workspace (`docker.hf_cache`), may be
  shared with other projects or workspaces. Say so before removing them.

## Removing trained results

Only when the user names the run and says what to do:

```sh
kura cleanup --archive-outputs <run> --to <folder> --yes   # move the results elsewhere
kura cleanup --checkpoints <run> --keep <file>... --yes    # remove all but these checkpoints
kura run prune <run>... --yes                              # delete whole runs
```

- Ask the user which checkpoints to keep; never pick for them.
- A run a render is using is refused until that render ends.

## How to say it

In the user's words, effect first:

```text
✅ About 31 GB can be freed.

| What | Size | Removing it costs |
|---|---|---|
| temporary files | 12 GB | nothing |
| 2 models unused since September | 15 GB | a download if you use them again |
| training states of 3 finished runs | 4 GB | those runs cannot be continued (LoRAs stay) |

Your datasets and trained LoRAs are not touched.

▶ Choose
- A. Remove all three → about 31 GB freed
- B. Temporary files only → about 12 GB freed
```

Not like this: "May I run `kura cleanup --models org/model --yes`?"

## Files that cannot be removed

Files a container created can belong to another user. `kura fix-permissions` shows what it
would fix; `kura fix-permissions --yes` makes them yours. It removes nothing.

## What Kura does not do

Deleting datasets, `docker system prune -a`, and anything on the WSL or Windows side (shrinking
the virtual disk, say). When one is needed, explain what it does and let the user do it.
