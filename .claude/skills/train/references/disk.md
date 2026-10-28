# Disk

A local run downloads models, writes checkpoints, and fills the disk Docker and WSL live on.
Running out mid-run can crash Docker or WSL. Kura measures before it starts and watches while
it runs.

- Run `kura doctor disk` before a local run, and before anything that downloads large models.
- `kura run plan` shows what the run will download and write, and how many checkpoints it
  leaves. Many checkpoints are the user's choice; mention the count when it looks unintended.
- The launch refuses a run whose downloads and writes do not fit, and the disk watch stops a
  running one before the disk falls below the reserve (`docker.min_free_gb`); it then ends
  `interrupted` with its reason, and what it wrote stays.
- When space is short, offer the user two ways: free space (the `cleanup` skill), or train on
  RunPod (a cost decision). Do not lower Kura's reserve (`docker.min_free_gb`)
  yourself; it is the user's to change.
- When a container left files you cannot remove, `kura fix-permissions` (preview) and
  `kura fix-permissions --yes` make them yours again; it deletes nothing.

Say it by effect, in the user's words, for example:

```text
⛔ The workspace has 18 GB free, and this run needs about 30 GB for the model and checkpoints;
it could crash Docker partway through.

▶ Choose
- A. Free space first → I list what can be freed and wait for your yes
- B. Run it on RunPod → a new plan with the GPU price; your dataset is copied to the rented GPU
```
