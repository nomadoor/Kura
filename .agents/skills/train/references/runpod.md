# Training on RunPod

A RunPod run trains on a rented GPU (a Pod) that bills by the hour. Kura creates the Pod,
sends the inputs, follows the training, brings the outputs back, and deletes the Pod. The
user decides every cost.

## Before the yes

- Run `kura run plan` while the run is still a draft: it shows each requested GPU's stock and
  hourly price. Choose `compute.gpu` (a list, in order of preference) with the user, then
  compile and show the final plan.
- Show the **cost ceiling** with the hourly price: the most the Pod's GPU time can bill before
  its lease ends it. It does not cover disk storage, which RunPod bills separately; say so. When Kura says the ceiling is unknown, say why; do not estimate one.
- By default a run waits for its GPU when none is free (nothing bills while waiting); the plan
  says so. Prices may change while it waits.
- Moving a run from this machine to RunPod, or to a dearer GPU, is a new cost decision, never a
  fix for a local failure.

## While it runs

- `--max-lease` (default 12h) deletes the Pod that long after its container starts, whatever
  happens here; the image download before that bills too. If Kura warns that training looks
  longer than the lease, tell the user and ask before `kura run lease <run-id> <duration>`;
  a longer lease is their billing decision.
- After training, if the outputs were not collected, the Pod waits (`--unattended-wait`, by
  default the longer of 2 hours and the job time) and then deletes itself.
- Kura's runner keeps following if your session ends. `kura run execute <run-id>` follows it
  again; it never creates a second Pod.

## After it runs

- `pod_stopped_at` in `kura run status`: Kura deleted the Pod; it no longer bills.
  `pod_missing_at`: RunPod said the Pod no longer exists (it deleted itself or was deleted
  elsewhere); outputs it had not sent back are gone.
- The Pod's cost is `pod_gpu_cost` in `kura run status`; quote it, do not compute one.

## When something goes wrong

```sh
kura doctor runpod                    # the account, the key, Pods Kura knows about
kura run reconcile <run-id>           # look at the Pod and record what is true
kura run download <run-id> --force    # collect the outputs by hand
kura run stop <run-id>                # delete the Pod, only after the user's yes
```

`kura run stop` ends the training. Tell the user what stops and what is kept, and run it only
on their word. It refuses while the Pod holds work not collected yet and says how to collect
it; tell the user what would be lost, and add `--yes` only on their word. Report the cause
`runs/<run-id>/logs/runner.log` names. A missing or refused `RUNPOD_API_KEY` is the user's to
set with `kura secrets set RUNPOD_API_KEY`. If a request fails with a permission error before
it reaches RunPod, the agent may lack network access: see `.kura/reference/external-access.md`.
