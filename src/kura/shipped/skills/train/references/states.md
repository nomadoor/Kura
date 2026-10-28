# Run states

`kura run status <run-id>` shows the state, and its `summary` says what Kura is doing about the
run. `kura run execute` exits 0 when the run completed, 1 when it failed, 130 when it was
interrupted, 2 for any other end, and 3 when it stopped following a run that needs a look
and is still going.

| State | Meaning | Next |
|---|---|---|
| `draft` | `run.yaml` is not frozen yet | `kura run compile <run-id>` |
| `compiled` | frozen, never started | show `kura run plan`; after the user's yes, `kura run execute <run-id> --yes` |
| `queued`, `staged`, `launching` | waiting for, or being put on, a machine | follow: `kura run execute <run-id>` |
| `running` | training; on RunPod also until the outputs are collected | follow: `kura run execute <run-id>` |
| `publishing` | local: training ended; outputs are being moved into place | keep following |
| `completed` | training ended and its outputs are in place | check the outputs and report |
| `failed` | the trainer stopped with an error | read `kura run logs <run-id>`; a fix is a new run |
| `launch_failed` | the trainer never started | read the error; a fix is a new run |
| `interrupted` | stopped (`kura run stop`, the disk watch, or the machine went away) | report; what a local run wrote stays, what a RunPod Pod had not sent back is gone |
| `unknown` | the end was not seen | `kura run reconcile <run-id>`, or `kura run stop <run-id>`, so nothing is left running; then report |
| `recovery_required` | trained, but its files could not all be put in place | report the reason `kura run status` gives; collecting again cannot change it |
| `stopped` | an older Kura's word for `interrupted` | as `interrupted` |

If your session ended while a run was going, follow it with `kura run execute <run-id>`; never
assume it finished and never start it again. A run that ended without completing starts
again only as a new run: `kura run new --from <run-id> --slug <words>`, with a new plan and
yes.

## Saves in the plan

`kura run plan` lists two kinds of saves:

| Plan line | What it is |
|---|---|
| `checkpoint` `keep_last 0` | the LoRA saves (`save_every_n_steps`); `0` keeps every one |
| `Training state` `keep` | how many training states are kept to continue the run (`kura run resume`) |
