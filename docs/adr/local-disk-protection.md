# ADR: A runtime watch guards the local disk; the launch checks one bound

Status: accepted owner decision.

Date: 2026-10-18

## Context

A full disk is an irreversible accident on the user's machine: a trainer or
a download that fills the drive can corrupt files, stop WSL or Docker, and
lose work that has nothing to do with Kura. Until now Kura guarded it only at
launch: a free-space floor (`docker.min_free_gb`, default 100 GiB) plus an
estimate of the run's writes.

Thirteen context-free acceptance runs on a 12 GB GPU (2026-10-17) measured
what the estimate was worth:

- 95% of the bytes written were model downloads into the Hugging Face cache.
  A run's own writes (checkpoints, training states, caches) were at most about
  3 GiB.
- The checkpoint estimate was a flat 1 GiB per file, 3.7 to 23 times the real
  LoRA sizes. Training states were not counted at all.
- AI-Toolkit's own model downloads (13 to 19 GiB on a first run) were counted
  as 0 unless the user declared a size.
- The 100 GiB floor absorbed every error. It also refused runs that would
  have fit, and it guarded nothing once the run had started.

A launch check alone cannot be right: it predicts writes that Kura does not
make (a trainer's own downloads), and anything else on the machine can fill
the disk while the run trains.

## Decision

**1. A runtime watch is the enforcement.** While a local Docker run trains,
the job runner's follower for that run checks the effective free space
(`kura.storage.probe_storage`, which reads the Windows drive behind a WSL
virtual disk) of every store the run writes to: the workspace, the Hugging
Face cache, and every writable mount, one check per backing store. It checks
about every 30 seconds, not at every poll, because on WSL each check starts
PowerShell. When a store is below the reserve, the follower records the stop
and its reason (the store, its free bytes, the reserve) before it stops the
container, then ends the run as interrupted with that reason, and the
notification says it. Checkpoints and training states already written stay;
a run that kept training state can be resumed. A store whose WSL backing
drive cannot be measured (PowerShell timed out, or WSL interop is broken) is
judged by Linux free space, as the RunPod stage and pull gate judges it, and
the follower logs once that it could not measure it: one failed PowerShell
call never stops a long run, and a full virtual disk is still caught. The
launch keeps refusing an unmeasured backing unless the run sets
`safety.allow_storage_risk: true`.

The reserve must absorb what the fastest writer can write between two
checks: it assumes reserve ≥ 30 s × the fastest write rate (20 GiB holds about
680 MiB/s for 30 s, more than a model download or a trainer writes on the
machines Kura targets). A machine that writes faster raises
`docker.min_free_gb`.

This guards the accident whatever wrote the bytes: the trainer, its own
downloads, or another program.

**2. The reserve is 20 GiB.** `docker.min_free_gb` defaults to 20; a run may
raise its own reserve with `safety.max_run_disk_gb`, and one function,
`local_reserve_gib`, gives the launch and the watch the larger of the two
(image builds and `kura doctor disk`, which have no run, keep
`docker.min_free_gb`). One function, `kura.storage.storage_floor_problem`,
decides "below the reserve" for the launch, the watch, `kura image build`,
and `kura doctor disk`.

**3. The launch checks one bound.** One function, `run_write_estimate`,
returns per store what a run will write: the downloads Kura makes, measured
from Hugging Face metadata (a file already in the cache counts 0), and the
run's own weights-sized writes, bounded by `rank × 64 MiB` per weights-sized
file on disk at once (the most checkpoints at once, the trainer's training
state folders, and Kura's copies of them; a state counts its optimizer
moments too). There is no per-architecture size table. The launch refuses a
run when a store's free space minus the estimate is below the reserve; the
plan shows the same numbers. The bound uses only measured facts and the
rank, so it is never a size the user declares. It assumes linear LoRA
modules, whose size grows with the rank; conv LoRA, LoKr, full-matrix
training, and very large models can exceed it, and the watch covers them.

**4. AI-Toolkit's own downloads are measured data per pin.** At each pinned
AI-Toolkit version's real smoke, the Hugging Face cache growth is recorded
and shipped with Kura as data keyed by the pin, the model architecture, the
model selector, and the quantization. A known key is counted (0 for files
already in the cache, in either cache layout); an unknown key is a warning
and is not counted, because the watch covers it.

**5. Declared estimates are retired.** `safety.checkpoint_estimate_gb`,
`safety.disk_cache_estimate_gb`, and `safety.model_download_estimate_gb` are
refused with a message naming what replaces them; `safety.max_run_disk_gb`
goes too if it only caps an estimate. The large-download guard
(`safety.large_model_download_gb`, `safety.allow_large_model_downloads`) is a
separate decision and stays.

**6. RunPod.** A full Pod disk fails the run, the maximum lease bounds the
bill, and nothing irreversible happens on the user's machine. The RunPod
container-disk check compares the same `run_write_estimate` with
`runpod.container_disk_gb` as information, and refuses only when the measured
downloads Kura makes alone exceed the container disk.

**7. Promise P8** becomes: "Kura stops a local run before it drives any
store it writes to below the reserve; the launch refuses a run whose
measured downloads and bounded writes do not fit above the reserve."

## Alternatives considered

- **Keep a launch-only estimate and make it precise.** A per-architecture
  size table and declared sizes for unmeasurable writes would still miss
  what the trainer or another program writes after launch, and every new
  model would need a table entry. The watch covers all of it with one check.
- **Keep the 100 GiB floor as the safety margin.** It refuses runs that fit
  and still does nothing once training starts.
- **Watch at every poll.** On WSL each probe starts PowerShell; every 30
  seconds leaves the run's own writes far from filling 20 GiB between checks
  (the measured runs wrote at most about 3 GiB in total).

## Consequences

- The job runner's local follower gains a periodic disk check and a stop with
  a recorded reason. The stop request carries the reason, so the intent and
  its reason are on disk before `docker stop` runs.
- Workspaces that relied on the 100 GiB default get a 20 GiB reserve; a
  workspace that sets `docker.min_free_gb` keeps its value.
- Run files that set the retired keys are refused with a pointer to their
  replacement; run records written before keep their meaning.
- The conformance P8 check compares the bound with the bytes of the
  weights-sized files the run had on disk, and checks that the watch did not
  stop the run.
