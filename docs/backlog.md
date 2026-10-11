# Backlog

Findings a review round deferred (`AGENTS.md`, "How a change is made"): Minor
and Nit findings, and pre-existing issues a review happened to find. They are
fixed later in one batch pull request, which removes the lines it fixes.

One line per item:

```text
- YYYY-MM-DD | area | finding | source (pull request or review) | promise (P1–P11) or -
```

A rejected item keeps its line, starts its finding with `REJECTED:`, and says
why, so it is not raised again.

## Open

- 2026-10-11 | AI-Toolkit outputs | AI-Toolkit writes outputs under `outputs/<run-id>/` on Docker and under `outputs/` on RunPod; both are recorded correctly in `status.outputs`. Accepted deviation for the folder layout, kept here until the layouts are made the same or the deviation is documented for users | `docs/adr/promises-and-verification.md` | P5
- 2026-10-11 | RunPod wording | The wording of the maximum lease and of the unattended wait does not tell the two apart | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | progress | Progress text shows `unknown/100` | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | job runner | The runner's status wording after an idle stop | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | AI-Toolkit checkpoints | AI-Toolkit names a step checkpoint by its zero-based step index, so `_000000050` holds 51 updates while the training state saved with it is published at step 51; users need this explained | deferred before this backlog existed; source not recorded | P2
- 2026-10-11 | doctor | `kura doctor disk` output is too long | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | plan JSON | `cadence_steps` mixes integers and the text "trainer default" | deferred before this backlog existed; source not recorded | -

- 2026-10-11 | conformance | P8 can only fail on the checkpoint count: the byte estimate is 1 GiB per checkpoint against small LoRAs, no scenario run exercises retention or pruning, and the 0.5 s sampling would miss a short peak | review of the promises-and-verification pull request | P8
- 2026-10-11 | conformance | `_sample_peak` can raise `FileNotFoundError` when a staging folder disappears during `rglob`, which stops the harness while `kura run execute` keeps running | review of the promises-and-verification pull request | -
- 2026-10-11 | conformance | With `--runpod` and no `--yes`, the harness still creates datasets and compiles a run before printing the cost, and `--yes` launches whatever the ceiling (even unknown) | review of the promises-and-verification pull request | P6
- 2026-10-11 | conformance | The AI-Toolkit conformance model (`hf-internal-testing/tiny-stable-diffusion-pipe`) has no pinned revision | review of the promises-and-verification pull request | P4
- 2026-10-11 | conformance | A Resume that fails to run shows `-` for P3 instead of FAIL (P1 still fails and the exit code is 1) | review of the promises-and-verification pull request | P3

## Rejected

- 2026-10-11 | capabilities | REJECTED: declare field types in `kura run capabilities`; typed fields would break runs that give `learning_rate` as a string | deferred before this backlog existed; source not recorded | -
