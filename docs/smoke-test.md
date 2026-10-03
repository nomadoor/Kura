# Runtime smoke tests

Kura's local Docker training runtime and ComfyUI render runtime have both been exercised end to end.

```bash
uv run kura doctor docker
uv run kura run launch <docker-smoke-run> --executor docker
uv run kura render launch <comfyui-render-run>
```

The Docker smoke run produced `logs/stdout.log`, lifecycle events, a completed status, and a realization record. The ComfyUI run produced an image under `samples/images/` and a matching `samples/images.jsonl` entry.

`resolved/env.lock` is the immutable compile-time environment lock. Each `realizations/<id>.json` is an append-only launch-time record containing the Docker image ID, command, mounts, GPU flag, secret presence, and exit code. Both record `kura_version` and `kura_source`, which says where the installed Kura came from (Git URL and commit, or editable checkout and commit). Secret values are never recorded. Beside it, the append-only `realizations/<id>.phases.jsonl` records when each launch boundary was observed — Pod creation, SSH readiness (with the container start RunPod reports), upload, remote job start and exit, download, and Pod stop for RunPod; the start request and Docker's own start and finish times locally — and the completion summary prints the resulting `time` breakdown. A timing line that cannot be written only warns. RunPod `startup` starts at the successful create request, so a capacity wait (recorded separately) is excluded; when RunPod reports the container start, `startup` is split into allocation plus image pull and container boot. The RunPod `job` segment ends when the controller observes the remote exit record, a few seconds (typically under 6s) after the remote timestamp, which is kept as a fact on that line.
