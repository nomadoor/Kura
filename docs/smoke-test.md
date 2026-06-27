# Runtime smoke tests

Kura's local Docker training runtime and ComfyUI render runtime have both been exercised end to end.

```bash
uv run kura image build ai-toolkit
uv run kura doctor docker
uv run kura run launch <docker-smoke-run> --executor docker
uv run kura render launch <comfyui-render-run>
```

The Docker smoke run produced `logs/stdout.log`, lifecycle events, a completed status, and a realization record. The ComfyUI run produced an image under `samples/images/` and a matching `samples/images.jsonl` entry.

`resolved/env.lock` is the immutable compile-time environment lock. Each `realizations/<id>.json` is an append-only launch-time record containing the Docker image ID, command, mounts, GPU flag, secret presence, and exit code. Secret values are never recorded.
