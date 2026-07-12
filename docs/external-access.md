# External access from AI agents

Kura is a normal CLI process. It does not grant or bypass permissions belonging
to Codex, Claude Code, the shell, the operating system, Docker, or ComfyUI.
The same Kura command can therefore fail inside an agent sandbox and succeed in
a regular terminal.

Kura workflows may need access outside the workspace:

- network access for Hugging Face and RunPod;
- the Docker daemon for local training;
- a local ComfyUI HTTP endpoint;
- temporary write access to the configured `comfyui.lora_dir`.

Use the relevant existing diagnostic:

```sh
uv run kura doctor runpod
uv run kura doctor docker
uv run kura doctor comfyui --probe-stage
```

When a diagnostic says "this process", it reports only the permissions of the
process that ran the command. It does not prove that RunPod, Docker, ComfyUI, or
the host is broken, and the observation is not frozen into run artifacts.

## Codex CLI

For a workspace-write session, enable network access in the Codex user config:

```toml
[sandbox_workspace_write]
network_access = true
```

Add a known external directory when starting the session:

```sh
codex -C /path/to/kura-workspace --add-dir /path/to/ComfyUI
```

With `approval_policy = "never"`, Codex cannot ask to elevate a denied action.
Use `on-request` when you want Codex to request permission for exceptional
operations. Restart the session after changing its permission context.

## Claude Code and other agents

Allow network access and the configured external directories using that
agent's own permission settings. Kura does not detect the agent or edit its
configuration. Keep permissions scoped to the workspace and the external
resources the workflow actually uses.
