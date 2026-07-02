# Kura last-mile hardening plan

This is the accepted implementation plan for the final robustness pass before
larger feature work resumes. It is intentionally internal-facing: the work is
about implementation safety and maintainability, not user-facing behavior.

## Context

Kura is close to its intended shape as an agent-operated LoRA training and
ComfyUI render workspace. The remaining weaknesses are not in the core model,
but in implementation mechanics:

1. State files such as `status.json` are written non-atomically, so a crash or
   concurrent monitor read can observe a partial file.
2. Python scripts executed inside containers are embedded as strings in
   `backends.py` and `doctor.py`, which keeps them outside normal lint and
   `py_compile` coverage.
3. `backends.py`, `executors.py`, and `run_commands.py` are large enough that
   focused maintenance is becoming harder. The previous blocker for splitting
   was mostly string-based `mock.patch` targets in tests; `backends.py` has no
   such patch targets and can be split first.

The work should proceed in this order: WS1, then WS2, then WS3. Each stage must
stay behavior-preserving and green before the next stage begins.

## Known facts

- Internal imports use absolute `from kura.x import y` style.
- The only known cycle-like edge is the lazy import in `tui.py` that reaches
  `_runpod_ssh_details` and `_ssh_base` through `kura.cli`; those functions
  actually live in `run_commands.py` and are re-exported through `cli.py`.
- CLI wiring is eager and flat: argparse registers command functions with
  `set_defaults(func=...)`.
- Tests patch by string path, such as `kura.<module>.<name>`. After moving a
  function, patch targets must point at the module where the calling code now
  looks up that name.

## WS1: Atomic Writes

Add `src/kura/fsio.py` with small, Kura-independent helpers:

```python
def atomic_write_text(path: Path, text: str) -> None:
    ...

def atomic_write_json(path: Path, value: Any) -> None:
    ...

def atomic_write_yaml(path: Path, value: Any) -> None:
    ...
```

`atomic_write_text` must write a temporary file in the destination directory,
flush it, and replace the target with `os.replace`.

Replace existing state writes without changing public signatures:

| Location | Target | Change |
| --- | --- | --- |
| `executors.py` `_write_json` | `status.json`, observations | Delegate to `atomic_write_json` after existing redaction. |
| `workspace.py` `dump_yaml` | `run.yaml`, `manifest.lock.yaml`, `env.lock` | Delegate to `atomic_write_yaml`. |
| `render.py` `write_yaml` | render lock files | Remove the duplicate helper and use `workspace.dump_yaml`. |
| `render.py` `status()` | `status.json` | Write atomically after read-modify-write. |
| `run_commands.py` status writes | `status.json` | Write atomically, especially the remote stdout sync loop. |
| `cli.py` run creation/compile | `status.json` | Write atomically. |
| `cli.py` index rebuild | `index.jsonl` | Build the full text and atomically replace the file. |

Do not change append-only files such as `events.jsonl` and `stdout.log`.

Tests:

```sh
uv run python -m unittest discover -s tests
uv run python scripts/check_python.py
```

Add focused `fsio` unit tests for replacement, JSON/YAML formatting, and
temporary-file cleanup.

## WS2: File-Based Container Scripts

Keep the delivery mechanism as `python -c <source>` so Docker images and
Dockerfiles do not need to change. Move the source strings into real files under
`src/kura/container_scripts/`, loaded with `importlib.resources`.

Add:

- `src/kura/container_scripts/__init__.py` with `script_source(name: str) -> str`
- `hf_download.py`
- `safetensors_validator.py`
- `prune_checkpoints.py`
- `musubi_probe.py`

Rules:

- Container scripts must not import `kura`; they run inside runtime containers.
- Preserve argv/env behavior. If any embedded source currently relies on
  interpolation, move that data to argv or env instead.
- Convert top-level execution to `main()` plus
  `if __name__ == "__main__": main()`.
- Leave shell snippets in `run_commands.py` and templates in
  `init_templates.py` alone.

Tests:

```sh
uv run python -m unittest discover -s tests
uv run python scripts/check_python.py
```

Add tests that compile each extracted script, including the nested `CHILD`
source in `hf_download.py`.

## WS3: Package Splits

Split the large modules in three stages. Each old module becomes a package with
an `__init__.py` that re-exports the previous public surface, including private
symbols that current tests or modules import.

Rules:

1. Move code only. Do not change behavior, names, or signatures.
2. Update `mock.patch` targets to the module where the caller now resolves the
   patched name.
3. Run tests after each stage before continuing.

### Stage A: `kura/backends/`

Lowest risk because current tests do not patch `kura.backends.*`.

Suggested modules:

- `common.py`: shared helpers such as `_datasets`, `_toml_scalar`, `_truthy`,
  `_int_or_none`, `_require_paths`, `_script_command`, `_append_flag`, and
  `_extra_args`
- `ai_toolkit.py`: AI-Toolkit compile and command generation
- `musubi_datasets.py`: Musubi dataset TOML and paired JSONL generation
- `musubi_models.py`: Musubi model paths, bundles, downloads, validation, and
  `MUSUBI_ADAPTER_SCRIPTS`
- `musubi_command.py`: Musubi training command assembly and checkpoint pruning

`__init__.py` must re-export at least:

- `compile_ai_toolkit`
- `command_ai_toolkit`
- `compile_musubi_tuner`
- `command_musubi_tuner`
- `musubi_model_download_specs`
- `MUSUBI_ADAPTER_SCRIPTS`
- `_safetensors_validator_code`

### Stage B: `kura/executors/`

Suggested modules:

- `common.py`: redaction helpers, status/observation helpers, event writing,
  constants, `_write_json`, and progress materialization
- `docker.py`: Docker preflight, command building, launch, reconcile, and stop
- `runpod.py`: RunPod requests, settings, staging, session launch, reconcile,
  and stop

`__init__.py` must re-export the previous executor API used by imports and
tests, including `_redact_secret_text`, `_redact_secrets`, and
`_materialize_stdout_progress`.

Expected patch updates include moving `_runpod_request` patches to
`kura.executors.runpod._runpod_request` and moving `subprocess.run` /
`shutil.disk_usage` patches to the caller's new submodule.

### Stage C: `kura/run_commands/`

Last stage because it has the most patch targets.

Suggested modules:

- `plan.py`: run plan, stop, logs, stage, duration parsing, disk and checkpoint
  preflight
- `runpod_ssh.py`: upload, download, pull, SSH/SCP helpers, lease guard, stdout
  sync, secret env payload, and retry downloads
- `render_runpod.py`: RunPod ComfyUI render orchestration
- `launch.py`: local/RunPod launch, wait, and remote run lifecycle

`__init__.py` must re-export the symbols currently imported by `cli.py` and the
private helpers imported by tests.

As part of this stage, change the lazy import in `tui.py` from `kura.cli` to
`kura.run_commands` or the appropriate submodule so the CLI re-export is no
longer needed for that path.

## Commit Plan

Use one commit per stage:

1. WS1: atomic write helpers and replacements
2. WS2: container script extraction
3. WS3 Stage A: `backends` package split
4. WS3 Stage B: `executors` package split
5. WS3 Stage C: `run_commands` package split

Do not create a PR without owner approval.

## Validation

After every commit:

```sh
uv run python -m unittest discover -s tests
uv run python scripts/check_python.py
```

After all stages:

```sh
uv run python scripts/check_release.py
uv run kura --help
uv run kura doctor workspace
uv run kura run plan <existing-run-id>
```

For WS2 and WS3, spot-check command generation by compiling an existing run
before and after the refactor and comparing `resolved/command.json`. Only
irrelevant whitespace differences are acceptable.
