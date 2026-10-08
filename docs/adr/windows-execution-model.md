# ADR: Windows users run Kura in their own WSL distribution

Status: accepted owner decision (2026-09-30). Supersedes the 2026-09-30
direction toward a Kura-owned WSL distribution.

## Context

Most ComfyUI users run Windows, and most of them are not developers. Kura
trains in Linux containers, so a Windows install always involves the WSL2
virtual machine: Docker Desktop runs its engine there too. The question is
where the Kura CLI, the workspace, and the Docker engine live.

Measured on 2026-09-29 (RTX 4070 Ti, WSL 2.7.14, Docker Desktop). Each read
was made from a container using the same data:

| Read | WSL ext4 workspace | Windows NTFS workspace (`/mnt/c`) |
| --- | --- | --- |
| 4 GiB file, first read | 490 to 1900 MiB/s | about 250 MiB/s |
| 4 GiB file, cached | 4700 to 14700 MiB/s | about 250 MiB/s (no cache benefit) |
| `stat` on 3000 small files | 0.03 s | 3.2 s |

Native Windows is not viable today. Kura's dataset reader relies on no-follow,
directory-relative file opening, which Windows Python lacks. It also meets the
POSIX gaps the Windows CI job records. Developer Mode is not acceptable for the
target users.

Two layouts were considered:

- **A. A Kura-owned WSL distribution.** An installer adds a separate
  distribution holding Kura, the workspace, and Docker Engine with the NVIDIA
  Container Toolkit, without Docker Desktop.
- **B. The user's own WSL distribution.** Kura lives in the Ubuntu the user
  already has, or the one `wsl --install` creates, and uses Docker Desktop's
  WSL integration. This is the documented path today.

**Hazard observed with A.** All WSL distributions share one VM, one kernel,
and one `binfmt_misc`. On 2026-09-29 and 2026-09-30, a disposable
systemd-enabled Ubuntu 24.04 distribution twice removed the user's shared
`WSLInterop` registration and Docker Desktop's `/mnt/wsl` CLI mount.
Restoring them required the user's `sudo` and a Docker Desktop restart.

| Disposable distribution | WSLInterop afterwards |
| --- | --- |
| Docker Engine installed and a GPU container run, then `wsl --unregister` while running | lost |
| systemd booted, then left idle until WSL stopped it | lost |
| systemd booted, then `wsl --terminate`, twice | kept |
| stopped, then `wsl --unregister` | kept |
| systemd booted and running | kept |

WSL already empties `systemd-binfmt`'s `ExecStop` through its generated
override, so that unit is not the cause. The root cause is unknown.

Losing interop also breaks Kura itself: `src/kura/storage.py` calls
`powershell.exe` through interop for Windows drive facts, so
`kura doctor disk` degrades as well.

## Decision

1. **Windows users run Kura inside their own WSL distribution (B).** The CLI
   and the workspace live on its ext4 filesystem, under the user's home and
   never under `/mnt/c`. Docker Desktop's WSL integration provides the engine.
2. **A is deferred, not rejected.** In A, the idle stop and the unregister
   that caused the losses are routine operations: the Kura distribution idles
   after every session, and uninstall is an unregister. Shipping A with an
   unexplained hazard would break the owner's rule of not disturbing existing
   environments. For users without WSL, A and B start the same way (`wsl
   --install` plus one Ubuntu), so A's isolation benefit reaches only users who
   already run WSL, and they are the ones the hazard would hit.
3. **Docker Desktop is the default engine.** Its costs (a tray process, "not
   running" failures, update breakage, licensing terms) are known, and
   `kura doctor docker` already diagnoses them. Installing Docker Engine into
   the user's distribution changes their systemd setup. It is not a supported
   second path. Kura does not own a Docker Engine, which would mean
   maintaining its base OS, CUDA, and security updates as another product.
4. **Windows-side agents reach Kura through a shim, when one is built.** A
   `kura.cmd` calls `wsl.exe -d <distro> --cd <workspace> -- <absolute uv>
   run kura ...` with the distribution name recorded in a small setting. It
   removes the quoting and `PATH` traps. The same shim works for A.
5. **Native Windows stays unsupported.** The README says so, and the Windows
   CI job stays non-blocking as a record of the gaps.

## Conditions that would reopen A

- The root cause of the interop loss is found, and a guard passes
  reproductions of both the idle-stop case and the running-unregister case.
  Any A uninstall must still terminate the distribution before unregistering
  it.
- Docker Desktop becomes unusable for the target users, for example through a
  licensing change or a WSL integration regression.
- Supporting the variety of user distributions (non-Ubuntu, systemd disabled,
  old libraries) turns out to cost more than owning one distribution.
- The loss turns out to happen after any systemd distribution stops,
  including the user's own. That would be a WSL bug affecting A and B equally.

## Open verification

These are needed for B and do not require a GPU, so they can run in a Windows
virtual machine instead of the owner's working machine:

- The `kura.cmd` shim: arguments with spaces, Japanese text, and quotes; exit
  codes; UTF-8; and Ctrl-C.
- What happens to a long `kura run execute` behind `wsl.exe` when the
  Windows-side agent process exits. A detached heartbeat in WSL kept running
  for 20 minutes after every Windows-side window closed on the owner's PC with
  Docker Desktop running (`docs/smoke-evidence/2026-10-08-wsl-runner-lifetime.yaml`);
  the job runner, which carries the run since M3, is detached the same way.
  The runner itself and the case without Docker Desktop are still open.
- The first-run path on a Windows machine that never had WSL: `wsl --install`,
  Docker Desktop with WSL integration, then `kura doctor`.

Known native-Windows gaps, recorded so they are not mistaken for regressions:

- Kura refuses to read datasets on native Windows, which lacks the
  no-follow, directory-relative opens the dataset contract requires. The
  Windows CI job skips the tests that depend on this (`tests/platform_support.py`)
  and checks that the refusal stays explicit.
- About twenty places record run-relative paths with the host separator
  (`str(path.relative_to(run_dir))`), so a record written on native Windows
  would read `outputs\\x.safetensors`. Records written in WSL2 are unaffected.
  Fix this, with `as_posix()`, before any native-Windows process writes run
  records, for example a Windows-side Web UI.

Reproducing the interop loss for A needs a spare physical machine, because the
Docker Engine and GPU case cannot run in a VM without GPU passthrough. It must
not run on the owner's working machine.
