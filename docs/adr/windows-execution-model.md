# ADR: Windows users run Kura in a Kura-owned WSL distribution

Status: accepted owner direction (2026-09-30); the guard and the installer are not yet verified.

Date: 2026-09-30

## Context

Most ComfyUI users run Windows, and most of them are not developers. Kura
trains in Linux containers, so a Windows install always involves the WSL2
virtual machine: Docker Desktop runs its engine there too. The open question
is where the Kura CLI and workspace live.

Measured on 2026-09-29 (RTX 4070 Ti, WSL2, Docker Desktop). Each read was made
from a container using the same data:

| Read | WSL ext4 workspace | Windows NTFS workspace (`/mnt/c`) |
| --- | --- | --- |
| 4 GiB file, first read | 490 to 1900 MiB/s | about 250 MiB/s |
| 4 GiB file, cached | 4700 to 14700 MiB/s | about 250 MiB/s (no cache benefit) |
| `stat` on 3000 small files | 0.03 s | 3.2 s |

A native Windows CLI with an NTFS workspace pays that cost on every model load
and dataset scan. It also meets every POSIX mismatch: symlinks without Developer
Mode, `os.getuid`, path separators, CRLF, and 8.3 short names. The Windows CI
job shows those gaps today. Developer Mode is not an acceptable requirement for
the target users.

A Docker Engine installed directly in a WSL distribution runs GPU containers
without Docker Desktop. This was verified with Docker 29.8.1, the NVIDIA
Container Toolkit, and `nvidia-smi` in a CUDA container. Removing Docker Desktop
removes its tray process, its "not running" failures, its update breakage, and
its licensing questions.

A Windows-side agent can call `wsl.exe -d <distro> --cd <workspace> --
<uv> run kura ...`. Output, exit codes, and UTF-8 text cross the boundary
intact. Shell quoting breaks once PowerShell or cmd sits in the middle, and a
non-login WSL shell does not have `uv` on `PATH`.

**Hazard observed.** All WSL distributions share one VM and kernel. Starting a
fresh systemd-enabled Ubuntu 24.04 distribution removed the shared
`WSLInterop` binfmt registration, so Windows executables stopped running from
the user's existing distribution. The same session also lost Docker Desktop's
CLI mount. The likely cause is `systemd-binfmt` resetting `binfmt_misc`; this
is inferred, not proven. Restoring the registration with `sudo` and restarting
Docker Desktop repaired it.

## Decision (proposed)

1. On Windows, Kura runs inside a **Kura-owned WSL distribution**. The CLI, the
   workspace, and the Docker Engine with the NVIDIA Container Toolkit all live
   there. Docker Desktop is not required.
2. The distribution **must not disturb the user's other distributions**. Before
   the distribution is ever started with systemd, the installer masks
   `systemd-binfmt` or ships a binfmt configuration that re-registers
   `WSLInterop`, or it starts `dockerd` without systemd. Whichever guard is
   chosen must pass a reproduction of the hazard above before it ships.
3. A Windows-side shim, `kura.cmd`, forwards arguments through `wsl.exe` with
   the absolute `uv` path. Agents and users type `kura ...`, and the shim
   removes the quoting and `PATH` traps.
4. The workspace stays on ext4. Windows reaches datasets and outputs through
   Explorer at `\\wsl.localhost\<distro>\...`; the installer adds shortcuts.
   ComfyUI stays on Windows and is reached over HTTP.
5. Native Windows stays unsupported. The README says so, and the Windows CI job
   stays non-blocking as a record of the gaps.

## Open verification before implementation

- Reproduce the interop hazard on a disposable distribution and prove the
  chosen guard prevents it. The user's own distribution must be able to run a
  Windows executable before and after the Kura distribution starts.
- Decide how the installer obtains the distribution and whether it needs one
  administrator prompt and a reboot on a machine that has never had WSL.
- Confirm what happens to a long training run when the Windows-side agent
  process exits during a `wsl.exe` call.
- Decide how an existing Docker Desktop user migrates. The two engines must not
  both claim the `docker` command in one distribution.
