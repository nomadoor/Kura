# Each image records its CUDA version, and RunPod hosts are chosen to match it

Status: accepted owner decision.

Date: 2026-10-07

## Context

A RunPod host runs an image only when its NVIDIA driver supports the CUDA
version the image was built with. Some RunPod hosts still run driver 550, which
supports CUDA 12.4. Kura's images need CUDA 12.8 (driver 570 or later) or 13.0
(driver 580 or later), so a Pod on such a host bills but cannot train. A real
smoke hit one.

RunPod's Pod creation accepts `minCudaVersion`: the oldest CUDA version a
host's driver may support. RunPod also accepts `allowedCudaVersions`, an exact
list, but its documented values stop at 13.0 while hosts reporting 13.2 already
exist (measured 2026-10-07), so an exact list silently drops newer hosts.

## Decision

**The CUDA version is recorded with the image, and Kura derives the RunPod host
filter from it.** Users do not configure it.

- Kura records the CUDA version of every image it pins, keyed by digest, so a
  run compiled against an earlier image keeps that image's requirement.
- When an image is rebuilt, its Dockerfile also records the version as an
  image label.
- Every Pod creation, for training and for rendering, sends the image's CUDA
  version as `minCudaVersion`. GPU stock and price shown before launch use the
  same filter.
- An image whose CUDA version Kura does not know, such as a workspace override
  or a RunPod template, asks for the newest version Kura has seen on RunPod. A
  newer driver runs an older image, so this finds fewer hosts rather than
  hosts that are too old. The plan says so.

## Consequences

- Pods no longer land on hosts whose driver is too old for the image.
- Fewer hosts qualify, mostly on Community Cloud. The plan's stock reflects
  that.
- New CUDA versions on RunPod need no change for images Kura knows. For
  unknown images, the newest version Kura has seen is updated by hand.
- Re-pinning an image means recording its CUDA version alongside the digest.

## Alternatives

- **A workspace setting naming the CUDA version.** Rejected: forgetting it
  brings back the failure, and the version is a fact about the image, not a
  user choice.
- **Always requiring the newest CUDA version.** Safe, but it excludes hosts
  that would run the CUDA 12.8 images, and GPUs are already scarce.
- **An exact `allowedCudaVersions` list.** Rejected: a host reporting a version
  missing from the list is never chosen, and RunPod already has such hosts.
