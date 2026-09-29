"""Translate opaque Musubi task selectors into native and dataset properties."""

from __future__ import annotations

from dataclasses import dataclass

from kura.backends.common import canonical_musubi_architecture


@dataclass(frozen=True)
class MusubiNativeTask:
    """Pinned Musubi task facts shared by command and dataset projection."""

    dataset_kind: str
    conditioning: str
    default: bool = False
    i2v_cache: bool = False
    clip_required: bool = False
    dual_dit_allowed: bool = False
    one_frame_kind: str | None = None

    @property
    def one_frame_allowed(self) -> bool:
        return self.one_frame_kind is not None


def _task(
    dataset_kind: str,
    conditioning: str,
    *,
    default: bool = False,
    i2v_cache: bool = False,
    clip_required: bool = False,
    dual_dit_allowed: bool = False,
    one_frame_kind: str | None = None,
) -> MusubiNativeTask:
    return MusubiNativeTask(
        dataset_kind=dataset_kind,
        conditioning=conditioning,
        default=default,
        i2v_cache=i2v_cache,
        clip_required=clip_required,
        dual_dit_allowed=dual_dit_allowed,
        one_frame_kind=one_frame_kind,
    )


# One Musubi-local registry owns native names, defaults, and dataset properties.
# These are deliberately not Kura-wide task or model categories.
MUSUBI_NATIVE_TASKS: dict[str, dict[str, MusubiNativeTask]] = {
    "wan": {
        "t2v-1.3B": _task("video", "text", default=True),
        "t2v-14B": _task("video", "text"),
        "i2v-14B": _task("video", "first-frame", i2v_cache=True, clip_required=True, one_frame_kind="single"),
        "t2i-14B": _task("image", "text"),
        "t2v-1.3B-FC": _task("video-control", "fun-control"),
        "t2v-14B-FC": _task("video-control", "fun-control"),
        "i2v-14B-FC": _task("video-control", "fun-control", i2v_cache=True, clip_required=True),
        "t2v-A14B": _task("video", "text", dual_dit_allowed=True),
        "i2v-A14B": _task("video", "first-frame", i2v_cache=True, dual_dit_allowed=True),
        "flf2v-14B": _task("video", "first-last-frame", i2v_cache=True, clip_required=True, one_frame_kind="intermediate"),
    },
    "minimax_h3": {
        "t2va": _task("video-audio", "text", default=True),
        "fl2va": _task("video-audio", "first-last-frame"),
        "ref2va": _task("video-references", "references"),
    },
    "hidream_o1": {
        "t2i": _task("image", "text", default=True),
        "i2i": _task("image-control", "image-reference"),
    },
    "hunyuan_video_1_5": {
        "t2v": _task("video", "text", default=True),
        "i2v": _task("video", "first-frame"),
    },
    "kandinsky5": {
        "k5-lite-t2v-5s-sd": _task("video", "text"),
        "k5-lite-t2v-10s-sd": _task("video", "text"),
        "k5-lite-i2v-5s-sd": _task("video", "first-frame"),
        "k5-pro-t2v-5s-sd": _task("video", "text", default=True),
        "k5-pro-t2v-5s-hd": _task("video", "text"),
        "k5-pro-t2v-10s-sd": _task("video", "text"),
        "k5-pro-t2v-10s-hd": _task("video", "text"),
        "k5-pro-i2v-5s-sd": _task("video", "first-frame"),
        "k5-pro-i2v-5s-hd": _task("video", "first-frame"),
        "k5-lite-t2v-5s-distil-sd": _task("video", "text"),
        "k5-lite-t2v-10s-distil-sd": _task("video", "text"),
        "k5-lite-t2v-5s-nocfg-sd": _task("video", "text"),
        "k5-lite-t2v-10s-nocfg-sd": _task("video", "text"),
        "k5-lite-t2v-5s-pretrain-sd": _task("video", "text"),
        "k5-lite-t2v-10s-pretrain-sd": _task("video", "text"),
    },
}

def musubi_native_task(architecture: str, value: object) -> str:
    """Resolve and validate the native task selector shared by projection and command."""
    canonical = canonical_musubi_architecture(architecture)
    tasks = MUSUBI_NATIVE_TASKS.get(canonical)
    if tasks is None:
        return str(value or "")
    if value is None or value == "":
        defaults = [name for name, profile in tasks.items() if profile.default]
        if len(defaults) != 1:
            raise ValueError(f"Musubi task table for {canonical!r} must declare exactly one default")
        return defaults[0]
    task = str(value)
    if task not in tasks:
        supported = ", ".join(tasks)
        label = "Wan" if canonical == "wan" else canonical
        raise ValueError(
            f"unsupported Musubi {label} native selector {task!r}; supported: {supported}"
        )
    return task


def musubi_native_task_profile(architecture: str, value: object) -> MusubiNativeTask | None:
    """Return declared task properties, or ``None`` for a taskless architecture."""
    canonical = canonical_musubi_architecture(architecture)
    if canonical not in MUSUBI_NATIVE_TASKS:
        return None
    task = musubi_native_task(canonical, value)
    return MUSUBI_NATIVE_TASKS[canonical][task]


def wan_native_selector(value: str) -> MusubiNativeTask:
    """Compatibility seam for command generation, backed by the shared task table."""
    profile = musubi_native_task_profile("wan", value)
    assert profile is not None
    return profile
