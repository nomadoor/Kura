"""Declarative task contracts for first-class Musubi adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WanTaskContract:
    target_modality: str
    i2v_cache: bool = False
    clip_required: bool = False
    control_required: bool = False
    dual_dit_allowed: bool = False
    one_frame_allowed: bool = False


WAN_TASK_CONTRACTS: dict[str, WanTaskContract] = {
    "t2v-1.3B": WanTaskContract("video"),
    "t2v-14B": WanTaskContract("video"),
    "i2v-14B": WanTaskContract("video", i2v_cache=True, clip_required=True, one_frame_allowed=True),
    "t2i-14B": WanTaskContract("image"),
    "t2v-1.3B-FC": WanTaskContract("video", control_required=True),
    "t2v-14B-FC": WanTaskContract("video", control_required=True),
    "i2v-14B-FC": WanTaskContract("video", i2v_cache=True, clip_required=True, control_required=True),
    "t2v-A14B": WanTaskContract("video", dual_dit_allowed=True),
    "i2v-A14B": WanTaskContract("video", i2v_cache=True, dual_dit_allowed=True),
    "flf2v-14B": WanTaskContract("video", i2v_cache=True, clip_required=True, one_frame_allowed=True),
}


def wan_task_contract(task: str) -> WanTaskContract:
    try:
        return WAN_TASK_CONTRACTS[task]
    except KeyError as exc:
        supported = ", ".join(WAN_TASK_CONTRACTS)
        raise ValueError(f"unsupported Musubi Wan task {task!r}; supported: {supported}") from exc
