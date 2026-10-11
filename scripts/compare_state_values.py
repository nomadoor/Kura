#!/usr/bin/env python3
"""Compare two training-state payloads by value (the conformance check of P3).

`real_smoke.py` runs this inside a run's pinned trainer image, which has torch, with
no network and both payloads mounted read-only:

    python -I compare_state_values.py <payload-a> <payload-b> <file>...

Torch archives (optimizer.bin, optimizer.pt) are loaded with torch.load and
safetensors weights with safetensors.torch.load_file, so two saves of equal tensors
compare equal although their archive bytes differ. It prints one JSON object:
{file: [differing leaves, leaves]}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def leaves(a: Any, b: Any, *, torch: Any = None, numpy: Any = None) -> tuple[int, int]:
    """How many leaves of two loaded values differ, of how many: dictionaries by key, lists
    and tuples by position, tensors and arrays by dtype, shape, and every element."""
    if isinstance(a, dict) and isinstance(b, dict):
        counts = [leaves(a[key], b[key], torch=torch, numpy=numpy) if key in a and key in b else (1, 1) for key in a.keys() | b.keys()]
        return sum(count[0] for count in counts), sum(count[1] for count in counts)
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        counts = [leaves(left, right, torch=torch, numpy=numpy) for left, right in zip(a, b)]
        return sum(count[0] for count in counts), sum(count[1] for count in counts)
    if torch is not None and (torch.is_tensor(a) or torch.is_tensor(b)):
        same = torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif numpy is not None and (isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray)):
        same = isinstance(a, numpy.ndarray) and isinstance(b, numpy.ndarray) and a.dtype == b.dtype and a.shape == b.shape and bool(numpy.array_equal(a, b))
    else:
        try:
            same = type(a) is type(b) and bool(a == b)
        except Exception:  # a value whose comparison is not a truth value counts as different
            same = False
    return (0 if same else 1), 1


def main(argv: list[str]) -> int:
    import torch
    from safetensors.torch import load_file

    try:
        import numpy
    except ImportError:
        numpy = None

    def load(path: Path) -> Any:
        if path.suffix == ".safetensors":
            return load_file(str(path))
        return torch.load(path, map_location="cpu", weights_only=False)

    first, second, *names = argv
    result = {name: list(leaves(load(Path(first) / name), load(Path(second) / name), torch=torch, numpy=numpy)) for name in names}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
