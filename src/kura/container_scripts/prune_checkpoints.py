# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import glob
import os
import re
import sys


def main():
    output_dir, output_name, threshold_raw = sys.argv[1], sys.argv[2], sys.argv[3]
    threshold = int(threshold_raw)
    pattern = os.path.join(output_dir.rstrip("/"), output_name + "-step*.safetensors")
    removed = []
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"-step(\d+)\.safetensors$", os.path.basename(path))
        if not match:
            continue
        step = int(match.group(1))
        if step < threshold:
            os.remove(path)
            removed.append(os.path.basename(path))
    print(f"[kura] pruned {len(removed)} checkpoints before step {threshold}", flush=True)


if __name__ == "__main__":
    main()
