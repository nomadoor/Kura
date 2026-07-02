# This script runs inside the Musubi container with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import json
import os
import subprocess
import sys


def main():
    root = "/opt/musubi-tuner/src/musubi_tuner"
    items = json.loads(sys.argv[1])
    do_help = sys.argv[2] == "1"
    script_timeout = float(sys.argv[3])
    results = []
    ok = True
    for adapter, script in items:
        path = os.path.join(root, script)
        item = {"adapter": adapter, "script": script, "exists": os.path.isfile(path)}
        if not item["exists"]:
            ok = False
        elif do_help:
            try:
                proc = subprocess.run(
                    [sys.executable, path, "--help"],
                    cwd="/opt/musubi-tuner",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=script_timeout,
                )
                item["help_returncode"] = proc.returncode
                if proc.returncode != 0:
                    ok = False
                    item["help_output_tail"] = proc.stdout[-2000:]
            except Exception as exc:
                ok = False
                item["help_error"] = str(exc)
        results.append(item)
    print(json.dumps({"script_root": root, "results": results}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
