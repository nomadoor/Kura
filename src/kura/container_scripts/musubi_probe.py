# This script runs inside the Musubi container with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import concurrent.futures
import json
import os
import subprocess
import sys


def probe_item(payload):
    root, adapter, script, do_help, script_timeout = payload
    path = os.path.join(root, script)
    item = {"adapter": adapter, "script": script, "exists": os.path.isfile(path)}
    if item["exists"] and do_help:
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
                item["help_output_tail"] = proc.stdout[-2000:]
        except Exception as exc:
            item["help_error"] = str(exc)
    return item


def main():
    root = "/opt/musubi-tuner/src/musubi_tuner"
    items = json.loads(sys.argv[1])
    do_help = sys.argv[2] == "1"
    script_timeout = float(sys.argv[3])
    payloads = [(root, adapter, script, do_help, script_timeout) for adapter, script in items]
    workers = min(4, max(1, len(payloads)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(probe_item, payloads))
    ok = all(
        item["exists"]
        and (not do_help or item.get("help_returncode") == 0)
        for item in results
    )
    print(json.dumps({"script_root": root, "results": results}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
