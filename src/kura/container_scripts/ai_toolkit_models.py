# This script runs on a RunPod Pod with stdlib only, and Kura imports it on the
# host for local Docker runs. Do not import kura here; the Pod receives it as
# source text.
"""List the model files AI-Toolkit resolved and downloaded into MODELS_PATH.

The trainer resolves `model.base` itself and does not report which local
candidate it loaded, so the record lists every file under the directory with
its size and mtime. No snapshot is taken before the run: a reader compares each
mtime with the realization's `launched_at` to tell what this run wrote.
"""

import json
import os
import sys
from datetime import datetime, timezone

RECORD_SUFFIX = ".ai-toolkit-models.json"
LOADED_CANDIDATE_NOTE = (
    "AI-Toolkit does not report which local candidate it loaded; compare each mtime "
    "with the realization's launched_at to see which files this run wrote"
)


def models_root(write_roots):
    """The MODELS_PATH directory a frozen command declares, or None when it declares none."""
    for item in write_roots if isinstance(write_roots, list) else []:
        if isinstance(item, dict) and item.get("role") == "model-cache" and item.get("env") == "MODELS_PATH" and isinstance(item.get("path"), str):
            return item["path"]
    return None


def list_model_files(root):
    """Every file under root as {path, size_bytes, mtime}, leaving out dot-named entries such as .cache/."""
    files = []
    if not os.path.isdir(root):
        return files
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            path = os.path.join(current, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            files.append({
                "path": os.path.relpath(path, root).replace(os.sep, "/"),
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).astimezone().isoformat(),
            })
    return files


def models_record(root, *, models_path, realization_id):
    return {
        "kind": "ai_toolkit_models",
        "schema_version": 1,
        "realization_id": realization_id,
        "observed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "models_path": models_path,
        "exists": os.path.isdir(root),
        "files": list_model_files(root),
        "loaded_candidate": "not-reported-by-trainer",
        "note": LOADED_CANDIDATE_NOTE,
    }


def write_models_record(root, record_path, *, models_path, realization_id):
    """Write the record once; an existing record is append-only and kept."""
    if os.path.exists(record_path):
        return False
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    temporary = record_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(models_record(root, models_path=models_path, realization_id=realization_id), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, record_path)
    return True


def main():
    models_path, record_path, realization_id = sys.argv[1], sys.argv[2], sys.argv[3]
    write_models_record(models_path, record_path, models_path=models_path, realization_id=realization_id)
    print(f"[kura] recorded AI-Toolkit model files under {models_path}", flush=True)


if __name__ == "__main__":
    main()
