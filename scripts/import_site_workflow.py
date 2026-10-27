"""Import a workflow from the maintainer's site, Comfy with ComfyUI, as a Kura workflow sample.

Development only: Kura never reads the site when it runs. The site publishes each workflow in
ComfyUI's UI form with its models and their download URLs (https://comfyui.nomadoor.net/workflows/index.json);
a sample needs the API form, which only ComfyUI's own frontend writes exactly. So an import has
two steps around one export by hand:

    uv run python scripts/import_site_workflow.py fetch <workflow URL on the site> --family <folder>
    # open the UI JSON it wrote in ComfyUI, then File > Export (API) to the path it printed
    uv run python scripts/import_site_workflow.py finish <family>/<name>_api.json --site-url <same URL>

`finish` checks the export against the UI form, writes the `.kura.yaml` sidecar from the site's
model list (pinning each Hugging Face repo to its current commit), and proposes `lora_insert`.
Ship the sample only after it has rendered once on Kura's ComfyUI image
(`src/kura/shipped/workflow-samples/README.md`).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kura.comfyui_models import COMFYUI_MODEL_DIRS, merged_registry, required_model_refs, resolve_model_specs  # noqa: E402
from kura.yamlio import dump_yaml  # noqa: E402

SITE_INDEX = "https://comfyui.nomadoor.net/workflows/index.json"
SAMPLES = REPO / "src" / "kura" / "shipped" / "workflow-samples"
# Loaders whose model output a LoRA loader can take: Kura inserts LoraLoaderModelOnly after one.
LORA_MODEL_LOADERS = ("UNETLoader", "CheckpointLoaderSimple")


def _get_json(url: str) -> Any:
    # The site refuses Python's default User-Agent; this script names itself.
    request = urllib.request.Request(url, headers={"User-Agent": "kura-import-site-workflow"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https hosts
        return json.loads(response.read().decode("utf-8"))


def site_entry(index: dict[str, Any], url: str) -> dict[str, Any]:
    """The site's index entry for a workflow URL."""
    for entry in index.get("workflows") or []:
        if isinstance(entry, dict) and entry.get("url") == url:
            return entry
    raise ValueError(f"{url} is not in the site's workflow list ({SITE_INDEX})")


def hf_file(url: str) -> dict[str, str]:
    """`repo`, `filename`, and `revision` of a Hugging Face `resolve` URL."""
    parsed = urllib.parse.urlparse(url)
    parts = [urllib.parse.unquote(part) for part in parsed.path.strip("/").split("/")]
    if parsed.netloc != "huggingface.co" or len(parts) < 5 or parts[2] != "resolve":
        raise ValueError(f"not a Hugging Face file URL: {url}")
    return {"repo": f"{parts[0]}/{parts[1]}", "revision": parts[3], "filename": "/".join(parts[4:])}


def hf_commit(repo: str, revision: str, *, get_json: Callable[[str], Any] = _get_json) -> str:
    """The commit a Hugging Face revision (a branch such as `main`) points to now."""
    info = get_json(f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}")
    sha = info.get("sha") if isinstance(info, dict) else None
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"Hugging Face did not name a commit for {repo}@{revision}")
    return sha


def api_problems(api: dict[str, Any], ui: dict[str, Any]) -> list[str]:
    """How an API export differs from the UI workflow it came from. Every API node must be a node
    of the UI workflow with the same type; the UI may have more (notes, reroutes, primitives are
    not exported)."""
    if not api or not all(isinstance(node, dict) and "class_type" in node for node in api.values()):
        return ["the API file is not a ComfyUI API export (File > Export (API)): every entry needs a class_type"]
    ui_types = {str(node.get("id")): node.get("type") for node in ui.get("nodes") or [] if isinstance(node, dict)}
    problems = []
    for node_id, node in api.items():
        if node_id not in ui_types:
            problems.append(f"API node {node_id} ({node['class_type']}) is not in the UI workflow")
        elif ui_types[node_id] != node["class_type"]:
            problems.append(f"API node {node_id} is {node['class_type']}, the UI node {ui_types[node_id]}")
    return problems


def lora_candidates(api: dict[str, Any]) -> list[str]:
    """The nodes a LoRA loader could follow: loaders whose output is the model."""
    return [node_id for node_id, node in api.items() if node.get("class_type") in LORA_MODEL_LOADERS]


def build_sidecar(api: dict[str, Any], entry: dict[str, Any], *, imported_on: str,
                  commit: Callable[[str, str], str]) -> tuple[dict[str, Any], list[str]]:
    """The sidecar for an API workflow from its site entry, and notes for the importer."""
    notes: list[str] = []
    if entry.get("custom_nodes"):
        raise ValueError("the site lists custom nodes for this workflow; samples use core ComfyUI nodes only")
    site_models = {model.get("name"): model for model in entry.get("models") or [] if isinstance(model, dict)}
    models: dict[str, dict[str, Any]] = {}
    commits: dict[tuple[str, str], str] = {}
    for ref in required_model_refs(api):
        site = site_models.pop(ref["name"], None)
        if not site or not site.get("url"):
            raise ValueError(f"the site gives no download URL for {ref['name']} ({ref['class_type']} node {ref['node']})")
        source = hf_file(site["url"])
        key = (source["repo"], source["revision"])
        if key not in commits:
            commits[key] = commit(*key)
        spec: dict[str, Any] = {"repo": source["repo"], "filename": source["filename"], "revision": commits[key]}
        if site.get("directory") and site["directory"] != COMFYUI_MODEL_DIRS.get(ref["type"], ref["type"]):
            spec["target_dir"] = site["directory"]
        models.setdefault(ref["type"], {})[ref["name"]] = spec
    for name in site_models:
        notes.append(f"the site lists {name}, but no loader Kura knows reads it (comfyui_models.MODEL_INPUTS); "
                     "RunPod will not download it")
    sidecar: dict[str, Any] = {
        "source": entry["url"],
        "provenance": f"Imported {imported_on} from the site's UI workflow (updated {entry.get('updated', 'unknown')}); "
                      "API form exported with the ComfyUI frontend.",
        "models": models,
    }
    candidates = lora_candidates(api)
    if len(candidates) == 1:
        sidecar["lora_insert"] = {"kind": "model_only", "model_node": candidates[0], "strength_model": 0.8}
    else:
        notes.append(f"lora_insert left out: model loaders {candidates or 'none'}; add it by hand if the sample takes a LoRA")
    return sidecar, notes


def check_sidecar(api: dict[str, Any], sidecar: dict[str, Any]) -> None:
    """Every model the workflow loads resolves through the sidecar, as a RunPod render would."""
    _, unknown = resolve_model_specs(api, merged_registry(sidecar.get("models") or {}))
    if unknown:
        raise ValueError("the sidecar does not resolve: " + ", ".join(ref["name"] for ref in unknown))


def cmd_fetch(args: argparse.Namespace) -> int:
    entry = site_entry(_get_json(SITE_INDEX), args.url)
    if entry.get("custom_nodes"):
        print(f"{args.url} uses custom nodes; samples use core ComfyUI nodes only", file=sys.stderr)
        return 1
    name = args.name or Path(urllib.parse.urlparse(args.url).path).stem
    ui_path = SAMPLES / args.family / f"{name}.json"
    if ui_path.exists():
        print(f"{ui_path.relative_to(REPO)} exists; remove it to import again", file=sys.stderr)
        return 1
    ui_path.parent.mkdir(parents=True, exist_ok=True)
    ui_path.write_text(json.dumps(_get_json(args.url), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    api_path = ui_path.with_name(f"{name}_api.json")
    print(f"wrote {ui_path.relative_to(REPO)}")
    print(f"next: open it in ComfyUI (the ref of Kura's ComfyUI image), File > Export (API) to {api_path.relative_to(REPO)},")
    print(f"then: uv run python scripts/import_site_workflow.py finish {api_path.relative_to(SAMPLES)} --site-url {args.url}")
    return 0


def cmd_finish(args: argparse.Namespace) -> int:
    api_path = SAMPLES / args.api
    if not api_path.name.endswith("_api.json"):
        print("name the API export <name>_api.json, beside <name>.json", file=sys.stderr)
        return 1
    ui_path = api_path.with_name(api_path.name.removesuffix("_api.json") + ".json")
    api = json.loads(api_path.read_text(encoding="utf-8"))
    ui = json.loads(ui_path.read_text(encoding="utf-8"))
    problems = api_problems(api, ui)
    if problems:
        print("the API export does not match the UI workflow:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    entry = site_entry(_get_json(SITE_INDEX), args.site_url)
    try:
        sidecar, notes = build_sidecar(api, entry, imported_on=date.today().isoformat(), commit=hf_commit)
        check_sidecar(api, sidecar)
    except ValueError as exc:
        print(f"cannot write the sidecar: {exc}", file=sys.stderr)
        return 1
    sidecar_path = api_path.with_suffix(".kura.yaml")
    sidecar_path.write_text(dump_yaml(sidecar), encoding="utf-8")
    print(f"wrote {sidecar_path.relative_to(REPO)}")
    for note in notes:
        print(f"note: {note}")
    print("next: render it once on Kura's ComfyUI image before shipping it")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download a site workflow's UI form into a sample folder")
    fetch.add_argument("url", help="The workflow's URL on the site, as its index lists it")
    fetch.add_argument("--family", required=True, help="The sample folder, such as moss-image")
    fetch.add_argument("--name", help="The file name without .json; default the site's")
    fetch.set_defaults(func=cmd_fetch)
    finish = sub.add_parser("finish", help="Check the API export and write its sidecar")
    finish.add_argument("api", help="<family>/<name>_api.json under the samples folder")
    finish.add_argument("--site-url", required=True, help="The workflow's URL on the site")
    finish.set_defaults(func=cmd_finish)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
