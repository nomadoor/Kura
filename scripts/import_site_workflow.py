"""Import a workflow from the maintainer's site, Comfy with ComfyUI, as a Kura workflow sample.

Development only: Kura never reads the site when it runs. The site publishes each workflow in
ComfyUI's UI form with its models and their download URLs (https://comfyui.nomadoor.net/workflows/index.json);
a sample needs the API form, which only ComfyUI's own frontend writes exactly. So an import is:

    uv run python scripts/import_site_workflow.py fetch <workflow URL on the site> --family <folder>
    uv run --with playwright python scripts/import_site_workflow.py export <family>/<name>.json --comfyui <URL>
    uv run python scripts/import_site_workflow.py finish <family>/<name>_api.json --site-url <same URL>

`export` opens the ComfyUI frontend that a running ComfyUI serves in a headless browser, loads the
UI form, and saves what the frontend's own `graphToPrompt` produces: the same API form as
File > Export (API) by hand (checked on the MOSS-Image sample). Exporting by hand works too.

`finish` checks the export against the UI form, writes the `.kura.yaml` sidecar from the site's
model list (pinning each Hugging Face repo to its current commit), and proposes `lora_insert`.
The maintainer ran every site workflow before publishing it, so the sample counts as rendered
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
# Loaders whose model output a LoRA loader can take. After a checkpoint loader, which also loads the
# text encoder, Kura inserts LoraLoader on both (as the SD1.5 and SDXL samples do); after a
# diffusion-model loader, LoraLoaderModelOnly.
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
    ui_nodes = {str(node.get("id")): node for node in ui.get("nodes") or [] if isinstance(node, dict)}
    problems = []
    for node_id, node in api.items():
        if node_id not in ui_nodes:
            problems.append(f"API node {node_id} ({node['class_type']}) is not in the UI workflow")
            continue
        if ui_nodes[node_id].get("type") != node["class_type"]:
            problems.append(f"API node {node_id} is {node['class_type']}, the UI node {ui_nodes[node_id].get('type')}")
            continue
        # A hand export applies dynamic-prompt syntax to text with braces; the text must stay as written.
        widgets = ui_nodes[node_id].get("widgets_values")
        texts = {value for value in node.get("inputs", {}).values() if isinstance(value, str)}
        for value in widgets if isinstance(widgets, list) else []:
            if isinstance(value, str) and "{" in value and value not in texts:
                problems.append(f"API node {node_id} ({node['class_type']}) changed a text with braces (dynamic-prompt syntax); "
                                "export again with this script's export, which keeps it")
    return problems


def site_model_names(api: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Rename models the workflow names inside its author's own folders (`Flux.1\\flux1-dev.safetensors`)
    to the file names the site lists, in place; one line per renamed model."""
    listed = {model.get("name") for model in entry.get("models") or [] if isinstance(model, dict)}
    renamed = []
    for ref in required_model_refs(api):
        base = ref["name"].replace("\\", "/").rsplit("/", 1)[-1]
        if ref["name"] not in listed and base in listed:
            api[ref["node"]]["inputs"][ref["input"]] = base
            renamed.append(f"{ref['name']} -> {base}")
    return renamed


def lora_candidates(api: dict[str, Any]) -> list[str]:
    """The nodes a LoRA loader could follow: loaders whose output is the model."""
    return [node_id for node_id, node in api.items() if node.get("class_type") in LORA_MODEL_LOADERS]


def build_sidecar(api: dict[str, Any], entry: dict[str, Any], *, imported_on: str,
                  commit: Callable[[str, str], str], lora_node: str | None = None) -> tuple[dict[str, Any], list[str]]:
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
    if lora_node is not None:
        if lora_node not in candidates:
            raise ValueError(f"--lora-node {lora_node} is not a model loader of this workflow: {candidates or 'none'}")
        candidates = [lora_node]
    if len(candidates) == 1 and api[candidates[0]]["class_type"] == "CheckpointLoaderSimple":
        sidecar["lora_insert"] = {"kind": "model_clip", "model_node": candidates[0], "clip_node": candidates[0],
                                  "strength_model": 0.8, "strength_clip": 0.8}
    elif len(candidates) == 1:
        sidecar["lora_insert"] = {"kind": "model_only", "model_node": candidates[0], "strength_model": 0.8}
    else:
        notes.append(f"lora_insert left out: model loaders {candidates or 'none'}; name the one the LoRA follows with --lora-node")
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
    print(f"next: uv run --with playwright python scripts/import_site_workflow.py export {ui_path.relative_to(SAMPLES)} --comfyui <URL>")
    print(f"      (or File > Export (API) by hand to {api_path.relative_to(REPO)}),")
    print(f"then: uv run python scripts/import_site_workflow.py finish {api_path.relative_to(SAMPLES)} --site-url {args.url}")
    return 0


# Run in the ComfyUI frontend: load a UI workflow and convert it as File > Export (API) does, except
# that text widgets keep their text. The frontend treats `{a|b}` in a prompt widget as a dynamic
# prompt (one choice at random, braces removed), which breaks a JSON prompt such as Ideogram 4's.
_EXPORT_SCRIPT = """async (workflow) => {
    await window.app.loadGraphData(workflow);
    const output = (await window.app.graphToPrompt()).output;
    for (const node of window.app.graph._nodes) {
        const inputs = output[String(node.id)]?.inputs;
        for (const widget of node.widgets ?? []) {
            if (inputs && typeof widget.value === "string" && widget.value.includes("{") && widget.name in inputs) {
                inputs[widget.name] = widget.value;
            }
        }
    }
    return output;
}"""


def cmd_export(args: argparse.Namespace) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("export needs Playwright: run it with `uv run --with playwright python scripts/import_site_workflow.py export ...`", file=sys.stderr)
        return 1
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(args.comfyui, wait_until="domcontentloaded")
        page.wait_for_function("() => window.app && typeof window.app.graphToPrompt === 'function'", timeout=120_000)
        for name in args.ui:
            ui_path = SAMPLES / name
            api_path = ui_path.with_name(ui_path.stem + "_api.json")
            api = page.evaluate(_EXPORT_SCRIPT, json.loads(ui_path.read_text(encoding="utf-8")))
            api_path.write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {api_path.relative_to(REPO)}")
        browser.close()
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
    renamed = site_model_names(api, entry)
    try:
        sidecar, notes = build_sidecar(api, entry, imported_on=date.today().isoformat(), commit=hf_commit, lora_node=args.lora_node)
        check_sidecar(api, sidecar)
    except ValueError as exc:
        print(f"cannot write the sidecar: {exc}", file=sys.stderr)
        return 1
    if renamed:
        api_path.write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        notes = [f"model renamed to the site's file name: {line}" for line in renamed] + notes
    sidecar_path = api_path.with_suffix(".kura.yaml")
    sidecar_path.write_text(dump_yaml(sidecar), encoding="utf-8")
    print(f"wrote {sidecar_path.relative_to(REPO)}")
    for note in notes:
        print(f"note: {note}")
    print("next: commit the UI form, the API form, and the sidecar together")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download a site workflow's UI form into a sample folder")
    fetch.add_argument("url", help="The workflow's URL on the site, as its index lists it")
    fetch.add_argument("--family", required=True, help="The sample folder, such as moss-image")
    fetch.add_argument("--name", help="The file name without .json; default the site's")
    fetch.set_defaults(func=cmd_fetch)
    export = sub.add_parser("export", help="Export the API form of UI workflows with a running ComfyUI's frontend")
    export.add_argument("ui", nargs="+", help="<family>/<name>.json under the samples folder")
    export.add_argument("--comfyui", required=True, help="The ComfyUI to use, such as http://127.0.0.1:8188")
    export.set_defaults(func=cmd_export)
    finish = sub.add_parser("finish", help="Check the API export and write its sidecar")
    finish.add_argument("api", help="<family>/<name>_api.json under the samples folder")
    finish.add_argument("--site-url", required=True, help="The workflow's URL on the site")
    finish.add_argument("--lora-node", help="The model loader a LoRA follows, when the workflow has more than one")
    finish.set_defaults(func=cmd_finish)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
