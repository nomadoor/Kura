"""One reclaim inventory in four classes by what deleting costs (docs/adr/reclaimable-data.md):
`kura cleanup plan`, `kura cleanup`, and `kura doctor disk` read the same owner, and nothing a user
made (outputs) is listed as reclaimable or removed unless the user names it."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from kura import runner
from kura.cli import build_parser, cmd_cleanup, cmd_doctor_disk, cmd_run_prune
from kura.images import PINNED_IMAGES
from kura.reclaim import reclaim_inventory
from kura.status_projection import project_status
from kura.storage import RECLAIM_POINTER

MIB = 1024**2
NO_DOCKER = {"daemon_reachable": False, "diagnosis": "Docker CLI was not found on PATH."}


def _write(path: Path, data: bytes | str = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def _run(root: Path, run_id: str, *, state: str = "completed", ended: str | None = None, repos: tuple[str, ...] = (),
         selector: str | None = None, image: str | None = None, continuation: str | None = None) -> Path:
    run_dir = root / "runs" / run_id
    (run_dir / "realizations").mkdir(parents=True)
    status: dict[str, Any] = {"state": state, "last_realization": "realizations/r1.json"}
    if ended:
        status["ended"] = ended
    _write(run_dir / "status.json", json.dumps(status))
    _write(run_dir / "realizations" / "r1.json", json.dumps({"id": "r1", "state": state, "launched_at": "2026-10-01T00:00:00+00:00"}))
    run = {"id": run_id, "type": "train"}
    if continuation:
        run["continuation"] = {"source": {"artifact_id": continuation}}
    _write(run_dir / "run.yaml", yaml.safe_dump(run))
    requirements = [{"role": "dit", "identity": {"kind": "huggingface-file", "repo_id": repo, "filename": "w.safetensors"}} for repo in repos]
    if selector:
        requirements.append({"role": "base_model", "identity": {"kind": "backend-resolved-selector", "value": selector}})
    _write(run_dir / "resolved" / "model-requirements.lock.yaml", yaml.safe_dump({"schema_version": 1, "requirements": requirements}))
    if image:
        _write(run_dir / "resolved" / "env.lock", yaml.safe_dump({"selected_image": image}))
    return run_dir


def _state(root: Path, artifact_id: str, source_run: str) -> Path:
    directory = root / "artifacts" / "training-state" / artifact_id
    _write(directory / "manifest.json", json.dumps({"schema_version": 1, "id": artifact_id, "source_run": source_run, "observed_step": 10}))
    _write(directory / "payload" / "model.safetensors", b"s" * 100)
    return directory


@contextlib.contextmanager
def _workspace(config: dict[str, Any] | None = None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        _write(root / "workspace.yaml", yaml.safe_dump({"schema_version": 2, **(config or {})}))
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(previous)


def _cleanup(argv: list[str]) -> tuple[int, str, str]:
    args = build_parser().parse_args(["cleanup", *argv])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch("kura.doctor._docker_storage_summary", return_value=NO_DOCKER):
        code = cmd_cleanup(args)
    return code, out.getvalue(), err.getvalue()


def _items(root: Path, **kwargs: Any) -> list[dict[str, Any]]:
    with patch("kura.doctor._docker_storage_summary", return_value=NO_DOCKER):
        return reclaim_inventory(root, yaml.safe_load((root / "workspace.yaml").read_text()), **kwargs)["items"]


class ReclaimInventoryTests(unittest.TestCase):
    def test_a_repository_two_runs_use_lists_both_and_the_newest_end(self) -> None:
        with _workspace() as root:
            _run(root, "a", ended="2026-10-01T10:00:00+00:00", repos=("org/model",))
            _run(root, "b", ended="2026-10-03T10:00:00+09:00", selector="org/model")
            _write(root / "cache" / "huggingface" / "hub" / "models--org--model" / "blobs" / "x", b"m" * 1000)
            _write(root / "cache" / "huggingface" / "hub" / "models--org--unused" / "blobs" / "y", b"u" * 10)
            items = {item["id"]: item for item in _items(root) if item["kind"] == "hf-repository"}
        self.assertEqual(items["org/model"]["class"], "download")
        self.assertEqual(items["org/model"]["referenced_by"], ["a", "b"])
        self.assertEqual(items["org/model"]["last_used"], "2026-10-03T10:00:00+09:00")
        self.assertEqual(items["org/unused"]["referenced_by"], [])
        self.assertIsNotNone(items["org/unused"]["last_used"])

    def test_nothing_a_user_made_is_reclaimable_by_default(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "done", ended="2026-10-01T00:00:00+00:00")
            _write(run_dir / "outputs" / "done-step00000010.safetensors", b"w" * 50)
            _write(run_dir / "outputs" / "done.safetensors", b"w" * 50)
            _write(run_dir / "outputs" / "samples" / "a.png")
            _write(run_dir / "downloads" / "remote.bin")
            _write(run_dir / "cache" / "latents.npz")
            _write(root / "datasets" / "tiny" / "a.png")
            items = _items(root)
            listed = {path for item in items for path in item["paths"]}
            self.assertEqual({item["class"] for item in items}, {"free"})
            self.assertTrue(all("outputs" not in path and "downloads" not in path and "datasets" not in path for path in listed))
            code, out, _ = _cleanup(["plan", "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(set(json.loads(out)["totals"]), {"free", "download", "resume"})
            # The free class is what `kura cleanup --yes` removes; outputs stay.
            self.assertEqual(_cleanup(["--yes"])[0], 0)
            self.assertFalse((run_dir / "cache").exists())
            self.assertTrue((run_dir / "outputs" / "done-step00000010.safetensors").is_file())
            self.assertTrue((run_dir / "downloads" / "remote.bin").is_file())
            # Asked for, outputs are listed with their checkpoints, still outside the totals.
            outputs = [item for item in _items(root, outputs=True) if item["class"] == "outputs"]
            self.assertEqual([entry["path"] for entry in outputs[0]["checkpoints"]], ["done-step00000010.safetensors", "done.safetensors"])

    def test_the_hugging_face_cache_outside_the_workspace_is_listed_and_removed_one_by_one(self) -> None:
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = Path(elsewhere).resolve() / "hf"
            _write(outside / "hub" / "models--org--one" / "blobs" / "a", b"1" * 10)
            _write(outside / "hub" / "models--org--two" / "blobs" / "b", b"2" * 10)
            _write(outside / "ai-toolkit-models" / "vae" / "vae.safetensors", b"v" * 10)
            with _workspace({"docker": {"hf_cache": str(outside)}}) as root:
                ids = {item["id"] for item in _items(root)}
                self.assertTrue({"org/one", "org/two", "ai-toolkit-models/vae/vae.safetensors"} <= ids)
                # `--yes` alone never touches a download.
                self.assertEqual(_cleanup(["--yes"])[0], 0)
                self.assertTrue((outside / "hub" / "models--org--one").is_dir())
                code, out, _ = _cleanup(["--models", "org/one"])
                self.assertEqual(code, 0)
                self.assertTrue(json.loads(out)["dry_run"])
                self.assertTrue((outside / "hub" / "models--org--one").is_dir())
                self.assertEqual(_cleanup(["--models", "org/one", "--yes"])[0], 0)
            self.assertFalse((outside / "hub" / "models--org--one").exists())
            self.assertTrue((outside / "hub" / "models--org--two").is_dir())
            self.assertTrue((outside / "ai-toolkit-models" / "vae" / "vae.safetensors").is_file())

    def test_ai_toolkit_model_files_are_referenced_by_the_runs_whose_record_lists_them(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "ait", ended="2026-10-02T00:00:00+00:00")
            _write(run_dir / "realizations" / "r1.ai-toolkit-models.json", json.dumps({
                "models_path": "/workspace/cache/huggingface/ai-toolkit-models", "files": [{"path": "flux/t5.safetensors"}],
            }))
            _write(root / "cache" / "huggingface" / "ai-toolkit-models" / "flux" / "t5.safetensors", b"t" * 10)
            _write(root / "cache" / "huggingface" / "ai-toolkit-models" / "flux" / "old.safetensors", b"o" * 10)
            items = {item["id"]: item for item in _items(root) if item["kind"] == "ai-toolkit-model-file"}
            self.assertEqual(items["ai-toolkit-models/flux/t5.safetensors"]["referenced_by"], ["ait"])
            self.assertEqual(items["ai-toolkit-models/flux/old.safetensors"]["referenced_by"], [])
            self.assertEqual(_cleanup(["--models", "unreferenced", "--yes"])[0], 0)
            self.assertTrue((root / "cache" / "huggingface" / "ai-toolkit-models" / "flux" / "t5.safetensors").is_file())
            self.assertFalse((root / "cache" / "huggingface" / "ai-toolkit-models" / "flux" / "old.safetensors").exists())

    def test_a_model_an_unfinished_run_uses_is_refused(self) -> None:
        with _workspace() as root:
            _run(root, "live", state="running", repos=("org/model",))
            _write(root / "cache" / "huggingface" / "hub" / "models--org--model" / "blobs" / "x")
            code, _, err = _cleanup(["--models", "org/model", "--yes"])
            self.assertEqual(code, 1)
            self.assertIn("in use by run(s) live", err)
            self.assertTrue((root / "cache" / "huggingface" / "hub" / "models--org--model").is_dir())
            code, _, err = _cleanup(["--models", "org/typo"])
            self.assertEqual(code, 1)
            self.assertIn("`kura cleanup plan` lists model ids", err)

    def test_an_old_image_is_old_and_the_pin_and_a_frozen_unfinished_one_are_not(self) -> None:
        pinned = PINNED_IMAGES["musubi-tuner"]
        repository = pinned.split("@")[0]
        old, frozen = "sha256:" + "1" * 64, "sha256:" + "2" * 64
        lines = [
            {"Repository": repository, "Tag": "<none>", "Digest": pinned.split("@")[1], "ID": "pin", "Size": "15GB"},
            {"Repository": repository, "Tag": "<none>", "Digest": old, "ID": "old", "Size": "14.5GB", "CreatedAt": "2026-09-01"},
            {"Repository": repository, "Tag": "<none>", "Digest": frozen, "ID": "frozen", "Size": "14GB"},
            {"Repository": "someone/else", "Tag": "latest", "Digest": "sha256:" + "3" * 64, "ID": "other", "Size": "1GB"},
        ]
        with _workspace() as root:
            _run(root, "queued", state="queued", image=f"{repository}@{frozen}")
            _run(root, "past", ended="2026-10-01T00:00:00+00:00", image=f"{repository}@{old}")
            calls: list[list[str]] = []

            def docker(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": []}),
                patch("kura.doctor._docker_json_lines", side_effect=lambda command: lines if command[:3] == ["docker", "image", "ls"] else []),
                patch("kura.cli.subprocess.run", side_effect=docker),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                items = [item for item in reclaim_inventory(root, {})["items"] if item["kind"] == "image"]
                code = cmd_cleanup(build_parser().parse_args(["cleanup", "--images", "old", "--yes"]))
        self.assertEqual([item["id"] for item in items], [f"{repository}@{old}"])
        self.assertEqual(items[0]["referenced_by"], ["past"])
        self.assertEqual(items[0]["last_used"], "2026-10-01T00:00:00+00:00")
        self.assertEqual(code, 0)
        self.assertIn(["docker", "image", "rm", f"{repository}@{old}"], calls)


class ReclaimApplyTests(unittest.TestCase):
    def test_resume_data_keeps_protected_states_and_every_lora(self) -> None:
        with _workspace() as root:
            source = _run(root, "source", ended="2026-10-01T00:00:00+00:00")
            _write(source / "outputs" / "source-step00000010.safetensors", b"w")
            _write(source / "outputs" / "source.safetensors", b"w")
            native = _write(source / "outputs" / "source-step00000010-state" / "optimizer.bin", b"o").parent
            kept = _state(root, "state-step-00000010-aaaaaaaaaaaa", "source")
            freed = _state(root, "state-step-00000005-bbbbbbbbbbbb", "source")
            _run(root, "continued", state="compiled", continuation=kept.name)
            code, out, _ = _cleanup(["--resume-data", "source"])
            self.assertEqual(code, 0)
            self.assertTrue(freed.is_dir())
            self.assertEqual(_cleanup(["--resume-data", "finished", "--yes"])[0], 0)
            self.assertFalse(freed.exists())
            self.assertFalse(native.exists())
            self.assertTrue(kept.is_dir())
            self.assertTrue((source / "outputs" / "source-step00000010.safetensors").is_file())
            self.assertTrue((source / "outputs" / "source.safetensors").is_file())
            records = list((source / "realizations").glob("r1.reclaim-*.json"))
            self.assertEqual(len(records), 1)
            facts = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual((facts["kind"], facts["action"]), ("reclaim", "remove-resume-data"))
            self.assertEqual(facts["removed_artifact_ids"], [freed.name])
            # Nothing is left to free for that run; naming it again says why.
            code, _, err = _cleanup(["--resume-data", "source"])
            self.assertEqual(code, 1)
            self.assertIn("has no resume data to free", err)

    def test_checkpoints_keep_exactly_the_named_files(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "pick", ended="2026-10-01T00:00:00+00:00")
            names = ["pick-step00000100.safetensors", "pick-step00000200.safetensors", "pick-step00000300.safetensors", "pick.safetensors"]
            for name in names:
                _write(run_dir / "outputs" / name, b"w")
            state = _write(run_dir / "outputs" / "pick-step00000300-state" / "model.safetensors", b"s")
            sample = _write(run_dir / "outputs" / "samples" / "a.png")
            code, _, err = _cleanup(["--checkpoints", "pick", "--keep", "pick-step00000999.safetensors"])
            self.assertEqual(code, 1)
            self.assertIn("has no checkpoint named 'pick-step00000999.safetensors'", err)
            code, _, err = _cleanup(["--checkpoints", "pick"])
            self.assertEqual(code, 1)
            self.assertIn("--keep", err)
            code, out, _ = _cleanup(["--checkpoints", "pick", "--keep", "pick-step00000200.safetensors", "pick.safetensors", "--yes"])
            self.assertEqual(code, 0)
            remaining = sorted(path.name for path in (run_dir / "outputs").glob("*.safetensors"))
            self.assertEqual(remaining, ["pick-step00000200.safetensors", "pick.safetensors"])
            self.assertTrue(state.is_file())
            self.assertTrue(sample.is_file())
            record = json.loads(next((run_dir / "realizations").glob("r1.reclaim-*.json")).read_text(encoding="utf-8"))
            self.assertEqual(record["action"], "remove-checkpoints")
            self.assertEqual(sorted(record["removed"]), ["outputs/pick-step00000100.safetensors", "outputs/pick-step00000300.safetensors"])

    def test_archive_moves_outputs_records_the_move_and_status_still_loads(self) -> None:
        with _workspace() as root, tempfile.TemporaryDirectory() as archive:
            run_dir = _run(root, "keep", ended="2026-10-01T00:00:00+00:00")
            _write(run_dir / "outputs" / "keep.safetensors", b"w" * 10)
            _write(run_dir / "downloads" / "remote.bin")
            before = project_status(run_dir)
            code, out, _ = _cleanup(["--archive-outputs", "keep", "--to", archive])
            self.assertEqual(code, 0)
            self.assertTrue((run_dir / "outputs").is_dir())
            self.assertEqual(_cleanup(["--archive-outputs", "keep", "--to", archive, "--yes"])[0], 0)
            destination = Path(archive).resolve() / "keep"
            self.assertTrue((destination / "outputs" / "keep.safetensors").is_file())
            self.assertTrue((destination / "downloads" / "remote.bin").is_file())
            self.assertFalse((run_dir / "outputs").exists())
            self.assertTrue((run_dir / "run.yaml").is_file())
            record = json.loads(next((run_dir / "realizations").glob("r1.reclaim-*.json")).read_text(encoding="utf-8"))
            self.assertEqual((record["action"], record["destination"], sorted(record["moved"])), ("archive-outputs", str(destination), ["downloads", "outputs"]))
            self.assertEqual(project_status(run_dir), before)
            code, _, err = _cleanup(["--archive-outputs", "keep", "--to", archive])
            self.assertEqual(code, 1)

    def test_doctor_and_cleanup_report_the_same_totals(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "done", ended="2026-10-01T00:00:00+00:00", repos=("org/model",))
            _write(run_dir / "tmp" / "scratch", b"t" * MIB)
            _write(root / "cache" / "huggingface" / "hub" / "models--org--model" / "blobs" / "x", b"m" * MIB)
            _state(root, "state-step-00000010-cccccccccccc", "done")
            _, plan, _ = _cleanup(["plan", "--json"])
            out = io.StringIO()
            with (
                patch("kura.doctor._docker_storage_summary", return_value=NO_DOCKER),
                patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                contextlib.redirect_stdout(out),
            ):
                cmd_doctor_disk(argparse.Namespace())
        doctor = json.loads(out.getvalue())
        totals = json.loads(plan)["totals"]
        self.assertEqual({key: value for key, value in doctor["reclaimable"].items() if key != "list"}, totals)
        self.assertEqual(doctor["reclaimable"]["list"], "kura cleanup plan")
        self.assertTrue(all(totals[cls]["count"] == 1 for cls in ("free", "download", "resume")))
        self.assertNotIn("advisories", doctor)


class FailClosedTests(unittest.TestCase):
    def _model(self, root: Path, repo: str = "org/model") -> Path:
        return _write(root / "cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "blobs" / "x").parent.parent

    def test_an_unreadable_status_keeps_the_models_its_run_names(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "broken", ended="2026-10-01T00:00:00+00:00", repos=("org/model",))
            (run_dir / "status.json").write_text("{not json", encoding="utf-8")
            model = self._model(root)
            code, _, err = _cleanup(["--models", "org/model", "--yes"])
            self.assertEqual(code, 1)
            self.assertIn("broken", err)
            self.assertTrue(model.is_dir())

    def test_an_unreadable_lock_refuses_every_model_and_image_removal(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "done", ended="2026-10-01T00:00:00+00:00")
            (run_dir / "resolved" / "model-requirements.lock.yaml").write_text("requirements: [unclosed", encoding="utf-8")
            model = self._model(root, "org/other")
            code, _, err = _cleanup(["--models", "org/other", "--yes"])
            self.assertEqual(code, 1)
            self.assertIn("cannot be read", err)
            self.assertTrue(model.is_dir())

    def test_a_compiled_run_uses_its_models(self) -> None:
        with _workspace() as root:
            _run(root, "next", state="compiled", repos=("org/model",))
            model = self._model(root)
            self.assertEqual(_cleanup(["--models", "org/model", "--yes"])[0], 1)
            self.assertTrue(model.is_dir())

    def test_unreferenced_waits_for_a_running_ai_toolkit_run(self) -> None:
        with _workspace() as root:
            _run(root, "ait", state="running", selector="black-forest-labs/FLUX.1-dev")
            downloading = _write(root / "cache" / "huggingface" / "ai-toolkit-models" / "flux" / "t5.safetensors")
            code, _, err = _cleanup(["--models", "unreferenced", "--yes"])
            self.assertEqual(code, 1)
            self.assertIn("ait", err)
            self.assertIn("after they finish", err)
            self.assertTrue(downloading.is_file())
            # Named by id, a model no running run's record names can still go.
            self.assertEqual(_cleanup(["--models", "ai-toolkit-models/flux/t5.safetensors", "--yes"])[0], 0)
            self.assertFalse(downloading.exists())

    def test_a_hub_entry_linked_outside_the_cache_is_refused(self) -> None:
        with _workspace() as root, tempfile.TemporaryDirectory() as elsewhere:
            target = _write(Path(elsewhere) / "models--org--linked" / "blobs" / "x").parent.parent
            hub = root / "cache" / "huggingface" / "hub"
            hub.mkdir(parents=True)
            (hub / "models--org--linked").symlink_to(target, target_is_directory=True)
            code, _, err = _cleanup(["--models", "org/linked", "--yes"])
            self.assertEqual(code, 1)
            self.assertIn("is a link to", err)
            self.assertTrue((target / "blobs" / "x").is_file())

    def test_a_run_launched_after_the_inventory_is_skipped(self) -> None:
        from kura.reclaim import became_in_use

        with _workspace() as root:
            run_dir = _run(root, "done", ended="2026-10-01T00:00:00+00:00", repos=("org/model",))
            self._model(root)
            item = next(item for item in _items(root) if item["id"] == "org/model")
            self.assertEqual(became_in_use(root, item), [])
            _write(run_dir / "status.json", json.dumps({"state": "running"}))
            self.assertEqual(became_in_use(root, item), ["done"])


class RunDoneTests(unittest.TestCase):
    """One rule decides whether a run's data may be offered or deleted, for the inventory and prune alike."""

    def test_a_pending_publication_or_a_live_pod_keeps_a_run(self) -> None:
        cases = {
            "docker-pending": {"state": "completed", "publication_state": "pending", "ended": "2026-10-01T00:00:00+00:00"},
            "runpod-live-pod": {"state": "interrupted", "pod_id": "pod-1", "ended": "2026-10-01T00:00:00+00:00"},
        }
        with _workspace() as root:
            for run_id, status in cases.items():
                run_dir = _run(root, run_id)
                _write(run_dir / "status.json", json.dumps({**status, "last_realization": "realizations/r1.json"}))
                _write(run_dir / "cache" / "x")
                _state(root, f"state-step-00000010-{run_id[:6]:0<12}", run_id)
            done = _run(root, "runpod-stopped")
            _write(done / "status.json", json.dumps({"state": "interrupted", "pod_id": "pod-2", "pod_stopped_at": "2026-10-01T01:00:00+00:00",
                                                     "ended": "2026-10-01T00:00:00+00:00", "last_realization": "realizations/r1.json"}))
            _write(done / "cache" / "x")
            listed = {(item["class"], item["id"]) for item in _items(root, outputs=True)}
            self.assertFalse(any(run_id in {item_id for _, item_id in listed} for run_id in cases))
            self.assertIn(("free", "runpod-stopped"), listed)
            for run_id, expected in (("docker-pending", "publication pending"), ("runpod-live-pod", "kura run stop runpod-live-pod")):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(cmd_run_prune(build_parser().parse_args(["run", "prune", run_id, "--yes"])), 1)
                self.assertIn(expected, err.getvalue())
                self.assertTrue((root / "runs" / run_id).is_dir())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_run_prune(build_parser().parse_args(["run", "prune", "runpod-stopped", "--yes"])), 0)
            self.assertFalse(done.exists())


class ApplyScopeTests(unittest.TestCase):
    def test_a_named_option_does_only_what_it_names(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "done", ended="2026-10-01T00:00:00+00:00")
            scratch = _write(run_dir / "tmp" / "x")
            _write(root / "cache" / "huggingface" / "hub" / "models--org--m" / "blobs" / "x")
            code, out, _ = _cleanup(["--models", "org/m"])
            payload = json.loads(out)
            self.assertEqual([item["action"] for item in payload["actions"]], ["models"])
            self.assertIn("hf_cache", payload)
            self.assertTrue(payload["hf_cache_inside_workspace"])
            self.assertEqual(_cleanup(["--models", "org/m", "--yes"])[0], 0)
            self.assertTrue(scratch.is_file())

    def test_doctor_asks_docker_for_its_usage_once(self) -> None:
        with _workspace():
            with (
                patch("kura.doctor._docker_storage_summary", return_value=NO_DOCKER) as summary,
                patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cmd_doctor_disk(argparse.Namespace())
        self.assertEqual(summary.call_count, 1)

    def test_run_prune_with_no_run_names_the_listing(self) -> None:
        with _workspace(), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cmd_run_prune(build_parser().parse_args(["run", "prune"])), 2)
        self.assertIn("kura cleanup plan --outputs", err.getvalue())

    def test_the_retired_runpod_download_floor_is_dropped_by_migrate_and_named_by_validate(self) -> None:
        from kura.workspace import migrate_workspace_config, validate_workspace_config

        with self.assertRaises(ValueError) as caught:
            validate_workspace_config({"runpod": {"download_min_free_gb": 50}})
        self.assertIn("docker.min_free_gb", str(caught.exception))
        migrated, notes = migrate_workspace_config({"schema_version": 2, "runpod": {"download_min_free_gb": 50}})
        self.assertNotIn("download_min_free_gb", migrated.get("runpod", {}))
        self.assertTrue(any("runpod.download_min_free_gb" in note for note in notes))


class RetiredSurfaceTests(unittest.TestCase):
    def test_retired_cleanup_targets_and_options_name_their_replacement(self) -> None:
        with _workspace():
            for target in ("cache", "runs", "docker-cache", "all"):
                code, _, err = _cleanup([target])
                self.assertEqual(code, 2)
                self.assertIn("kura cleanup plan", err)
        for argv, replacement in (
            (["cleanup", "--keep-last", "30"], "kura cleanup plan"),
            (["cleanup", "--delete-final-artifacts"], "kura run prune <run-id>"),
            (["run", "prune", "--keep", "30", "x"], "kura run prune <run-id>"),
            (["run", "prune", "--outputs-only", "x"], "--archive-outputs"),
            (["run", "prune", "--docker-containers", "x"], "kura cleanup --yes"),
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                build_parser().parse_args(argv)
            self.assertIn("was retired", err.getvalue())
            self.assertIn(replacement, err.getvalue())

    def test_run_prune_deletes_only_the_finished_runs_it_names(self) -> None:
        with _workspace() as root:
            old = _run(root, "old", ended="2026-01-01T00:00:00+00:00")
            other = _run(root, "other", ended="2026-01-01T00:00:00+00:00")
            live = _run(root, "live", state="running")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(cmd_run_prune(build_parser().parse_args(["run", "prune", "old", "live", "--yes"])), 1)
            self.assertIn("run 'live' is running", err.getvalue())
            self.assertTrue(old.is_dir())
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cmd_run_prune(build_parser().parse_args(["run", "prune", "old"])), 0)
            self.assertTrue(json.loads(out.getvalue())["dry_run"])
            self.assertTrue(old.is_dir())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_run_prune(build_parser().parse_args(["run", "prune", "old", "--yes"])), 0)
            self.assertFalse(old.exists())
            self.assertTrue(other.is_dir())
            self.assertTrue(live.is_dir())


class ReclaimPointerTests(unittest.TestCase):
    def test_the_disk_watch_stop_reason_names_the_listing(self) -> None:
        with _workspace() as root:
            run_dir = _run(root, "example", state="running")
            _write(run_dir / "resolved" / "manifest.lock.yaml", yaml.safe_dump({"type": "train"}))
            request = runner.write_launch_request(run_dir, executor="docker")
            with patch("kura.run_commands.plan.local_reserve_problem", return_value=("C: is low", [])), \
                    patch("kura.runner.write_stop_request") as stop, patch("kura.runner._log"):
                self.assertTrue(runner._disk_watch(root, run_dir, request, set()))
        self.assertTrue(stop.call_args.kwargs["reason"].endswith(RECLAIM_POINTER))

    def test_runpod_pulls_keep_the_local_reserve(self) -> None:
        from kura.run_commands import runpod_ssh

        with _workspace({"docker": {"min_free_gb": 7}}) as root:
            run_dir = _run(root, "remote")
            _write(run_dir / "resolved" / "manifest.lock.yaml", yaml.safe_dump({"type": "train", "safety": {"max_run_disk_gb": 9}}))
            self.assertEqual(runpod_ssh._local_reserve_bytes(run_dir, {"docker": {"min_free_gb": 7}}), 9 * 1024**3)
            self.assertEqual(runpod_ssh._local_reserve_bytes(root / "runs" / "missing", {"docker": {"min_free_gb": 7}}), 7 * 1024**3)


if __name__ == "__main__":
    unittest.main()
