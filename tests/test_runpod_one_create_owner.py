"""Every RunPod Pod Kura creates comes from one request builder and one create loop.

Training (`launch_runpod`) and render (`launch_runpod_session`) pass only what really
differs: the start script, the env, the image, the settings, and the facts their records
carry. `tests/runpod_create_golden.json` holds the request bodies and records both
launches wrote before they shared the owner; the tests here prove the owner writes the
same, apart from the changes listed in `CHANGED`.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from kura.executors import runpod
from kura.executors.runpod import RunPodAPIError
from kura.secrets import runpod_ssh_key
from tests.user_config import isolate_user_config

GOLDEN = Path(__file__).with_name("runpod_create_golden.json")
RID = "20261023-100000-000001"
NOW = "2026-10-23T10:00:00+09:00"
CONFIG = {
    "storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40", "NVIDIA L40S"], "gpu_type_priority": "custom",
    "cloud_types": ["COMMUNITY", "SECURE"], "ports": ["8675/http", "22/tcp"], "support_public_ip": True,
    "container_disk_gb": 120, "data_center_ids": ["EU-RO-1"],
}
CAPACITY = RunPodAPIError("RunPod GraphQL failed: There are no longer any instances available with the requested specifications", status_code=400)
SPEC = {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {"FOO": "bar"},
        "adapter_source": {"kind": "image", "reference": "registry/image:tag"}}
# Facts that differ by machine or by Kura build, never by the launch.
VOLATILE = ("kura_version", "kura_source", "container_secrets")


def _training_run(root: Path) -> Path:
    run_dir = root / "runs" / "train"
    (run_dir / "transfer").mkdir(parents=True)
    (run_dir / "transfer" / "stage.json").write_text(json.dumps(
        {"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled", "last_stage": "transfer/stage.json"}), encoding="utf-8")
    return run_dir


def _render_run(root: Path) -> Path:
    run_dir = root / "runs" / "render"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
    return run_dir


def _launch_training(run_dir: Path, config: dict[str, Any]) -> str | None:
    return runpod.launch_runpod(run_dir=run_dir, spec=SPEC, image="registry/image:tag", config=config, yes=True, max_lease_sec=3600)


def _launch_render(run_dir: Path, config: dict[str, Any]) -> str | None:
    return runpod.launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=config, purpose="comfyui-render",
                                        yes=True, max_lease_sec=3600)


LAUNCHES = {"training": (_training_run, _launch_training), "render": (_render_run, _launch_render)}


def _normalized(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in VOLATILE}


def capture(kind: str, outcome: str, config: dict[str, Any] = CONFIG) -> dict[str, Any]:
    """Launch once with RunPod mocked; the request bodies sent and the records written."""
    make_run, launch = LAUNCHES[kind]
    created = {"id": "pod-1", "name": f"kura-x-{RID}", "desiredStatus": "RUNNING", "costPerHr": 0.4}
    with tempfile.TemporaryDirectory() as directory:
        run_dir = make_run(Path(directory))
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                patch("kura.executors.runpod.time.sleep"), \
                patch("kura.executors.runpod._realization_id", return_value=RID), \
                patch("kura.executors.runpod._now", return_value=NOW), \
                patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "not measured", "candidates": []}), \
                patch("kura.executors.runpod._runpod_request", side_effect=CAPACITY if outcome == "failed" else None,
                      return_value=created) as request:
            try:
                launch(run_dir, config)
            except ValueError:
                pass
        realizations = run_dir / "realizations"
        events_path = run_dir / "logs" / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()] if events_path.exists() else []
        return {
            "sent": [call.args[3] for call in request.call_args_list],
            "intent": _normalized(json.loads((realizations / f"{RID}.create-intent.json").read_text(encoding="utf-8"))),
            "realization": _normalized(json.loads((realizations / f"{RID}.json").read_text(encoding="utf-8"))),
            "phases": [json.loads(line)["phase"] for line in (realizations / f"{RID}.phases.jsonl").read_text(encoding="utf-8").splitlines()],
            "events": [{key: value for key, value in event.items() if key != "timestamp"} for event in events],
        }


def capture_all() -> dict[str, Any]:
    return {f"{kind}/{outcome}": capture(kind, outcome) for kind in LAUNCHES for outcome in ("created", "failed")}


class OneCreateOwnerTests(unittest.TestCase):
    def test_request_bodies_and_records_match_the_golden_capture(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        for case, captured in capture_all().items():
            with self.subTest(case):
                self.assertEqual(json.loads(json.dumps(captured)), changed(case, golden[case]))

    def test_a_ports_list_without_ssh_gets_22_tcp_and_keeps_its_order(self) -> None:
        for ports, opened in ((["8888/http"], ["8888/http", "22/tcp"]), ([], ["22/tcp"]), (None, ["22/tcp"]),
                              (["22/tcp", "8888/http"], ["22/tcp", "8888/http"])):
            config = {key: value for key, value in CONFIG.items() if key != "ports"}
            if ports is not None:
                config["ports"] = ports
            for kind in LAUNCHES:
                with self.subTest(kind=kind, ports=ports):
                    captured = capture(kind, "created", config)
                    self.assertEqual(captured["sent"][0]["ports"], opened)
                    self.assertEqual(captured["realization"]["request"]["ports"], opened)

    def test_an_interrupt_during_the_create_hands_either_launch_over_as_unconfirmed(self) -> None:
        for kind, (make_run, launch) in LAUNCHES.items():
            with self.subTest(kind), tempfile.TemporaryDirectory() as directory:
                run_dir = make_run(Path(directory))
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                        patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "not measured", "candidates": []}), \
                        patch("kura.executors.runpod._runpod_request", side_effect=KeyboardInterrupt):
                    with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                        launch(run_dir, CONFIG)
                self.assertEqual(len(list((run_dir / "realizations").glob("*.create-unconfirmed.json"))), 1)
                self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")
                events = [json.loads(line)["event"] for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(events, ["runpod_create_unconfirmed"])

    def test_both_launches_create_through_the_one_builder_and_loop(self) -> None:
        for kind in LAUNCHES:
            with self.subTest(kind):
                with patch("kura.executors.runpod._pod_create_request", wraps=runpod._pod_create_request) as build, \
                        patch("kura.executors.runpod._create_pod", wraps=runpod._create_pod) as create:
                    capture(kind, "created")
                build.assert_called_once()
                create.assert_called_once()


# What every Pod ran to let SSH in before Kura sent its own key: only the account keys in PUBLIC_KEY.
ACCOUNT_KEYS_ONLY = r'''
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" | grep '^ssh-' > /root/.ssh/authorized_keys || true
  chmod 600 /root/.ssh/authorized_keys
fi
'''.lstrip("\n")
KURA_KEY_AND_ACCOUNT_KEYS = r'''
printf '%s\n' "${KURA_SSH_PUBLIC_KEY:-}" "${PUBLIC_KEY:-}" | grep '^ssh-' > /root/.ssh/authorized_keys || true
chmod 600 /root/.ssh/authorized_keys
'''.lstrip("\n")


def _with_kura_ssh_key(request: dict[str, Any]) -> None:
    """Kura sends its own SSH public key with every Pod and lets it in beside the account keys."""
    request["env"] = {**request["env"], "KURA_SSH_PUBLIC_KEY": runpod_ssh_key()[1].read_text(encoding="utf-8").strip()}
    script = request["dockerStartCmd"][2]
    assert script.count(ACCOUNT_KEYS_ONLY) == 1
    request["dockerStartCmd"][2] = script.replace(ACCOUNT_KEYS_ONLY, KURA_KEY_AND_ACCOUNT_KEYS)


def changed(case: str, before: dict[str, Any]) -> dict[str, Any]:
    """The golden capture with what the one owner changed on purpose: each was a difference
    between the two loops that training and render do not really need."""
    realization = before["realization"]
    for request in [*before["sent"], before["intent"]["request"], realization["request"]]:
        _with_kura_ssh_key(request)
    if case == "render/created":
        # The render loop never recorded when its Pod was created.
        before["phases"] = [*before["phases"], "pod_created"]
    if case == "render/failed":
        # The render loop recorded no failure classification and no workspace contract on a failure.
        for attempt in realization["request"]["launch_attempts"]:
            attempt["classification"] = "capacity"
        realization["workspace_contract"] = "Thin RunPod session; Kura connects over SSH tunnel and records render artifacts locally"
    if case == "training/failed":
        # The launch's facts are recorded whatever its outcome.
        realization["transfer"] = {"archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}
    return before


def setUpModule() -> None:
    isolate_user_config()


if __name__ == "__main__":
    unittest.main()
