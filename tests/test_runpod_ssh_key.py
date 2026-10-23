"""RunPod works with RUNPOD_API_KEY alone: Kura reaches every Pod with its own SSH key.

The key is made once per user beside the user secrets file (`kura.secrets.runpod_ssh_key`),
every Pod gets its public half in `KURA_SSH_PUBLIC_KEY` and lets it in through one start
block (`_pod_ssh_setup_shell`), and the Pod's SSH address is read from RunPod's API
(`runpod_ssh_endpoint`), so no local `runpodctl` and no key in the RunPod account is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from kura import secrets
from kura.executors import runpod
from kura.executors.runpod import RunPodAPIError
from kura.run_commands.runpod_ssh import _runpod_ssh_details
from kura.secrets import is_secret_name, runpod_ssh_key
from tests.platform_support import posix_only
from tests.test_runpod_one_create_owner import CONFIG, LAUNCHES, capture
from tests.user_config import isolate_user_config

HAS_SSH_KEYGEN = shutil.which("ssh-keygen") is not None


def setUpModule() -> None:
    isolate_user_config()


@unittest.skipUnless(HAS_SSH_KEYGEN, "needs the OpenSSH client")
class KeyOwnerTests(unittest.TestCase):
    def test_the_key_is_made_once_beside_the_user_secrets_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.secrets.user_secrets_path", return_value=Path(directory) / "secrets.env"), \
                patch("kura.secrets.subprocess.run", wraps=subprocess.run) as run:
            private, public = runpod_ssh_key()
            first = public.read_text(encoding="utf-8")
            self.assertEqual(runpod_ssh_key(), (private, public))
            self.assertEqual(public.read_text(encoding="utf-8"), first)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(private.parent, Path(directory))
            self.assertTrue(first.startswith("ssh-ed25519 "))

    def test_a_half_written_pair_is_made_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.secrets.user_secrets_path", return_value=Path(directory) / "secrets.env"):
            private, public = runpod_ssh_key()
            public.unlink()
            self.assertEqual(runpod_ssh_key(), (private, public))
            self.assertTrue(public.is_file())

    @posix_only("file modes")
    def test_the_private_key_is_readable_only_by_the_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.secrets.user_secrets_path", return_value=Path(directory) / "secrets.env"):
            private, _ = runpod_ssh_key()
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)


class KeyOwnerWithoutOpenSSHTests(unittest.TestCase):
    def test_a_missing_ssh_keygen_says_to_install_the_openssh_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.secrets.user_secrets_path", return_value=Path(directory) / "secrets.env"), \
                patch("kura.secrets.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "install the OpenSSH client"):
                runpod_ssh_key()


@unittest.skipUnless(HAS_SSH_KEYGEN, "needs the OpenSSH client")
class EveryPodGetsTheKeyTests(unittest.TestCase):
    def test_the_public_key_is_not_a_secret_name(self) -> None:
        self.assertFalse(is_secret_name("KURA_SSH_PUBLIC_KEY"))

    def test_both_launches_send_the_public_key_and_the_one_ssh_start_block(self) -> None:
        public = runpod_ssh_key()[1].read_text(encoding="utf-8").strip()
        for kind in LAUNCHES:
            with self.subTest(kind), patch("kura.executors.runpod._pod_ssh_setup_shell", wraps=runpod._pod_ssh_setup_shell) as setup:
                captured = capture(kind, "created")
                setup.assert_called_once_with()
                sent = captured["sent"][0]
                self.assertEqual(sent["env"]["KURA_SSH_PUBLIC_KEY"], public)
                self.assertIn(runpod._pod_ssh_setup_shell(), sent["dockerStartCmd"][2])
                # Records show the public key as it is: it is not a secret.
                self.assertEqual(captured["intent"]["request"]["env"]["KURA_SSH_PUBLIC_KEY"], public)
                self.assertEqual(captured["realization"]["request"]["env"]["KURA_SSH_PUBLIC_KEY"], public)

    def test_the_start_block_lets_in_kuras_key_and_keeps_the_account_keys(self) -> None:
        script = runpod._pod_ssh_setup_shell()
        self.assertIn('"${KURA_SSH_PUBLIC_KEY:-}" "${PUBLIC_KEY:-}"', script)
        self.assertIn("> /root/.ssh/authorized_keys", script)

    def test_no_launch_writes_the_private_key_anywhere_it_records(self) -> None:
        private = runpod_ssh_key()[0].read_text(encoding="utf-8").strip()
        body = "".join(line for line in private.splitlines() if not line.startswith("-----"))
        for kind, (make_run, launch) in LAUNCHES.items():
            with self.subTest(kind), tempfile.TemporaryDirectory() as directory:
                run_dir = make_run(Path(directory))
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                        patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "not measured", "candidates": []}), \
                        patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}), \
                        patch("sys.stdout"), patch("sys.stderr"):
                    launch(run_dir, CONFIG)
                for path in Path(directory).rglob("*"):
                    if path.is_file():
                        text = path.read_text(encoding="utf-8", errors="replace")
                        self.assertNotIn("PRIVATE KEY", text, path)
                        self.assertNotIn(body[:40], text, path)


def _pod(ports: Any, uptime: int = 30) -> dict[str, Any]:
    return {"id": "pod-1", "desiredStatus": "RUNNING", "runtime": None if ports is None else {"uptimeInSeconds": uptime, "ports": ports}}


PRIVATE_22 = {"ip": "10.0.0.5", "isIpPublic": False, "privatePort": 22, "publicPort": 22, "type": "tcp"}
PUBLIC_HTTP = {"ip": "100.65.0.1", "isIpPublic": False, "privatePort": 8675, "publicPort": 60001, "type": "http"}
PUBLIC_22 = {"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 22, "publicPort": 40022, "type": "tcp"}


class SshEndpointTests(unittest.TestCase):
    def test_the_public_tcp_22_mapping_is_the_endpoint(self) -> None:
        self.assertEqual(runpod.runpod_ssh_endpoint(_pod([PRIVATE_22, PUBLIC_HTTP, PUBLIC_22])), ("203.0.113.7", 40022))

    def test_no_runtime_or_only_private_mappings_is_not_ready(self) -> None:
        self.assertIsNone(runpod.runpod_ssh_endpoint(_pod(None)))
        self.assertIsNone(runpod.runpod_ssh_endpoint(_pod([PRIVATE_22, PUBLIC_HTTP])))
        self.assertIsNone(runpod.runpod_ssh_endpoint({"id": "pod-1"}))

    def test_the_pod_lookup_asks_runpod_for_the_port_mappings(self) -> None:
        with patch("kura.executors.runpod._runpod_graphql", return_value={"pod": _pod([PUBLIC_22])}) as graphql:
            runpod._runpod_pod_or_none("pod-1", "api-secret")
        query = graphql.call_args.args[0]
        self.assertIn("runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }", query)


class SshDetailsTests(unittest.TestCase):
    def _details(self, answers: list[Any], *, timeout_sec: int = 5) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch("kura.run_commands.runpod_ssh.runpod_ssh_key", return_value=(Path("/keys/runpod_ssh_key"), Path("/keys/runpod_ssh_key.pub"))), \
                    patch("kura.executors.runpod._runpod_request", side_effect=answers) as request, \
                    patch("kura.run_commands.runpod_ssh.sleep_checking_stop"), \
                    patch("shutil.which", return_value=None):
                details = _runpod_ssh_details(run_dir, timeout_sec=timeout_sec, interval_sec=0)
            self.assertTrue(all(call.args[:2] == ("GET", "/pods/pod-1") for call in request.call_args_list))
            return details

    def test_the_public_ssh_address_and_kuras_key_are_used(self) -> None:
        details = self._details([_pod([PRIVATE_22, PUBLIC_22])])
        self.assertEqual((details["ip"], details["port"], details["key"]), ("203.0.113.7", 40022, str(Path("/keys/runpod_ssh_key"))))
        self.assertIsInstance(details["container_started_at"], str)

    def test_a_pod_still_starting_is_asked_again(self) -> None:
        details = self._details([_pod(None), ValueError("RunPod API is unreachable: timed out"), _pod([PRIVATE_22]), _pod([PUBLIC_22])])
        self.assertEqual(details["port"], 40022)

    def test_a_pod_runpod_does_not_know_is_gone(self) -> None:
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            self._details([RunPodAPIError("RunPod Pod not found", status_code=404)])

    def test_a_refused_key_is_raised_at_once(self) -> None:
        with self.assertRaises(RunPodAPIError):
            self._details([RunPodAPIError("RunPod GraphQL failed (401): unauthorized", status_code=401)])

    def test_a_pod_without_a_public_mapping_times_out_saying_so(self) -> None:
        with patch("kura.run_commands.runpod_ssh.time.monotonic", side_effect=[0, 0, 0, 10]):
            with self.assertRaisesRegex(ValueError, "no public IP and SSH port mapping"):
                self._details([_pod([PRIVATE_22]), _pod([PRIVATE_22])], timeout_sec=5)


if __name__ == "__main__":
    unittest.main()
