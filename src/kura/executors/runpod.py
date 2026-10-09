"""RunPod executor."""

from __future__ import annotations

import http.client
import json
import os
import platform
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from kura import __version__
from kura.images import runpod_min_cuda_version, runpod_min_cuda_for
from kura.install_source import kura_provenance
from kura.dataset_handoff import inspect_dataset_sources, load_frozen_dataset_handoff, handoff_was_frozen
from kura.dataset_transfer import build_transfer_inventory, estimate_transfer, pin_transfer_manifest, write_transfer_archive, write_transfer_manifest
from kura.provenance import image_reference_identity
from kura.training_artifacts import resume_artifact_directory
from kura.runtime_io import validated_write_roots
from kura.secrets import MissingSecret, missing
from kura.executors.common import PROGRESS_FIELDS, kura_container_env, CONTAINER_WORKSPACE, sleep_checking_stop, CREATE_INTENT_SUFFIX, TERMINAL_STATES, append_capacity_wait, settle_status_from_realization, unresolved_create_intents, write_create_unconfirmed, write_stop_record, _event_exists, append_run_event, dataset_input_drift_warning, _is_secret, _load_status, _materialize_stdout_progress, _mutate_run_status, _now, _realization_id, _redact_secret_text, _run_operation_lock, _safe_env, _write_json, _write_observation, _write_status, record_launch_phase
from kura.container_scripts import script_source
from kura.records import record as as_record


# Defines kura_pod_self_delete for every Pod-side guard (unattended wait and
# maximum lease, training and render Pods).
POD_SELF_DELETE_FUNCTION = script_source("pod_self_delete.sh")


class RunPodAPIError(ValueError):
    """A RunPod HTTP error with its status preserved for retry policy."""

    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _runpod_graphql(query: str, variables: dict[str, Any], api_key: str, *, timeout: float = 30.0) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        "https://api.runpod.io/graphql",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"Kura/{__version__}",
        },
    )
    request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = _redact_secret_text(exc.read().decode("utf-8", errors="replace"))
        raise RunPodAPIError(f"RunPod GraphQL failed ({exc.code}): {detail}", status_code=exc.code) from exc
    except URLError as exc:
        raise ValueError(f"RunPod API is unreachable: {exc.reason}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # A read that times out or breaks after the request was sent; the
        # request may still have taken effect.
        raise ValueError(f"RunPod API is unreachable: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RunPod API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("RunPod GraphQL returned an unexpected response")
    errors = value.get("errors")
    if isinstance(errors, list) and errors:
        messages = [str(item.get("message")) for item in errors if isinstance(item, dict) and item.get("message")]
        raise RunPodAPIError(
            "RunPod GraphQL failed: " + _redact_secret_text("; ".join(messages) or str(errors)),
            status_code=400,
        )
    data = value.get("data")
    if not isinstance(data, dict):
        raise ValueError("RunPod GraphQL response did not contain data")
    return data


def _runpod_graphql_create_input(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("interruptible"):
        raise ValueError("RunPod interruptible Pod creation is not supported by Kura's GraphQL control plane")
    gpu_type_ids = payload.get("gpuTypeIds")
    if not isinstance(gpu_type_ids, list) or len(gpu_type_ids) != 1 or not isinstance(gpu_type_ids[0], str):
        raise ValueError("RunPod GraphQL Pod creation requires exactly one gpuTypeIds entry")
    result: dict[str, Any] = {
        "gpuTypeId": gpu_type_ids[0],
        "gpuCount": payload.get("gpuCount", 1),
        "containerDiskInGb": payload.get("containerDiskInGb", 50),
        "volumeInGb": payload.get("volumeInGb", 0),
        "startSsh": True,
    }
    for source, target in (
        ("name", "name"),
        ("cloudType", "cloudType"),
        ("imageName", "imageName"),
        ("templateId", "templateId"),
        ("supportPublicIp", "supportPublicIp"),
        ("volumeMountPath", "volumeMountPath"),
        ("networkVolumeId", "networkVolumeId"),
        ("minCudaVersion", "minCudaVersion"),
    ):
        if payload.get(source) is not None:
            result[target] = payload[source]
    ports = payload.get("ports")
    if isinstance(ports, list):
        result["ports"] = ",".join(str(item) for item in ports)
    env = payload.get("env")
    if isinstance(env, dict):
        result["env"] = [{"key": str(key), "value": str(value)} for key, value in sorted(env.items())]
    start_command = payload.get("dockerStartCmd")
    if isinstance(start_command, list) and all(isinstance(item, str) for item in start_command):
        result["dockerArgs"] = shlex.join(start_command)
    data_center_ids = payload.get("dataCenterIds")
    if isinstance(data_center_ids, list) and data_center_ids:
        if len(data_center_ids) != 1:
            raise ValueError("RunPod GraphQL Pod creation requires at most one dataCenterIds entry per attempt")
        result["dataCenterId"] = str(data_center_ids[0])
    country_codes = payload.get("countryCodes")
    if isinstance(country_codes, list) and country_codes:
        if len(country_codes) != 1:
            raise ValueError("RunPod GraphQL Pod creation requires at most one countryCodes entry per attempt")
        result["countryCode"] = str(country_codes[0])
    return result


def _runpod_request(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
    if method == "POST" and path == "/pods" and isinstance(payload, dict):
        query = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id name imageName desiredStatus costPerHr containerDiskInGb volumeInGb
            volumeMountPath gpuCount memoryInGb vcpuCount ports lastStatusChange env
            machine { id dataCenterId gpuDisplayName location }
          }
        }
        """
        data = _runpod_graphql(query, {"input": _runpod_graphql_create_input(payload)}, api_key, timeout=timeout)
        pod = data.get("podFindAndDeployOnDemand")
        if not isinstance(pod, dict):
            raise ValueError("RunPod GraphQL create response did not contain a Pod")
        return pod
    if path.startswith("/pods/"):
        pod_id = path.removeprefix("/pods/")
        if not pod_id or "/" in pod_id:
            raise ValueError(f"unsupported RunPod API path: {path}")
        if method == "GET":
            query = """
            query getPod($podId: String!) {
              pod(input: {podId: $podId}) {
                id name imageName desiredStatus costPerHr containerDiskInGb volumeInGb
                volumeMountPath gpuCount memoryInGb vcpuCount ports lastStatusChange
                machine { id dataCenterId gpuDisplayName location }
                runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
              }
            }
            """
            data = _runpod_graphql(query, {"podId": pod_id}, api_key, timeout=timeout)
            pod = data.get("pod")
            if not isinstance(pod, dict):
                raise RunPodAPIError("RunPod Pod not found", status_code=404)
            return pod
        if method == "DELETE":
            query = """
            mutation terminatePod($podId: String!) {
              podTerminate(input: {podId: $podId})
            }
            """
            _runpod_graphql(query, {"podId": pod_id}, api_key, timeout=timeout)
            return {}
    raise ValueError(f"unsupported RunPod API operation: {method} {path}")


def runpod_gpu_availability(config: dict[str, Any], gpu_type_ids: list[str], *, min_cuda_version: str | None = None) -> dict[str, Any]:
    """Measure current RunPod stock and price for an ordered GPU candidate list.

    `min_cuda_version` limits the measurement to hosts a Pod for the image could land on.
    """

    settings = _runpod_settings(config)
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        return {"status": "unavailable", "reason": missing(settings["api_key_env"], "needed to list RunPod GPUs"), "candidates": []}
    aliases: list[str] = []
    cuda_filter = f", minCudaVersion: {json.dumps(min_cuda_version)}" if min_cuda_version else ""
    for index, gpu_type_id in enumerate(gpu_type_ids):
        cloud_fields = []
        for cloud_type in settings["cloud_types"]:
            secure = "true" if cloud_type == "SECURE" else "false"
            cloud_fields.append(
                f'{cloud_type.lower()}: lowestPrice(input: {{gpuCount: {settings["gpu_count"]}, secureCloud: {secure}{cuda_filter}}}) '
                "{ stockStatus uninterruptablePrice availableGpuCounts }"
            )
        aliases.append(
            f'g{index}: gpuTypes(input: {{id: {json.dumps(gpu_type_id)}}}) '
            "{ id displayName memoryInGb " + " ".join(cloud_fields) + " }"
        )
    query = "query KuraGpuAvailability { " + " ".join(aliases) + " }"
    body = json.dumps({"query": query}).encode("utf-8")
    request = Request(
        "https://api.runpod.io/graphql",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"Kura/{__version__}",
        },
    )
    request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _redact_secret_text(exc.read().decode("utf-8", errors="replace"))
        kind = "auth" if exc.code in {401, 403} else "rate_limit" if exc.code == 429 else "transient" if exc.code >= 500 else "request"
        return {"status": "unavailable", "error_kind": kind, "reason": f"RunPod GraphQL failed ({exc.code}): {detail}", "candidates": []}
    except (URLError, OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "error_kind": "transient", "reason": _redact_secret_text(str(exc)), "candidates": []}
    errors = payload.get("errors") if isinstance(payload, dict) else None
    data = payload.get("data") if isinstance(payload, dict) else None
    if errors or not isinstance(data, dict):
        return {"status": "unavailable", "error_kind": "request", "reason": _redact_secret_text(str(errors or "unexpected RunPod response")), "candidates": []}
    candidates: list[dict[str, Any]] = []
    for index, requested_id in enumerate(gpu_type_ids):
        matches = data.get(f"g{index}")
        item = matches[0] if isinstance(matches, list) and matches and isinstance(matches[0], dict) else {}
        clouds = []
        for cloud_type in settings["cloud_types"]:
            price = item.get(cloud_type.lower()) if isinstance(item.get(cloud_type.lower()), dict) else {}
            stock = price.get("stockStatus")
            counts_raw = price.get("availableGpuCounts")
            counts = counts_raw if isinstance(counts_raw, list) else []
            requested_count_available = settings["gpu_count"] in counts if isinstance(counts_raw, list) else True
            clouds.append(
                {
                    "cloud_type": cloud_type,
                    "stock_status": stock or "None",
                    "available": str(stock).lower() in {"high", "medium", "low"} and requested_count_available,
                    "price_per_hour": price.get("uninterruptablePrice"),
                    "available_gpu_counts": counts,
                }
            )
        candidates.append(
            {
                "gpu_type_id": item.get("id") or requested_id,
                "display_name": item.get("displayName") or requested_id,
                "memory_gb": item.get("memoryInGb"),
                "clouds": clouds,
            }
        )
    return {"status": "ok", "checked_at": _now(), "gpu_count": settings["gpu_count"], "candidates": candidates}


def _format_lease_limit(max_lease_sec: int | None) -> str:
    if max_lease_sec is None:
        return "none (this command does not install an automatic stop limit)"
    if max_lease_sec <= 0:
        return "disabled"
    if max_lease_sec % 3600 == 0:
        return f"{max_lease_sec // 3600}h"
    if max_lease_sec % 60 == 0:
        return f"{max_lease_sec // 60}m"
    return f"{max_lease_sec}s"


def _confirm_runpod_launch(
    config: dict[str, Any],
    settings: dict[str, Any],
    *,
    yes: bool,
    max_lease_sec: int | None,
    wait_for_capacity_sec: int = 0,
    unattended_wait: str | None = None,
    min_cuda_version: str | None = None,
    confirmed_at: str | None = None,
) -> dict[str, Any]:
    """Require one authorization before a billable Pod-creation attempt sequence."""

    interactive = sys.stdin.isatty()
    if not interactive and not yes:
        raise ValueError(
            "RunPod Pod creation requires confirmation in non-interactive mode; "
            "use --yes only when the user has explicitly instructed this billed launch"
        )

    measurement = runpod_gpu_availability(config, settings["gpu_type_ids"], min_cuda_version=min_cuda_version)
    if confirmed_at:
        # The command that wrote the launch request showed this and took the confirmation.
        print(f"Creating the RunPod Pod confirmed at {confirmed_at}", file=sys.stderr)
        return measurement
    candidates = measurement.get("candidates") if measurement.get("status") == "ok" else None
    measured_by_id = {
        candidate.get("gpu_type_id"): candidate
        for candidate in candidates or []
        if isinstance(candidate, dict) and isinstance(candidate.get("gpu_type_id"), str)
    }
    print("RunPod Pod creation will start billing:", file=sys.stderr)
    for gpu_type_id in settings["gpu_type_ids"]:
        candidate = measured_by_id.get(gpu_type_id, {})
        display_name = candidate.get("display_name") if isinstance(candidate.get("display_name"), str) else gpu_type_id
        print(f"  GPU: {display_name} x{settings['gpu_count']}", file=sys.stderr)
        prices: list[str] = []
        for cloud in candidate.get("clouds", []) if isinstance(candidate, dict) else []:
            if not isinstance(cloud, dict) or cloud.get("cloud_type") not in settings["cloud_types"]:
                continue
            price = cloud.get("price_per_hour")
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                prices.append(f"{cloud['cloud_type']} ${price:.3f}/hr")
        print(f"  Hourly price: {'; '.join(prices) if prices else 'unavailable'}", file=sys.stderr)
    if measurement.get("status") != "ok":
        reason = measurement.get("reason")
        if isinstance(reason, str) and reason:
            print(f"  Price lookup: {_redact_secret_text(reason)}", file=sys.stderr)
    if min_cuda_version:
        print(f"  Host CUDA: {min_cuda_version} or newer", file=sys.stderr)
    print(f"  Maximum lease: {_format_lease_limit(max_lease_sec)}", file=sys.stderr)
    if unattended_wait:
        print(f"  Unattended wait: {unattended_wait}", file=sys.stderr)
    if wait_for_capacity_sec > 0:
        print(
            f"  Capacity wait: up to {_format_lease_limit(wait_for_capacity_sec)}; "
            "hourly prices may change while waiting",
            file=sys.stderr,
        )
    if yes:
        return measurement
    print("Create the RunPod Pod? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        response = sys.stdin.readline()
    except KeyboardInterrupt as exc:
        raise ValueError("RunPod launch cancelled; no Pod was created") from exc
    if response.strip().lower() != "y":
        raise ValueError("RunPod launch cancelled; no Pod was created")
    return measurement


def _runpod_settings(config: dict[str, Any]) -> dict[str, Any]:
    storage_mode = config.get("storage_mode", "upload")
    if storage_mode not in ("upload", "container_disk", "object_staging"):
        raise ValueError("runpod.storage_mode must be upload, container_disk, or object_staging")
    required = ("gpu_type_ids",)
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError("runpod requires " + ", ".join(missing))
    gpu_types = config["gpu_type_ids"]
    if not isinstance(gpu_types, list) or not all(isinstance(value, str) and value for value in gpu_types):
        raise ValueError("runpod.gpu_type_ids must be a non-empty list of GPU type IDs")
    api_key_env = config.get("api_key_env", "RUNPOD_API_KEY")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("runpod.api_key_env must be a non-empty environment variable name")
    ports = config.get("ports")
    if ports is not None:
        if not isinstance(ports, list) or not all(isinstance(port, str) for port in ports):
            raise ValueError("runpod.ports must be a list of strings like '8675/http' or '22/tcp'")
        invalid_ports = []
        for port in ports:
            parts = port.rsplit("/", 1)
            if len(parts) != 2 or parts[1] not in ("http", "tcp"):
                invalid_ports.append(port)
        if invalid_ports:
            raise ValueError("runpod.ports only supports /http and /tcp entries; remove unsupported ports: " + ", ".join(invalid_ports))
    data_center_ids = config.get("data_center_ids")
    if data_center_ids is not None and (not isinstance(data_center_ids, list) or not all(isinstance(value, str) and value for value in data_center_ids)):
        raise ValueError("runpod.data_center_ids must be a list of data center IDs")
    country_codes = config.get("country_codes")
    if country_codes is not None and (not isinstance(country_codes, list) or not all(isinstance(value, str) and value for value in country_codes)):
        raise ValueError("runpod.country_codes must be a list of country codes")
    data_center_priority = config.get("data_center_priority")
    if data_center_priority is not None and data_center_priority not in ("availability", "custom"):
        raise ValueError("runpod.data_center_priority must be availability or custom")
    gpu_type_priority = config.get("gpu_type_priority")
    if gpu_type_priority is not None and gpu_type_priority not in ("availability", "custom"):
        raise ValueError("runpod.gpu_type_priority must be availability or custom")
    if gpu_type_priority == "availability" and len(gpu_types) > 1:
        raise ValueError(
            "runpod.gpu_type_priority=availability cannot preserve availability ordering "
            "across multiple GPU candidates through the GraphQL control plane; "
            "use custom ordering or configure one GPU type"
        )
    if data_center_priority == "availability" and data_center_ids is not None and len(data_center_ids) > 1:
        raise ValueError(
            "runpod.data_center_priority=availability cannot preserve availability ordering "
            "across multiple data centers through the GraphQL control plane; "
            "use custom ordering or configure one data center"
        )
    cloud_types_raw = config.get("cloud_types")
    cloud_type_raw = config.get("cloud_type", "ANY")
    if cloud_types_raw is not None:
        if not isinstance(cloud_types_raw, list) or not all(value in ("SECURE", "COMMUNITY") for value in cloud_types_raw):
            raise ValueError("runpod.cloud_types must be a list containing SECURE and/or COMMUNITY")
        cloud_types = list(dict.fromkeys(cloud_types_raw))
        if not cloud_types:
            raise ValueError("runpod.cloud_types must not be empty")
    else:
        if cloud_type_raw in ("ANY", "AUTO"):
            cloud_types = ["COMMUNITY", "SECURE"]
        elif cloud_type_raw in ("SECURE", "COMMUNITY"):
            cloud_types = [cloud_type_raw]
        else:
            raise ValueError("runpod.cloud_type must be SECURE, COMMUNITY, ANY, or AUTO")
    return {
        "api_key_env": api_key_env,
        "storage_mode": storage_mode,
        "template_id": config.get("template_id"),
        "gpu_type_ids": gpu_types,
        "gpu_count": config.get("gpu_count", 1),
        "container_disk_gb": config.get("container_disk_gb", 50),
        "volume_in_gb": config.get("volume_in_gb", 0),
        "workspace_path": config.get("workspace_path", CONTAINER_WORKSPACE),
        "ports": ports,
        "cloud_types": cloud_types,
        "support_public_ip": config.get("support_public_ip"),
        "interruptible": bool(config.get("interruptible", False)),
        "data_center_ids": data_center_ids,
        "data_center_priority": data_center_priority,
        "gpu_type_priority": gpu_type_priority,
        "country_codes": country_codes,
    }


def _runpod_gpu_attempts(gpu_type_ids: list[str]) -> list[list[str]]:
    """Return ordered GPU attempts for deterministic fallback."""

    return [[gpu_type_id] for gpu_type_id in gpu_type_ids]


def _runpod_location_attempts(settings: dict[str, Any]) -> list[dict[str, list[str]]]:
    """Expand location filters into GraphQL-compatible single-location attempts."""

    data_centers = settings.get("data_center_ids") or [None]
    countries = settings.get("country_codes") or [None]
    attempts: list[dict[str, list[str]]] = []
    for data_center in data_centers:
        for country in countries:
            attempt: dict[str, list[str]] = {}
            if data_center is not None:
                attempt["dataCenterIds"] = [str(data_center)]
            if country is not None:
                attempt["countryCodes"] = [str(country)]
            attempts.append(attempt)
    return attempts


def _is_runpod_capacity_error(exc: ValueError) -> bool:
    """Return whether a Pod-create failure says the requested GPU is unavailable."""

    text = str(exc).lower()
    markers = (
        "out of capacity",
        "gpu capacity",
        "compute capacity",
        "out of stock",
        "no gpu",
        "no available",
        "none available",
        "no longer any instances",
        "gpu unavailable",
        "gpu is unavailable",
    )
    if any(marker in text for marker in markers):
        return True
    requested_compute = "requested" in text and ("gpu" in text or "instance" in text)
    return requested_compute and ("not available" in text or "unavailable" in text)


def _is_runpod_transient_error(exc: ValueError) -> bool:
    if isinstance(exc, RunPodAPIError):
        return exc.status_code == 429 or exc.status_code >= 500
    text = str(exc).lower()
    return "runpod api is unreachable" in text or any(f"({code})" in text for code in (429, 500, 502, 503, 504))


def _create_outcome_uncertain(exc: ValueError) -> bool:
    """Whether a failed create may still have created a Pod.

    A refusal (capacity, validation, rate limit) means nothing was created. A
    server error or a lost connection after the request was sent does not say.
    """
    if isinstance(exc, RunPodAPIError):
        return exc.status_code >= 500
    text = str(exc).lower()
    # A reply that arrived but cannot be read says nothing about the create either.
    return any(marker in text for marker in ("runpod api is unreachable", "invalid json", "unexpected response", "did not contain"))


def unstopped_recovered_pod(run_dir: Path) -> str | None:
    """The id of a Pod a recovery recorded and nothing has stopped yet.

    Such a Pod still bills, and a new launch would drop its id from status, so
    launching waits until `kura run stop` has deleted it.
    """
    status = _load_status(run_dir)
    pod_id = status.get("pod_id")
    reference = status.get("last_realization")
    if not isinstance(pod_id, str) or not isinstance(reference, str) or status.get("pod_stopped_at") or status.get("pod_missing_at"):
        return None
    try:
        realization = json.loads((run_dir / reference).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return pod_id if isinstance(realization, dict) and realization.get("recovered_from_intent") else None


def _runpod_pods_named(api_key: str, name: str, *, timeout: float = 30.0) -> list[dict[str, Any]]:
    query = """
    query myPods {
      myself {
        pods {
          id name imageName desiredStatus costPerHr lastStatusChange
          machine { id dataCenterId gpuDisplayName location }
          runtime { uptimeInSeconds }
        }
      }
    }
    """
    data = _runpod_graphql(query, {}, api_key, timeout=timeout)
    myself = data.get("myself") if isinstance(data.get("myself"), dict) else {}
    pods = myself.get("pods") if isinstance(myself.get("pods"), list) else None
    if pods is None:
        raise ValueError("RunPod did not return the account's Pods")
    return [pod for pod in pods if isinstance(pod, dict) and pod.get("name") == name and isinstance(pod.get("id"), str)]


def _discover_pods(api_key: str, name: str, *, settle_sec: float = 5.0) -> list[dict[str, Any]]:
    """Pods carrying `name`, looking twice so a create still settling is seen."""
    found = _runpod_pods_named(api_key, name)
    if not found and settle_sec > 0:
        time.sleep(settle_sec)
        found = _runpod_pods_named(api_key, name)
    return found


def _write_create_intent(run_dir: Path, realization_id: str, *, pod_name: str, request: dict[str, Any], image: str, logs_path: str, purpose: str | None = None, controlled_by: dict[str, Any] | None = None) -> None:
    """Record that Kura is about to create a Pod, before the request is sent."""
    requested_at = _now()
    path = run_dir / "realizations" / f"{realization_id}{CREATE_INTENT_SUFFIX}"
    path.parent.mkdir(exist_ok=True)
    _write_json(path, {
        "kind": "pod_create_intent", "schema_version": 1, "realization_id": realization_id, "executor": "runpod",
        "pod_name": pod_name, "requested_at": requested_at, "remote_image": image, "logs_path": logs_path,
        "request": request, **({"purpose": purpose} if purpose else {}),
        **({"controlled_by": controlled_by} if controlled_by else {}),
    })
    record_launch_phase(run_dir, realization_id, "pod_create_requested", at=requested_at)

    def mutate(latest: dict[str, Any]) -> None:
        latest.update({"state": "launching", "host": "runpod", "started": None, "ended": None, "exit_code": None})
        # A previous realization's Pod must never be mistaken for this one.
        for key in ("pod_id", "last_observation", "pod_stopped_at", "pod_missing_at"):
            latest.pop(key, None)

    _mutate_run_status(run_dir, mutate)


def confirm_runpod_billing(
    config: dict[str, Any], image: str, *, yes: bool, max_lease_sec: int | None,
    wait_for_capacity_sec: int = 0, unattended_wait: str | None = None,
) -> dict[str, Any]:
    """Show the launch's cost and take the user's confirmation without creating anything.

    The job runner launches later from the same settings; a confirmation is the
    writer's, never the runner's.
    """
    settings = _runpod_settings(config)
    min_cuda = runpod_min_cuda_for(settings, image)
    return _confirm_runpod_launch(
        config, settings, yes=yes, max_lease_sec=max_lease_sec, wait_for_capacity_sec=wait_for_capacity_sec,
        unattended_wait=unattended_wait, min_cuda_version=min_cuda,
    )


def _hand_over_unconfirmed_create(run_dir: Path, realization_id: str, pod_name: str, error: str) -> ValueError:
    at = _now()
    write_create_unconfirmed(run_dir, realization_id, error=error)

    def mutate(latest: dict[str, Any]) -> None:
        latest.update({"state": "interrupted", "ended": at, "exit_code": None})
        latest.pop("capacity_wait", None)

    _mutate_run_status(run_dir, mutate)
    append_run_event(run_dir, {"event": "runpod_create_unconfirmed", "timestamp": at, "executor": "runpod", "realization_id": realization_id, "pod_name": pod_name, "error": error})
    return ValueError(
        f"RunPod did not confirm whether it created Pod {pod_name} ({error}). Kura does not retry a create it cannot "
        f"confirm, because a retry could start a second billed Pod. Run `kura run reconcile {run_dir.name}`: it looks "
        "for the Pod by name, records what it finds, and tells you whether to stop it before a new run starts."
    )


def _discover_after_unconfirmed_create(run_dir: Path, realization_id: str, api_key: str, pod_name: str, exc: ValueError) -> list[dict[str, Any]]:
    """The one Pod an unconfirmed create made, or hand the run over to the user."""
    error = _redact_secret_text(str(exc))
    try:
        found = _discover_pods(api_key, pod_name)
    except ValueError as lookup:
        raise _hand_over_unconfirmed_create(run_dir, realization_id, pod_name, f"{error}; looking it up also failed: {_redact_secret_text(str(lookup))}") from exc
    if len(found) != 1:
        raise _hand_over_unconfirmed_create(run_dir, realization_id, pod_name, error) from exc
    return found


def resolve_runpod_create_intents(run_dir: Path, config: dict[str, Any]) -> list[str]:
    """Settle every create intent that has no realization, by discovery only.

    Never creates or deletes a Pod. A Pod found is recorded as interrupted with
    its id, so `kura run stop` deletes it; finding none records the launch as
    failed. Returns one line per settled intent for the user.
    """
    intents = unresolved_create_intents(run_dir, "runpod")
    if not intents:
        return []
    settings = _runpod_settings(config)
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise MissingSecret(settings["api_key_env"], "needed to look for a Pod a crashed launch may have created")
    lines = []
    for intent_path in intents:
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        realization_id = intent_path.name[: -len(CREATE_INTENT_SUFFIX)]
        if (intent_path.parent / f"{realization_id}.json").exists():
            settle_status_from_realization(run_dir, intent_path.parent / f"{realization_id}.json")
            lines.append(f"the launch recorded {realization_id}.json but stopped before status followed it; status now does")
            continue
        pod_name = intent.get("pod_name") if isinstance(intent.get("pod_name"), str) else f"kura-{run_dir.name}-{realization_id}"
        pods = _discover_pods(api_key, pod_name)
        at = _now()
        realization_path = run_dir / "realizations" / f"{realization_id}.json"
        base = {
            "id": realization_id, "executor": "runpod", "recovered_from_intent": intent_path.name,
            "remote_image": intent.get("remote_image"), "request": intent.get("request"), "logs_path": intent.get("logs_path"),
            **({"controlled_by": intent["controlled_by"]} if isinstance(intent.get("controlled_by"), dict) else {}),
            **({"purpose": intent["purpose"]} if isinstance(intent.get("purpose"), str) else {}), **kura_provenance(),
        }
        if not pods:
            realization = {**base, "state": "launch_failed", "attempted_at": intent.get("requested_at"), "pod": None,
                           "error": f"RunPod has no Pod named {pod_name}: it was never created, or it was already deleted"}
            pod_id = None
            lines.append(f"no Pod named {pod_name} exists, so nothing is billing; the launch is recorded as failed (a render run needs compiling again before its next launch)")
        else:
            pods = sorted(pods, key=lambda pod: -((pod.get("runtime") or {}).get("uptimeInSeconds") or 0))
            realization = {**base, "state": "interrupted", "launched_at": intent.get("requested_at"), "pod": _runpod_pod_snapshot(pods[0]),
                           "error": "the launch stopped before Kura recorded this Pod; its job was not started by Kura"}
            if len(pods) > 1:
                realization["duplicate_pod_ids"] = [pod["id"] for pod in pods[1:]]
            pod_id = pods[0]["id"]
            ids = ", ".join(pod["id"] for pod in pods)
            lines.append(f"found Pod {ids} named {pod_name}; it is recorded as interrupted and still billing: run `kura run stop {run_dir.name}` to delete it")
        _write_json(realization_path, as_record("realization", realization))

        def mutate(latest: dict[str, Any], realization: dict[str, Any] = realization, pod_id: str | None = pod_id) -> None:
            latest.update({"state": realization["state"], "ended": at, "exit_code": None, "host": "runpod",
                           "last_realization": str(realization_path.relative_to(run_dir))})
            latest.pop("capacity_wait", None)
            latest.pop("last_observation", None)
            if pod_id:
                latest["pod_id"] = pod_id
            else:
                latest.pop("pod_id", None)

        _mutate_run_status(run_dir, mutate)
        append_run_event(run_dir, {"event": "runpod_create_intent_resolved", "timestamp": at, "executor": "runpod", "realization_id": realization_id,
                                   "pod_name": pod_name, "pod_ids": [pod["id"] for pod in pods]})
    return lines


def _runpod_training_env(
    spec_env: dict[str, str], *, workspace_path: str, run_id: str, realization_id: str,
) -> dict[str, str]:
    return {**spec_env, **kura_container_env(workspace_path=workspace_path, run_id=run_id, realization_id=realization_id)}


def _runpod_session_env(*, workspace_path: str, run_id: str, max_lease_sec: int = 12 * 3600) -> dict[str, str]:
    return {**kura_container_env(workspace_path=workspace_path, run_id=run_id), "KURA_MAX_LEASE_SEC": str(max_lease_sec)}


def _object_store_settings(config: dict[str, Any]) -> dict[str, str]:
    object_store = config.get("object_store")
    if not isinstance(object_store, dict):
        raise ValueError("runpod.object_store must be configured for object_staging")
    required = ("endpoint_url", "bucket")
    missing = [name for name in required if not object_store.get(name)]
    if missing:
        raise ValueError("runpod.object_store requires " + ", ".join(missing))
    access_key_env = object_store.get("access_key_env", "R2_ACCESS_KEY_ID")
    secret_key_env = object_store.get("secret_key_env", "R2_SECRET_ACCESS_KEY")
    if not isinstance(access_key_env, str) or not isinstance(secret_key_env, str):
        raise ValueError("runpod.object_store credential env names must be strings")
    access_key = os.environ.get(access_key_env)
    secret_key = os.environ.get(secret_key_env)
    if not access_key or not secret_key:
        raise ValueError(f"{access_key_env} and {secret_key_env} are needed for object staging; run `kura secrets set` for each in your own terminal")
    prefix = str(object_store.get("prefix", "kura")).strip("/")
    return {
        "endpoint_url": str(object_store["endpoint_url"]),
        "bucket": str(object_store["bucket"]),
        "region": str(object_store.get("region", "auto")),
        "prefix": prefix,
        "access_key_env": access_key_env,
        "secret_key_env": secret_key_env,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def _object_store_client(config: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    settings = _object_store_settings(config)
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ValueError("runpod.storage_mode=object_staging requires optional dependency: pip install 'kura[object-staging]'") from exc
    client = boto3.client("s3", endpoint_url=settings["endpoint_url"], region_name=settings["region"], aws_access_key_id=settings["access_key"], aws_secret_access_key=settings["secret_key"], config=Config(retries={"max_attempts": 10, "mode": "standard"}, read_timeout=7200))
    return client, settings


def _stage_selected_files(*, workspace: Path, run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    """Stage exactly the files a manifest-v2 run's frozen handoff selected."""
    inventory = build_transfer_inventory(workspace, run_dir, run)
    estimate = estimate_transfer(inventory)
    transfer_dir = run_dir / "transfer"
    transfer_dir.mkdir(exist_ok=True)
    free = shutil.disk_usage(transfer_dir).free
    if free < estimate["local_stage_free_bytes"]:
        raise ValueError(
            f"RunPod stage needs {estimate['local_stage_free_bytes']} bytes free in {transfer_dir}, "
            f"but only {free} bytes are available"
        )
    archive_name = f"kura-upload-{run_dir.name}.tar"
    archive_path = transfer_dir / archive_name
    manifest_path = transfer_dir / f"kura-upload-{run_dir.name}.manifest.json"
    proof = write_transfer_archive(workspace, run_dir, inventory, archive_path)
    write_transfer_manifest(manifest_path, proof)
    record = {
        "timestamp": _now(),
        "executor": "runpod",
        "storage_mode": "upload",
        "transfer": "selected-files",
        "archive": str(archive_path.relative_to(run_dir)),
        "archive_name": archive_name,
        "manifest": str(manifest_path.relative_to(run_dir)),
        "total_bytes": proof["payload_bytes"],
        "remote_peak_bytes": proof["tar_bytes"] + proof["payload_bytes"],
        **proof,
    }
    stage_path = run_dir / "realizations" / f"stage-{_realization_id()}.json"
    stage_path.parent.mkdir(exist_ok=True)
    _write_json(stage_path, as_record("stage", record))
    status = _load_status(run_dir)
    status["last_stage"] = str(stage_path.relative_to(run_dir))
    _write_status(run_dir, status)
    append_run_event(run_dir, {
        "event": "run_staged",
        **{key: value for key, value in record.items() if key != "entries"},
        "files": len(record["entries"]),
    })
    return record


def stage_runpod(*, workspace: Path, run_dir: Path, dataset_ids: list[str] | None = None, dataset_id: str | None = None, config: dict[str, Any]) -> dict[str, Any]:
    """Explicitly upload the compiled inputs needed by a RunPod Pod."""
    # Artifact lookups return resolved paths; staged names are relative to the
    # same resolved workspace even when it is reached through a symlink.
    workspace = workspace.resolve()
    run_dir = run_dir.resolve()
    settings = _runpod_settings(config)
    if settings["storage_mode"] == "object_staging":
        raise ValueError("runpod.storage_mode=object_staging is experimental and disabled; use storage_mode=upload")
    try:
        run = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("cannot stage invalid resolved manifest") from exc
    if not isinstance(run, dict):
        raise ValueError("cannot stage invalid resolved manifest")
    if (run_dir / "resolved" / "dataset-projection.lock.json").is_file():
        return _stage_selected_files(workspace=workspace, run_dir=run_dir, run=run)
    raw_ids = dataset_ids or ([dataset_id] if dataset_id else [])
    ids = list(dict.fromkeys(item for item in raw_ids if item))
    dependency = resume_artifact_directory(workspace, run)
    sources = [run_dir / "run.yaml", run_dir / "resolved", *(workspace / "datasets" / item for item in ids)]
    if dependency is not None:
        sources.append(dependency)
    files: list[tuple[Path, str]] = []
    for source in sources:
        if not source.exists():
            raise ValueError(f"cannot stage missing path: {source}")
        if source.is_file():
            files.append((source, str(source.relative_to(workspace))))
            continue
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if "_latent_cache" in relative.parts or path.name == ".aitk_size.json" or path.name.endswith(":Zone.Identifier"):
                continue
            if path.is_file() and not path.is_symlink():
                if not os.access(path, os.R_OK):
                    raise ValueError(f"cannot stage unreadable file: {path}")
                files.append((path, str(path.relative_to(workspace))))
    if not files:
        raise ValueError("nothing to stage")
    total_bytes = sum(path.stat().st_size for path, _ in files)
    staged_at = _now()
    transfer_dir = run_dir / "transfer"
    transfer_dir.mkdir(exist_ok=True)
    archive_name = f"kura-upload-{run_dir.name}.tar.gz"
    archive_path = transfer_dir / archive_name
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, key in files:
            archive.add(path, arcname=key)
    storage_label = {"storage_mode": "upload", "archive": str(archive_path.relative_to(run_dir)), "archive_name": archive_name}
    record = {"timestamp": staged_at, "executor": "runpod", **storage_label, "files": [key for _, key in files], "total_bytes": total_bytes}
    stage_path = run_dir / "realizations" / f"stage-{_realization_id()}.json"
    stage_path.parent.mkdir(exist_ok=True)
    _write_json(stage_path, as_record("stage", record))
    status = _load_status(run_dir)
    status["last_stage"] = str(stage_path.relative_to(run_dir))
    _write_status(run_dir, status)
    append_run_event(run_dir, {"event": "run_staged", **record})
    return record


_POSTFLIGHT_STATUSES = frozenset({"matched", "changed", "uncheckable"})


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a record written outside this process; anything but a JSON object is a ValueError."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"record {path.name} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"record {path.name} is not a JSON object")
    return value


def _remote_postflight(path: Path) -> tuple[str, str, str]:
    """The Pod's postflight verdicts, or ``uncheckable`` for anything missing or malformed."""
    try:
        record = _read_json_object(path)
    except ValueError:
        return "uncheckable", "uncheckable", "uncheckable"

    def verdict(key: str, allowed: frozenset[str]) -> str:
        value = record.get(key)
        return value if isinstance(value, str) and value in allowed else "uncheckable"

    sources = verdict("source_stat_verification", frozenset({"matched", "changed"}))
    links = verdict("view_link_verification", frozenset({"matched", "changed"}))
    derived = (
        "uncheckable" if "uncheckable" in {sources, links}
        else "changed" if "changed" in {sources, links}
        else "matched"
    )
    # The overall verdict must agree with its parts; anything else is untrusted.
    status = derived if verdict("status", frozenset({"matched", "changed"})) == derived else "uncheckable"
    return status, sources, links


def _announce_postflight(run_dir: Path, realization_id: str, ref: str, record: dict[str, Any]) -> None:
    """Append the postflight event once for ``ref``."""
    # A projection proves the event exists only when it says the event was
    # recorded; otherwise scan, which also backfills an event whose append
    # failed earlier or was lost in a crash before the projection.
    announced = _load_status(run_dir).get("dataset_input_postflight")
    proven = isinstance(announced, dict) and announced.get("record") == ref and announced.get("event_recorded") is True
    if not proven and not _event_exists(
        run_dir, event="dataset_input_postflight", realization_id=realization_id, record=ref,
    ):
        append_run_event(run_dir, {
            "event": "dataset_input_postflight",
            "timestamp": record["observed_at"],
            "realization_id": realization_id,
            "record": ref,
            "status": record["status"],
        })


def _postflight_projection(ref: str, record: dict[str, Any]) -> dict[str, Any]:
    warning = dataset_input_drift_warning(record["status"])
    return {
        "status": record["status"],
        "record": ref,
        # A Pod is discarded with its disposable view; nothing is left to clean.
        "view_cleanup": "not-required",
        **({"warning": warning} if warning else {}),
    }


def finalize_runpod_dataset_handoff(
    run_dir: Path, downloaded_run: Path, realization_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Bring the Pod's input records home and write the post-training input record.

    Pod-owned remote records are copied into ``realizations/`` without
    overwriting a record that already exists. The local source stat check is combined with
    the Pod's own postflight. Records read from outside are validated here;
    any problem is raised as ``ValueError`` or ``OSError`` for
    ``project_runpod_dataset_handoff`` to record as uncheckable. Returns the
    record reference and content; announcing it is the caller's separate step.
    """
    if not handoff_was_frozen(run_dir):
        return None
    local = run_dir / "realizations"
    local.mkdir(exist_ok=True)
    remote = downloaded_run / "realizations"
    conflicts: list[str] = []
    # Only names the Pod writes come home. A snapshot file named like a
    # controller-owned record (this postflight, the realization, a stage) must
    # not appear locally and be trusted as the controller's own.
    pod_owned = {
        f"{realization_id}.runpod-input.json",
        f"{realization_id}.runpod-input-postflight.json",
        f"{realization_id}.ai-toolkit-video-preflight.json",
        f"{realization_id}.musubi-video-preflight.json",
    }
    for record in sorted(remote.glob("*.json")) if remote.is_dir() else []:
        if record.name not in pod_owned and not record.name.startswith("remote-exit-"):
            continue
        target = local / record.name
        if not target.exists():
            shutil.copyfile(record, target)
        elif target.read_bytes() != record.read_bytes():
            conflicts.append(record.name)
    postflight_ref = f"realizations/{realization_id}.dataset-input-postflight.json"
    postflight_path = run_dir / postflight_ref
    if postflight_path.is_file():
        postflight = _read_json_object(postflight_path)
        if (
            postflight.get("schema_version") != 1
            or postflight.get("realization_id") != realization_id
            or not isinstance(postflight.get("observed_at"), str)
            or postflight.get("status") not in _POSTFLIGHT_STATUSES
        ):
            raise ValueError("existing dataset input postflight record is malformed")
    else:
        workspace = run_dir.parent.parent
        try:
            lock, _ = load_frozen_dataset_handoff(run_dir / "resolved")
            source_changes = inspect_dataset_sources(workspace, lock)
            local_status = "changed" if source_changes else "matched"
        except (OSError, ValueError) as exc:
            source_changes, local_status = [_redact_secret_text(str(exc))], "uncheckable"
        remote_path = local / f"{realization_id}.runpod-input-postflight.json"
        remote_status, remote_sources, remote_links = _remote_postflight(remote_path)
        statuses = {local_status, remote_status}
        status = "uncheckable" if "uncheckable" in statuses else "changed" if "changed" in statuses else "matched"
        postflight = {
            "schema_version": 1,
            "realization_id": realization_id,
            "observed_at": _now(),
            "status": status,
            "source_stat_verification": local_status,
            "source_changes": source_changes,
            "remote_record": f"realizations/{remote_path.name}",
            "remote_source_stat_verification": remote_sources,
            "remote_view_link_verification": remote_links,
            **({"record_conflicts": conflicts} if conflicts else {}),
        }
        _write_json(postflight_path, as_record("dataset_input_postflight", postflight))
    return postflight_ref, postflight


def project_runpod_dataset_handoff(run_dir: Path, downloaded_run: Path, realization_id: str) -> dict[str, Any] | None:
    """Project post-training input drift for a download; never raises.

    Drift is a warning, so recording it must never block the download that
    lets the Pod be stopped. A failure becomes its own append-only
    ``uncheckable`` record and event; status alone carries it only when even
    that record cannot be written.
    """
    detail: str | None = None
    try:
        finalized = finalize_runpod_dataset_handoff(run_dir, downloaded_run, realization_id)
        if finalized is None:
            return None
        ref, record = finalized
    except (OSError, ValueError) as exc:
        detail = _redact_secret_text(str(exc))
        ref = f"realizations/{realization_id}.dataset-input-postflight-uncheckable-{_realization_id()}.json"
        record = {
            "schema_version": 1,
            "realization_id": realization_id,
            "observed_at": _now(),
            "status": "uncheckable",
            "error": detail,
        }
        try:
            _write_json(run_dir / ref, as_record("dataset_input_postflight", record))
        except OSError as write_error:
            return {
                "status": "uncheckable",
                "view_cleanup": "not-required",
                "warning": dataset_input_drift_warning("uncheckable"),
                "error": f"{detail}; the uncheckable record could not be written: {_redact_secret_text(str(write_error))}",
            }
    # The record exists from here on; status always refers to it, and a missing
    # event is noted rather than hiding the record.
    projection = _postflight_projection(ref, record)
    errors = [detail] if detail else []
    try:
        _announce_postflight(run_dir, realization_id, ref, record)
        projection["event_recorded"] = True
    except (OSError, ValueError) as event_error:
        projection["event_recorded"] = False
        errors.append(f"the postflight event could not be appended: {_redact_secret_text(str(event_error))}")
    if errors:
        projection["error"] = "; ".join(errors)
    return projection


def _runpod_state(pod: dict[str, Any]) -> tuple[str, int | None]:
    desired = pod.get("desiredStatus")
    if desired == "RUNNING":
        return "running", None
    if desired == "TERMINATED":
        return "interrupted", None
    # EXITED lacks a process exit code in the Pod API; never invent failure.
    return "unknown", None


def _runpod_pod_snapshot(pod: dict[str, Any]) -> dict[str, Any]:
    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
    return {
        "id": pod.get("id"),
        "name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "last_started_at": pod.get("lastStartedAt"),
        "last_status_change": pod.get("lastStatusChange"),
        "cost_per_h": pod.get("costPerHr"),
        "public_ip": pod.get("publicIp"),
        "port_mappings": pod.get("portMappings"),
        "machine": {
            "id": machine.get("id"),
            "data_center_id": machine.get("dataCenterId"),
            "gpu_display_name": gpu.get("displayName") or gpu.get("id") or machine.get("gpuDisplayName"),
            "memory_gb": pod.get("memoryInGb"),
            "vcpu_count": pod.get("vcpuCount"),
        },
    }


def launch_runpod(
    *,
    run_dir: Path,
    spec: dict[str, Any],
    image: str,
    config: dict[str, Any],
    dry_run: bool = False,
    wait_for_capacity_sec: int = 0,
    capacity_poll_interval_sec: int = 30,
    yes: bool = False,
    max_lease_sec: int | None = None,
    unattended_wait: str | None = None,
    controlled_by: dict[str, Any] | None = None,
) -> str | None:
    """Create a RunPod Pod using a pre-staged workspace."""
    settings = _runpod_settings(config)
    realization_id = _realization_id()
    workspace_path = settings["workspace_path"]
    write_paths = validated_write_roots(spec, workspace_path=workspace_path)
    log_path = f"{workspace_path}/runs/{run_dir.name}/logs/stdout.log"
    runtime_env = _runpod_training_env(
        spec["env"], workspace_path=workspace_path, run_id=run_dir.name, realization_id=realization_id,
    )
    secret_keys = [key for key in runtime_env if _is_secret(key)]
    if secret_keys:
        raise ValueError("RunPod pod env must not contain secrets; use controller-side secret injection for " + ", ".join(sorted(secret_keys)))
    transfer_codes: dict[str, str] = {}
    if (run_dir / "resolved" / "dataset-projection.lock.json").is_file() and settings["storage_mode"] != "upload":
        raise ValueError("manifest-v2 RunPod runs require runpod.storage_mode=upload for the verified selected-file transfer")
    if settings["storage_mode"] == "object_staging":
        raise ValueError("runpod.storage_mode=object_staging is disabled until object-store credentials can be injected without Pod environment variables")
    elif settings["storage_mode"] == "upload":
        status = _load_status(run_dir)
        stage_ref = status.get("last_stage")
        if not isinstance(stage_ref, str):
            raise ValueError("runpod upload mode found no staged bundle; `kura run execute` stages one before it uploads")
        stage = json.loads((run_dir / stage_ref).read_text(encoding="utf-8"))
        if stage.get("storage_mode") != "upload" or not isinstance(stage.get("archive_name"), str):
            raise ValueError("latest stage is not a runpod upload bundle")
        if (run_dir / "resolved" / "dataset-projection.lock.json").is_file():
            if stage.get("transfer") != "selected-files":
                raise ValueError("manifest-v2 runs require a selected-file stage; stage the run again")
            run = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            pinned_manifest = f"realizations/{realization_id}.transfer-manifest.json"
            manifest_sha256 = pin_transfer_manifest(
                run_dir.parent.parent, run_dir, run, stage, run_dir / pinned_manifest,
            )
        upload_code = os.environ.get("KURA_RUNPOD_UPLOAD_CODE") or f"kura-{run_dir.name}-upload-{secrets.token_hex(4)}"
        download_code = f"kura-{run_dir.name}-download-{secrets.token_hex(4)}"
        transfer_codes = {"upload_code": upload_code, "download_code": download_code, "archive": str(stage.get("archive")), "archive_name": stage["archive_name"]}
        if stage.get("transfer") == "selected-files":
            # Verified above, before the Pod exists.
            transfer_codes.update({
                "stage": stage_ref,
                "archive_sha256": stage["archive_sha256"],
                "input_sha256": stage["input_sha256"],
                # The Pod accepts only this manifest; upload re-proves the stage
                # against it before sending anything.
                "pinned_manifest": pinned_manifest,
                "manifest_sha256": manifest_sha256,
                "verification": "stage-matches-compile-before-pod-creation",
            })
        runtime_env.update({
            "KURA_UPLOAD_CODE": upload_code,
            "KURA_DOWNLOAD_CODE": download_code,
            "KURA_UPLOAD_ARCHIVE_NAME": stage["archive_name"],
            "KURA_WORKSPACE": workspace_path,
            "KURA_RUN_ID": run_dir.name,
        })
        if isinstance(settings.get("template_id"), str) and settings["template_id"]:
            start_command = None
            workspace_contract = "RunPod starts the official template normally; Kura uploads the staged bundle with SCP, runs the backend command over SSH, then downloads outputs before stopping the disposable Pod"
        else:
            upload_script = r'''
set -u
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/samples" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/metrics"
touch "$KURA_LOG_PATH"
if ! command -v sshd >/dev/null 2>&1; then
  apt-get update >> "$KURA_LOG_PATH" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server >> "$KURA_LOG_PATH" 2>&1
fi
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" | grep '^ssh-' > /root/.ssh/authorized_keys || true
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd >> "$KURA_LOG_PATH" 2>&1 || true
echo "Kura SSH staging pod is ready; waiting for controller" >> "$KURA_LOG_PATH"
sleep infinity
'''.strip()
            start_command = ["sh", "-lc", upload_script]
            workspace_contract = "Kura starts an SSH staging container, uploads the staged bundle with SCP, runs the backend command over SSH, then downloads outputs before stopping the disposable Pod"
    else:
        mkdir_targets = [
            '"$(dirname "$KURA_LOG_PATH")"',
            *(f'"$KURA_WORKSPACE/runs/$KURA_RUN_ID/{name}"' for name in ("outputs", "checkpoints", "samples", "metrics")),
            *(shlex.quote(path) for path in write_paths),
        ]
        checks = [f"test -w {shlex.quote(path)}" for path in write_paths]
        wrapper = " && ".join([f"mkdir -p {' '.join(mkdir_targets)}", *checks, 'exec "$@" >> "$KURA_LOG_PATH" 2>&1'])
        start_command = ["sh", "-lc", wrapper, "kura-job", *spec["argv"]]
        workspace_contract = "Container disk only; caller must ensure inputs exist in the container workspace"
    request_body = {
        "name": f"kura-{run_dir.name}-{realization_id}",
        "gpuCount": settings["gpu_count"],
        "containerDiskInGb": settings["container_disk_gb"],
        "volumeInGb": settings["volume_in_gb"],
        "interruptible": settings["interruptible"], "env": runtime_env,
        # A template supplies its own image, whose CUDA version Kura does not know.
        "minCudaVersion": runpod_min_cuda_for(settings, image),
    }
    if settings.get("support_public_ip") is not None:
        request_body["supportPublicIp"] = bool(settings["support_public_ip"])
    if start_command is not None:
        request_body["dockerStartCmd"] = start_command
    if isinstance(settings.get("template_id"), str) and settings["template_id"]:
        request_body["templateId"] = settings["template_id"]
    else:
        request_body["imageName"] = image
    if isinstance(settings.get("ports"), list) and all(isinstance(port, str) for port in settings["ports"]):
        request_body["ports"] = settings["ports"]
    safe_request = dict(request_body)
    safe_request["env"] = _safe_env(runtime_env)
    safe_request["gpuTypeIds"] = _runpod_gpu_attempts(settings["gpu_type_ids"])[0]
    safe_request["gpuTypeCandidates"] = settings["gpu_type_ids"]
    safe_request["gpuTypePriority"] = settings.get("gpu_type_priority")
    safe_request["cloudTypeCandidates"] = settings["cloud_types"]
    safe_request["dataCenterCandidates"] = settings.get("data_center_ids")
    safe_request["dataCenterPriority"] = settings.get("data_center_priority")
    safe_request["countryCandidates"] = settings.get("country_codes")
    if dry_run:
        print(json.dumps({"runpod_create_request": safe_request, "logs_path": log_path}, ensure_ascii=False, indent=2))
        return None
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise MissingSecret(settings["api_key_env"], "needed to launch a RunPod run")
    if wait_for_capacity_sec < 0:
        raise ValueError("wait_for_capacity_sec must be zero or greater")
    if wait_for_capacity_sec and capacity_poll_interval_sec <= 0:
        raise ValueError("capacity_poll_interval_sec must be greater than zero while waiting for capacity")
    confirmation_measurement: dict[str, Any] | None = _confirm_runpod_launch(
        config,
        settings,
        yes=yes,
        max_lease_sec=max_lease_sec,
        wait_for_capacity_sec=wait_for_capacity_sec,
        unattended_wait=unattended_wait,
        min_cuda_version=request_body["minCudaVersion"],
        confirmed_at=(controlled_by or {}).get("billing_confirmed_at"),
    )
    pod: dict[str, Any] | None = None
    used_request: dict[str, Any] | None = None
    launch_errors: list[dict[str, str]] = []
    capacity_wait_started_at: str | None = None
    capacity_wait_closed = False

    def close_capacity_wait(outcome: str, **facts: Any) -> None:
        """Write the wait's one end line, if a wait started; the last line says where it stands."""
        nonlocal capacity_wait_closed
        if capacity_wait_started_at is None or capacity_wait_closed:
            return
        capacity_wait_closed = True
        append_capacity_wait(run_dir, realization_id, as_record("capacity_wait_end", {"at": _now(), "outcome": outcome, "failed_rounds": capacity_rounds, **facts}))

    capacity_wait_started_monotonic = time.monotonic()
    capacity_rounds = 0
    transient_rounds = 0
    controller_phase = "probe"
    intent_written = False
    try:
        while pod is None:
            if wait_for_capacity_sec and capacity_rounds and time.monotonic() - capacity_wait_started_monotonic >= wait_for_capacity_sec:
                break
            launch_errors = []
            round_exceptions: list[ValueError] = []
            attempts: list[tuple[list[str], str, dict[str, list[str]]]] = []
            transient_probe = False
            if wait_for_capacity_sec:
                controller_phase = "probe"
                measurement = confirmation_measurement
                confirmation_measurement = None
                if measurement is None:
                    measurement = runpod_gpu_availability(config, settings["gpu_type_ids"], min_cuda_version=request_body["minCudaVersion"])
                if measurement.get("status") == "ok":
                    for candidate in measurement.get("candidates", []):
                        if not isinstance(candidate, dict) or not isinstance(candidate.get("gpu_type_id"), str):
                            continue
                        for cloud in candidate.get("clouds", []):
                            if isinstance(cloud, dict) and cloud.get("available") and cloud.get("cloud_type") in settings["cloud_types"]:
                                attempts.extend(
                                    ([candidate["gpu_type_id"]], cloud["cloud_type"], placement)
                                    for placement in _runpod_location_attempts(settings)
                                )
                    if not attempts:
                        transient_rounds = 0
                        launch_errors.append({"gpu_type_ids": ", ".join(settings["gpu_type_ids"]), "cloud_type": ", ".join(settings["cloud_types"]), "error": "RunPod stock snapshot reports no matching GPU capacity", "classification": "capacity"})
                else:
                    kind = str(measurement.get("error_kind") or "request")
                    launch_errors.append({"gpu_type_ids": ", ".join(settings["gpu_type_ids"]), "cloud_type": "availability-probe", "error": _redact_secret_text(str(measurement.get("reason") or "RunPod availability probe failed")), "classification": kind})
                    transient_probe = kind in {"rate_limit", "transient"}
                    if not transient_probe:
                        break
                    transient_rounds += 1
            else:
                attempts = [
                    (gpu_type_ids, cloud_type, placement)
                    for gpu_type_ids in _runpod_gpu_attempts(settings["gpu_type_ids"])
                    for cloud_type in settings["cloud_types"]
                    for placement in _runpod_location_attempts(settings)
                ]

            for gpu_type_ids, cloud_type, placement in attempts:
                attempt_request = dict(request_body)
                attempt_request["gpuTypeIds"] = gpu_type_ids
                attempt_request["cloudType"] = cloud_type
                attempt_request.update(placement)
                controller_phase = "create"
                if not intent_written:
                    _write_create_intent(run_dir, realization_id, pod_name=request_body["name"], request=safe_request, image=image, logs_path=log_path, controlled_by=controlled_by)
                    intent_written = True
                try:
                    pod = _runpod_request("POST", "/pods", api_key, attempt_request)
                    used_request = attempt_request
                    controller_phase = "probe"
                    break
                except ValueError as exc:
                    if _create_outcome_uncertain(exc):
                        found = _discover_after_unconfirmed_create(run_dir, realization_id, api_key, request_body["name"], exc)
                        pod, used_request = found[0], attempt_request
                        controller_phase = "probe"
                        break
                    round_exceptions.append(exc)
                    classification = "capacity" if _is_runpod_capacity_error(exc) else "transient" if _is_runpod_transient_error(exc) else "fatal"
                    launch_errors.append({"gpu_type_ids": ", ".join(gpu_type_ids), "cloud_type": cloud_type, "error": _redact_secret_text(str(exc)), "classification": classification})
                    controller_phase = "probe"
            if pod is not None:
                break
            retryable_create = bool(round_exceptions) and all(_is_runpod_capacity_error(exc) or _is_runpod_transient_error(exc) for exc in round_exceptions)
            if not wait_for_capacity_sec or (attempts and not retryable_create):
                break
            if any(_is_runpod_transient_error(exc) for exc in round_exceptions):
                transient_rounds += 1
            elif round_exceptions:
                transient_rounds = 0
            elapsed = time.monotonic() - capacity_wait_started_monotonic
            if elapsed >= wait_for_capacity_sec:
                break
            capacity_rounds += 1
            now = _now()
            if capacity_wait_started_at is None:
                capacity_wait_started_at = now
                append_run_event(
                    run_dir,
                    {
                        "event": "runpod_capacity_wait_started",
                        "timestamp": now,
                        "executor": "runpod",
                        "timeout_sec": wait_for_capacity_sec,
                        "poll_interval_sec": capacity_poll_interval_sec,
                        "attempts": capacity_rounds,
                        "gpu_type_ids": settings["gpu_type_ids"],
                        "cloud_types": settings["cloud_types"],
                    },
                )
            remaining = max(wait_for_capacity_sec - elapsed, 0)
            capacity_wait = {
                "started_at": capacity_wait_started_at,
                "attempts": capacity_rounds,
                "last_attempt_at": now,
                "remaining_sec": round(remaining),
                "poll_interval_sec": capacity_poll_interval_sec,
                "gpu_type_ids": settings["gpu_type_ids"],
                "cloud_types": settings["cloud_types"],
                "last_result": launch_errors[-1].get("classification") if launch_errors else "capacity",
            }
            append_capacity_wait(run_dir, realization_id, as_record("capacity_wait_round", capacity_wait))

            def mutate_queued(latest: dict[str, Any], capacity_wait: dict[str, Any] = capacity_wait) -> None:
                latest.update({"state": "queued", "started": None, "ended": None, "exit_code": None, "host": "runpod", "capacity_wait": capacity_wait})
                latest.pop("pod_id", None)
                latest.pop("last_observation", None)

            _mutate_run_status(run_dir, mutate_queued)
            backoff = min(2 ** min(transient_rounds, 4), 10) if transient_probe or transient_rounds else 1
            sleep_for = min(capacity_poll_interval_sec * backoff, 300, remaining)
            print(
                f"RunPod capacity unavailable; waiting {sleep_for:.0f}s "
                f"before probe {capacity_rounds + 1} (up to {remaining:.0f}s remaining).",
                file=sys.stderr,
            )
            controller_phase = "sleep"
            sleep_checking_stop(sleep_for)
            controller_phase = "probe"
    except KeyboardInterrupt as exc:
        cancelled_at = _now()
        if capacity_wait_started_at is not None:
            close_capacity_wait("cancelled")
        if controller_phase == "create" and intent_written:
            write_create_unconfirmed(run_dir, realization_id, error="the launch was interrupted while RunPod was creating the Pod")

        def mutate_cancelled(latest: dict[str, Any]) -> None:
            latest.update({"state": "interrupted", "ended": cancelled_at, "exit_code": None, "host": "runpod"})
            latest.pop("capacity_wait", None)

        _mutate_run_status(run_dir, mutate_cancelled)
        append_run_event(run_dir, {"event": "runpod_capacity_wait_cancelled", "timestamp": cancelled_at, "executor": "runpod", "attempts": capacity_rounds, "phase": controller_phase})
        if controller_phase == "create":
            raise ValueError(
                "RunPod capacity wait was interrupted during Pod creation; creation is unconfirmed. "
                f"Run `kura run reconcile {run_dir.name}` before retrying: it looks for the Pod by name"
            ) from exc
        if intent_written:
            # Every create so far was refused, so no Pod exists; settle the intent here.
            realization_path = run_dir / "realizations" / f"{realization_id}.json"
            _write_json(realization_path, as_record("realization", {
                "id": realization_id, "executor": "runpod", **({"controlled_by": controlled_by} if controlled_by else {}), "state": "launch_failed", "attempted_at": cancelled_at, "pod": None,
                "request": safe_request, "logs_path": log_path, "create_intent": f"{realization_id}{CREATE_INTENT_SUFFIX}",
                "error": "the capacity wait was cancelled; every create attempt had been refused, so no Pod exists", **kura_provenance(),
            }))
            _mutate_run_status(run_dir, lambda latest: latest.update({"state": "launch_failed", "last_realization": str(realization_path.relative_to(run_dir))}))
        raise ValueError("RunPod capacity wait cancelled; no Pod was created") from exc
    except Exception as exc:
        # An unconfirmed create, or any other failure, still ends the wait.
        close_capacity_wait("abandoned", error=_redact_secret_text(str(exc)))
        raise
    if pod is None or used_request is None:
        failed_at = _now()
        if capacity_wait_started_at is not None:
            close_capacity_wait("gave_up", last_result=launch_errors[-1].get("classification") if launch_errors else None)
        realization_path = run_dir / "realizations" / f"{realization_id}.json"
        realization_path.parent.mkdir(exist_ok=True)
        failed_request = dict(safe_request)
        failed_request["launch_attempts"] = launch_errors
        realization = {
            "id": realization_id, "executor": "runpod", **({"controlled_by": controlled_by} if controlled_by else {}), "state": "launch_failed", "attempted_at": failed_at,
            "remote_image": image, "image_identity": image_reference_identity(image), **({"adapter_source": spec["adapter_source"]} if isinstance(spec.get("adapter_source"), dict) else {}), "pod": None, "request": failed_request,
            "container_cwd": spec["cwd"], "backend_command": spec["argv"], "write_roots": spec.get("write_roots", []),
            "logs_path": log_path,
            "workspace_contract": workspace_contract,
            "error": "; ".join(f"{item['gpu_type_ids']} {item['cloud_type']}: {item['error']}" for item in launch_errors),
            "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"},
            **kura_provenance(),
        }
        _write_json(realization_path, as_record("realization", realization))
        status = _load_status(run_dir)
        status.update({"state": "launch_failed", "started": None, "ended": failed_at, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir))})
        status.pop("pod_id", None)
        status.pop("last_observation", None)
        status.pop("capacity_wait", None)
        _write_status(run_dir, status)
        append_run_event(run_dir, {"event": "run_launch_failed", "timestamp": failed_at, "executor": "runpod", "realization_id": realization_id, "error": realization["error"]})
        raise ValueError("RunPod launch failed for all configured cloud types: " + realization["error"])
    safe_used_request = dict(used_request)
    safe_used_request["env"] = _safe_env(runtime_env)
    safe_used_request["gpuTypeCandidates"] = settings["gpu_type_ids"]
    safe_used_request["gpuTypePriority"] = settings.get("gpu_type_priority")
    safe_used_request["cloudTypeCandidates"] = settings["cloud_types"]
    safe_used_request["dataCenterCandidates"] = settings.get("data_center_ids")
    safe_used_request["dataCenterPriority"] = settings.get("data_center_priority")
    safe_used_request["countryCandidates"] = settings.get("country_codes")
    if capacity_wait_started_at is not None:
        safe_used_request["capacityWait"] = {"startedAt": capacity_wait_started_at, "failedRounds": capacity_rounds}
        close_capacity_wait("created")
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise ValueError("RunPod create response did not include a pod ID")
    state, _ = _runpod_state(pod)
    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    realization_path.parent.mkdir(exist_ok=True)
    realization = {
        "id": realization_id, "executor": "runpod", **({"controlled_by": controlled_by} if controlled_by else {}), "state": state, "launched_at": _now(),
        "remote_image": image, "image_identity": image_reference_identity(image), **({"adapter_source": spec["adapter_source"]} if isinstance(spec.get("adapter_source"), dict) else {}), "pod": _runpod_pod_snapshot(pod),
        "request": safe_used_request, "container_cwd": spec["cwd"], "backend_command": spec["argv"], "write_roots": spec.get("write_roots", []),
        "logs_path": log_path, "workspace_contract": workspace_contract, "transfer": transfer_codes,
        "create_intent": f"{realization_id}{CREATE_INTENT_SUFFIX}",
        "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"}, **kura_provenance(),
    }
    _write_json(realization_path, as_record("realization", realization))
    record_launch_phase(run_dir, realization_id, "pod_created", at=realization["launched_at"], pod_id=pod_id)
    status = _load_status(run_dir)
    status.update({"state": state, "started": realization["launched_at"], "ended": None, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir)), "pod_id": pod_id})
    status.pop("last_observation", None)
    status.pop("capacity_wait", None)
    # A new realization starts its own progress; an earlier one's step is not its.
    for key in PROGRESS_FIELDS:
        status.pop(key, None)
    _write_status(run_dir, status)
    if capacity_wait_started_at is not None:
        append_run_event(run_dir, {"event": "runpod_capacity_acquired", "timestamp": _now(), "executor": "runpod", "attempts": capacity_rounds + 1, "pod_id": pod_id})
    append_run_event(run_dir, {"event": "run_started", "timestamp": _now(), "executor": "runpod", "realization_id": realization_id, "pod_id": pod_id})
    return realization_id


# Where the Pod keeps its lease deadline, in seconds since the epoch.
LEASE_DEADLINE_PATH = "/tmp/kura-lease-deadline"


def _runpod_lease_guard_shell(*, max_lease_sec: int, pod_id: str, log_path: str) -> str:
    """The maximum lease: delete the Pod after ``max_lease_sec`` whatever the controller does."""
    if max_lease_sec <= 0:
        return ""
    pod_export = f"RUNPOD_POD_ID={shlex.quote(pod_id)}; export RUNPOD_POD_ID" if pod_id else ":"
    deadline_file = shlex.quote(LEASE_DEADLINE_PATH)
    # The deadline lives in a file, so `kura run lease` can move it; an unreadable
    # file falls back to the deadline set here.
    return f"""
{POD_SELF_DELETE_FUNCTION}
kura_lease_initial=$(( $(date +%s) + {int(max_lease_sec)} ))
# A guard started again on the same Pod never moves a deadline already set.
[ -s {deadline_file} ] || {{ echo "$kura_lease_initial" > {deadline_file}.tmp && mv {deadline_file}.tmp {deadline_file}; }}
(
  set +e
  {pod_export}
  while :; do
    kura_lease_deadline=$(cat {deadline_file} 2>/dev/null)
    # Anything but a plausible epoch (digits, at most 11 of them) falls back to the armed deadline.
    case "$kura_lease_deadline" in ''|*[!0-9]*|????????????*) kura_lease_deadline=$kura_lease_initial ;; esac
    [ "$(date +%s)" -ge "$kura_lease_deadline" ] && break
    sleep 30
  done
  mkdir -p "$(dirname {shlex.quote(log_path)})" || true
  echo "[kura] the maximum lease ended; deleting the Pod" >> {shlex.quote(log_path)} 2>&1 || true
  kura_pod_self_delete {shlex.quote(log_path)} || true
) </dev/null >/dev/null 2>&1 &
""".strip()



def launch_runpod_session(
    *,
    run_dir: Path,
    image: str,
    config: dict[str, Any],
    purpose: str,
    dry_run: bool = False,
    yes: bool = False,
    max_lease_sec: int = 12 * 3600,
    controlled_by: dict[str, Any] | None = None,
) -> str | None:
    """Create a thin disposable RunPod session without Kura training staging.

    The Pod's maximum lease runs from its start and lives in the deadline file,
    so `kura run lease` can move it like a training Pod's.
    """
    settings = _runpod_settings(config)
    realization_id = _realization_id()
    workspace_path = settings["workspace_path"]
    log_path = f"{workspace_path}/runs/{run_dir.name}/logs/stdout.log"
    runtime_env = _runpod_session_env(workspace_path=workspace_path, run_id=run_dir.name, max_lease_sec=max_lease_sec)
    # The lease is armed first, so a Pod whose SSH setup stalls is still bounded.
    ssh_script = r'''
set -u
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
touch "$KURA_LOG_PATH"
'''.strip() + "\n" + _runpod_lease_guard_shell(max_lease_sec=max_lease_sec, pod_id="", log_path=log_path) + "\n" + r'''
if ! command -v sshd >/dev/null 2>&1; then
  apt-get update >> "$KURA_LOG_PATH" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server >> "$KURA_LOG_PATH" 2>&1
fi
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" | grep '^ssh-' > /root/.ssh/authorized_keys || true
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd >> "$KURA_LOG_PATH" 2>&1 || true
echo "Kura RunPod session is ready for controller" >> "$KURA_LOG_PATH"
sleep infinity
'''.strip()
    request_body = {
        "name": f"kura-{run_dir.name}-{realization_id}",
        "gpuCount": settings["gpu_count"],
        "containerDiskInGb": settings["container_disk_gb"],
        "volumeInGb": settings["volume_in_gb"],
        "interruptible": settings["interruptible"],
        "env": runtime_env,
        "dockerStartCmd": ["sh", "-lc", POD_SELF_DELETE_FUNCTION + "\n" + ssh_script],
        "imageName": image,
        "minCudaVersion": runpod_min_cuda_version(image),
    }
    if settings.get("support_public_ip") is not None:
        request_body["supportPublicIp"] = bool(settings["support_public_ip"])
    if isinstance(settings.get("ports"), list) and all(isinstance(port, str) for port in settings["ports"]):
        request_body["ports"] = settings["ports"]
    safe_request = dict(request_body)
    safe_request["env"] = _safe_env(runtime_env)
    safe_request["gpuTypeIds"] = _runpod_gpu_attempts(settings["gpu_type_ids"])[0]
    safe_request["gpuTypeCandidates"] = settings["gpu_type_ids"]
    safe_request["gpuTypePriority"] = settings.get("gpu_type_priority")
    safe_request["cloudTypeCandidates"] = settings["cloud_types"]
    safe_request["dataCenterCandidates"] = settings.get("data_center_ids")
    safe_request["dataCenterPriority"] = settings.get("data_center_priority")
    safe_request["countryCandidates"] = settings.get("country_codes")
    if dry_run:
        print(json.dumps({"runpod_create_request": safe_request, "logs_path": log_path}, ensure_ascii=False, indent=2))
        return None
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise MissingSecret(settings["api_key_env"], "needed to launch a RunPod session")
    _confirm_runpod_launch(config, settings, yes=yes, max_lease_sec=max_lease_sec, min_cuda_version=request_body["minCudaVersion"],
                           confirmed_at=(controlled_by or {}).get("billing_confirmed_at"))
    pod: dict[str, Any] | None = None
    used_request: dict[str, Any] | None = None
    launch_errors: list[dict[str, str]] = []
    intent_written = False
    for gpu_type_ids in _runpod_gpu_attempts(settings["gpu_type_ids"]):
        for cloud_type in settings["cloud_types"]:
            for placement in _runpod_location_attempts(settings):
                attempt_request = dict(request_body)
                attempt_request["gpuTypeIds"] = gpu_type_ids
                attempt_request["cloudType"] = cloud_type
                attempt_request.update(placement)
                if not intent_written:
                    _write_create_intent(run_dir, realization_id, pod_name=request_body["name"], request=safe_request, image=image, logs_path=log_path, purpose=purpose, controlled_by=controlled_by)
                    intent_written = True
                try:
                    pod = _runpod_request("POST", "/pods", api_key, attempt_request)
                    used_request = attempt_request
                    break
                except KeyboardInterrupt as exc:
                    raise _hand_over_unconfirmed_create(run_dir, realization_id, request_body["name"], "interrupted during the create request") from exc
                except ValueError as exc:
                    if _create_outcome_uncertain(exc):
                        found = _discover_after_unconfirmed_create(run_dir, realization_id, api_key, request_body["name"], exc)
                        pod, used_request = found[0], attempt_request
                        break
                    launch_errors.append({"gpu_type_ids": ", ".join(gpu_type_ids), "cloud_type": cloud_type, "error": _redact_secret_text(str(exc))})
            if pod is not None:
                break
        if pod is not None:
            break
    if pod is None or used_request is None:
        failed_at = _now()
        realization_path = run_dir / "realizations" / f"{realization_id}.json"
        realization_path.parent.mkdir(exist_ok=True)
        failed_request = dict(safe_request)
        failed_request["launch_attempts"] = launch_errors
        realization = {"id": realization_id, "executor": "runpod", **({"controlled_by": controlled_by} if controlled_by else {}), "purpose": purpose, "state": "launch_failed", "attempted_at": failed_at, "remote_image": image, "pod": None, "request": failed_request, "logs_path": log_path, "error": "; ".join(f"{item['gpu_type_ids']} {item['cloud_type']}: {item['error']}" for item in launch_errors), **kura_provenance()}
        _write_json(realization_path, as_record("realization", realization))
        status = _load_status(run_dir)
        status.update({"state": "launch_failed", "started": None, "ended": failed_at, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir))})
        status.pop("pod_id", None)
        status.pop("last_observation", None)
        _write_status(run_dir, status)
        append_run_event(run_dir, {"event": "run_launch_failed", "timestamp": failed_at, "executor": "runpod", "realization_id": realization_id, "error": realization["error"]})
        raise ValueError("RunPod session launch failed for all configured cloud types: " + realization["error"])
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise ValueError("RunPod create response did not include a pod ID")
    safe_used_request = dict(used_request)
    safe_used_request["env"] = _safe_env(runtime_env)
    safe_used_request["gpuTypeCandidates"] = settings["gpu_type_ids"]
    safe_used_request["gpuTypePriority"] = settings.get("gpu_type_priority")
    safe_used_request["cloudTypeCandidates"] = settings["cloud_types"]
    safe_used_request["dataCenterCandidates"] = settings.get("data_center_ids")
    safe_used_request["dataCenterPriority"] = settings.get("data_center_priority")
    safe_used_request["countryCandidates"] = settings.get("country_codes")
    state, _ = _runpod_state(pod)
    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    realization_path.parent.mkdir(exist_ok=True)
    realization = {"id": realization_id, "executor": "runpod", **({"controlled_by": controlled_by} if controlled_by else {}), "purpose": purpose, "lease_guard": "deadline_file", "state": state, "launched_at": _now(), "remote_image": image, "pod": _runpod_pod_snapshot(pod), "request": safe_used_request, "logs_path": log_path, "workspace_contract": "Thin RunPod session; Kura connects over SSH tunnel and records render artifacts locally", "create_intent": f"{realization_id}{CREATE_INTENT_SUFFIX}", **kura_provenance()}
    _write_json(realization_path, as_record("realization", realization))
    status = _load_status(run_dir)
    status.update({"state": state, "started": realization["launched_at"], "ended": None, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir)), "pod_id": pod_id})
    status.pop("last_observation", None)
    _write_status(run_dir, status)
    append_run_event(run_dir, {"event": "run_started", "timestamp": _now(), "executor": "runpod", "purpose": purpose, "realization_id": realization_id, "pod_id": pod_id})
    return realization_id


def reconcile_runpod(
    run_dir: Path,
    config: dict[str, Any],
    *,
    timeout: float = 30.0,
    blocking: bool = True,
    source: str = "explicit",
) -> dict[str, Any]:
    with _run_operation_lock(run_dir, "observe", blocking=blocking):
        settings = _runpod_settings(config)
        api_key = os.environ.get(settings["api_key_env"])
        if not api_key:
            raise MissingSecret(settings["api_key_env"], "needed to reconcile a RunPod run")
        status = _load_status(run_dir)
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            raise ValueError("run has no launched realization")
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        pod_id = realization.get("pod", {}).get("id")
        if not isinstance(pod_id, str):
            raise ValueError("latest realization has no RunPod pod ID")
        try:
            pod: dict[str, Any] | None = _runpod_request("GET", f"/pods/{pod_id}", api_key, timeout=timeout)
        except RunPodAPIError as exc:
            # The Pod no longer exists: it deleted itself (unattended wait or
            # maximum lease) or was deleted elsewhere. Only an explicit
            # reconcile records that as terminal; an automatic observation
            # (the monitor) must not turn one flaky "not found" into a run that
            # looks safe to relaunch while its Pod may still be billing.
            if exc.status_code != 404 or source != "explicit":
                raise
            pod = None
        observed_at = _now()
        if pod is None:
            state, exit_code = "interrupted", None
        else:
            state, exit_code = _runpod_state(pod)
        ended = None if state == "running" else observed_at
        ended_source = None if state == "running" else "observed_at"
        observation = {
            "realization_id": realization["id"],
            "observed_at": observed_at,
            "source": source,
            "state": state,
            "exit_code": exit_code,
            "ended": ended,
            "ended_source": ended_source,
            "pod_id": pod_id,
            **(_runpod_pod_snapshot(pod) if pod is not None else {"pod_missing": True}),
        }
        recorded = False

        def mutate(latest: dict[str, Any]) -> None:
            nonlocal recorded
            if latest.get("last_realization") != realization_ref:
                return
            lifecycle_changed = latest.get("state") != state or latest.get("exit_code") != exit_code
            if source == "explicit" or lifecycle_changed:
                observation_path = _write_observation(run_dir, realization["id"], observation)
                latest["last_observation"] = str(observation_path.relative_to(run_dir))
                recorded = True
            if latest.get("state") not in TERMINAL_STATES:
                latest.update({"state": state, "exit_code": exit_code, "ended": ended})
            if pod is None:
                latest["pod_missing_at"] = observed_at
            effective_state = latest.get("state") if isinstance(latest.get("state"), str) else state
            _materialize_stdout_progress(run_dir, latest, state=effective_state)

        status = _mutate_run_status(run_dir, mutate, blocking=blocking)
        if recorded:
            append_run_event(run_dir, {"event": "run_reconciled", **observation})
        return status


def stop_runpod(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    settings = _runpod_settings(config)
    status = _load_status(run_dir)
    if status.get("state") == "queued" and isinstance(status.get("capacity_wait"), dict):
        raise ValueError(
            "run is waiting for RunPod capacity and no Pod exists to stop; "
            "interrupt the active launch controller with Ctrl+C; if no controller remains, "
            f"run `kura doctor runpod`, then `kura run execute {run_dir.name}`, which follows the wait "
            "or starts it again"
        )
    if unresolved_create_intents(run_dir):
        raise ValueError(
            f"a launch of this run stopped before recording whether its Pod was created; run `kura run reconcile {run_dir.name}` "
            "first, which finds the Pod by name so this command can delete it"
        )
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise MissingSecret(settings["api_key_env"], "needed to stop a RunPod run")
    pod_id = status.get("pod_id")
    if not isinstance(pod_id, str):
        raise ValueError("run has no RunPod pod ID")
    realization = status.get("last_realization")
    realization_id = Path(realization).stem if isinstance(realization, str) else None
    # A status from before realizations still gets its stop recorded.
    stop_record_id = realization_id or "unrecorded"
    if realization_id:
        record_launch_phase(run_dir, realization_id, "pod_stop_requested", pod_id=pod_id)
    # The Pod's container disk is disposable; terminate compute explicitly,
    # together with any duplicate a recovered launch recorded beside it.
    duplicates: list[str] = []
    if realization:
        try:
            recorded = json.loads((run_dir / realization).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
        if isinstance(recorded, dict) and isinstance(recorded.get("duplicate_pod_ids"), list):
            duplicates = [item for item in recorded["duplicate_pod_ids"] if isinstance(item, str) and item != pod_id]
    requested_at = _now()
    targets: list[dict[str, Any]] = []
    for target in (pod_id, *duplicates):
        try:
            _runpod_request("DELETE", f"/pods/{target}", api_key)
            targets.append({"pod_id": target, "result": "deleted"})
        except ValueError as exc:
            message = str(exc).lower()
            if "404" not in message and "pod not found" not in message:
                write_stop_record(run_dir, stop_record_id, executor="runpod", targets=targets + [{"pod_id": target, "result": "failed"}],
                                  requested_at=requested_at, stopped_at=None, outcome="failed", error=_redact_secret_text(str(exc)))
                raise
            targets.append({"pod_id": target, "result": "already_gone"})
    ended_at = _now()
    write_stop_record(run_dir, stop_record_id, executor="runpod", targets=targets, requested_at=requested_at, stopped_at=ended_at, outcome="stopped")

    def mutate(latest: dict[str, Any]) -> None:
        if latest.get("pod_id") != pod_id:
            return
        if latest.get("state") not in TERMINAL_STATES:
            latest.update({"state": "interrupted", "exit_code": None, "ended": ended_at})
        latest["pod_stopped_at"] = ended_at

    status = _mutate_run_status(run_dir, mutate)
    if realization_id:
        record_launch_phase(run_dir, realization_id, "pod_stopped", at=ended_at, pod_id=pod_id)
    append_run_event(run_dir, {"event": "runpod_pod_stopped", "timestamp": ended_at, "executor": "runpod", "pod_id": pod_id})
    return status
