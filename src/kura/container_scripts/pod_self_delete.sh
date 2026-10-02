# Kura's Pod-side self-delete, sourced into every Pod-side guard (unattended
# wait and maximum lease, for training and render Pods).
#
# An SSH session does not inherit the Pod's create-time environment, where
# RunPod injects RUNPOD_POD_ID and a Pod-scoped RUNPOD_API_KEY, so both are read
# from the init process. The function calls the GraphQL mutation the controller
# uses (podTerminate), then podStop if termination is refused. It avoids the
# Pod's preinstalled runpodctl, whose command syntax follows the version RunPod
# ships, and sends its own User-Agent because RunPod's edge rejects Python's.
kura_pod_self_delete() {
  kura_log="$1"
  kura_env_file="${KURA_POD_ENV_FILE:-/proc/1/environ}"
  if [ -z "${RUNPOD_API_KEY:-}" ] && [ -r "$kura_env_file" ]; then
    RUNPOD_API_KEY=$(tr '\000' '\n' < "$kura_env_file" | sed -n 's/^RUNPOD_API_KEY=//p' | head -n 1)
    export RUNPOD_API_KEY
  fi
  if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r "$kura_env_file" ]; then
    RUNPOD_POD_ID=$(tr '\000' '\n' < "$kura_env_file" | sed -n 's/^RUNPOD_POD_ID=//p' | head -n 1)
    export RUNPOD_POD_ID
  fi
  kura_python=$(command -v python3 || command -v python || true)
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "$kura_python" ]; then
    echo "[kura] Pod self-delete unavailable: RUNPOD_API_KEY, RUNPOD_POD_ID, or python is missing" >> "$kura_log" 2>&1 || true
    return 1
  fi
  "$kura_python" - "$RUNPOD_POD_ID" >> "$kura_log" 2>&1 <<'KURA_POD_SELF_DELETE'
import json, os, sys, urllib.request

pod_id = sys.argv[1]
url = os.environ.get("KURA_RUNPOD_GRAPHQL_URL", "https://api.runpod.io/graphql")
mutations = (
    ("podTerminate", "mutation terminatePod($podId: String!) { podTerminate(input: {podId: $podId}) }"),
    ("podStop", "mutation stopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }"),
)
for name, query in mutations:
    request = urllib.request.Request(
        url,
        data=json.dumps({"query": query, "variables": {"podId": pod_id}}).encode("utf-8"),
        method="POST",
        # RunPod's edge rejects Python's default User-Agent before checking the key.
        headers={"Content-Type": "application/json", "User-Agent": "Kura-pod-guard"},
    )
    request.add_unredirected_header("Authorization", "Bearer " + os.environ["RUNPOD_API_KEY"])
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8") or "{}")
    except Exception as exc:
        detail = ""
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                detail = ""
        print(f"[kura] Pod self-delete {name} failed: {getattr(exc, 'code', None) or type(exc).__name__} {detail}".rstrip(), flush=True)
        continue
    if body.get("errors"):
        messages = "; ".join(str(error.get("message", "")) for error in body["errors"] if isinstance(error, dict))
        print(f"[kura] Pod self-delete {name} was refused: {messages[:200]}", flush=True)
        continue
    print(f"[kura] Pod self-delete {name} accepted", flush=True)
    sys.exit(0)
sys.exit(1)
KURA_POD_SELF_DELETE
}
