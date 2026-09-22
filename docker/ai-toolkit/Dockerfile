ARG AI_TOOLKIT_IMAGE=ostris/aitoolkit:0.13.18@sha256:9bc99d51efc5b6c38a951b3bf8547bda0f9db58abeb75573548d449f82b34bcc
FROM ${AI_TOOLKIT_IMAGE}

ARG AI_TOOLKIT_IMAGE
ARG AI_TOOLKIT_H3_GRADIENT_PATCH_COMMIT=d1985f9bf380b6ce1c409b7875e2d367df486e19
ARG AI_TOOLKIT_H3_GRADIENT_PATCH_SHA256=3aecfda6af403496d0858bccf26a6d1e9764b2244ae3a526113827fe2591ddf6
ARG AI_TOOLKIT_H3_GRADIENT_PATCH_ADJUSTMENT=checkpoint-static-nonfinite-masking-v1

# Kura's image intentionally extends the official Ostris AI Toolkit RunPod image
# instead of rebuilding AI Toolkit from a different base. Keep the official
# layout (/app/ai-toolkit, /start.sh, port 8675, sshd support) intact.

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV KURA_AI_TOOLKIT_BASE_IMAGE=${AI_TOOLKIT_IMAGE}

COPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py
COPY docker/ai-toolkit/minimax_h3_finite_gradients.patch /tmp/minimax_h3_finite_gradients.patch

RUN echo "${AI_TOOLKIT_H3_GRADIENT_PATCH_SHA256}  /tmp/minimax_h3_finite_gradients.patch" | sha256sum -c - \
    && cd /app/ai-toolkit \
    && if git apply --check /tmp/minimax_h3_finite_gradients.patch; then \
        git apply /tmp/minimax_h3_finite_gradients.patch; \
        patch_status=applied; \
    elif git apply --reverse --check /tmp/minimax_h3_finite_gradients.patch; then \
        patch_status=already-present; \
    else \
        echo "MiniMax-H3 gradient patch is incompatible with the selected AI-Toolkit source" >&2; \
        exit 1; \
    fi \
    && printf '{"upstream_commit":"%s","sha256":"%s","kura_adjustment":"%s","status":"%s"}\n' \
        "${AI_TOOLKIT_H3_GRADIENT_PATCH_COMMIT}" \
        "${AI_TOOLKIT_H3_GRADIENT_PATCH_SHA256}" \
        "${AI_TOOLKIT_H3_GRADIENT_PATCH_ADJUSTMENT}" \
        "${patch_status}" \
        > /opt/kura-ai-toolkit-source-patch.json \
    && rm /tmp/minimax_h3_finite_gradients.patch

RUN ln -sfn /app/ai-toolkit /opt/ai-toolkit \
    && python - <<'PY'
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

packages = {}
for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

try:
    import torch
    cuda = torch.version.cuda
except Exception:
    cuda = None

ai_toolkit_commit = subprocess.run(
    ["git", "-C", "/app/ai-toolkit", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
source_patches = [json.loads(pathlib.Path("/opt/kura-ai-toolkit-source-patch.json").read_text(encoding="utf-8"))]

pathlib.Path("/opt/kura-runtime.json").write_text(
    json.dumps(
        {
            "base_image": os.environ["KURA_AI_TOOLKIT_BASE_IMAGE"],
            "ai_toolkit_path": "/app/ai-toolkit",
            "ai_toolkit_commit": ai_toolkit_commit,
            "source_patches": source_patches,
            "python": sys.version.split()[0],
            "packages": packages,
            "cuda": cuda,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

WORKDIR /workspace

# Preserve the official image's startup behavior for interactive RunPod use.
CMD ["/start.sh"]
