FROM ostris/aitoolkit:latest

# Kura's image intentionally extends the official Ostris AI Toolkit RunPod image
# instead of rebuilding AI Toolkit from a different base. Keep the official
# layout (/app/ai-toolkit, /start.sh, port 8675, sshd support) intact.

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5

COPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py

RUN ln -sfn /app/ai-toolkit /opt/ai-toolkit \
    && python - <<'PY'
import importlib.metadata
import json
import pathlib
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

pathlib.Path("/opt/kura-runtime.json").write_text(
    json.dumps(
        {
            "base_image": "ostris/aitoolkit:latest",
            "ai_toolkit_path": "/app/ai-toolkit",
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
