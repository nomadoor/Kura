ARG PYTORCH_BASE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
FROM ${PYTORCH_BASE}

# The CUDA version the image runs; Kura records it with the pinned digest.
ARG KURA_CUDA=13.0
LABEL io.kura.cuda="${KURA_CUDA}"

ARG COMFYUI_REF=0f42ba51463174fb255f2c4605ae0e0b441fe6d7
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
# From torch 2.11 the base image installs torch into the system Python, not conda;
# this single-purpose image installs ComfyUI's requirements beside it.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV COMFYUI_ROOT=/opt/ComfyUI

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFYUI_ROOT" \
    && cd "$COMFYUI_ROOT" \
    && git checkout "$COMFYUI_REF" \
    && pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY docker/comfyui/kura_comfy_prepare.py /opt/kura_comfy_prepare.py

# Installing the trainer must not replace the base image's torch build.
RUN python -c "import sys, torch; cuda = torch.version.cuda or ''; sys.exit(0 if cuda.startswith('${KURA_CUDA}') else 'torch ' + torch.__version__ + ' uses CUDA ' + cuda + ', expected ${KURA_CUDA}')"

WORKDIR /workspace
EXPOSE 8188
CMD ["python", "/opt/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"]
