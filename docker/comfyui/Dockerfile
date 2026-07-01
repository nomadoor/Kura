ARG PYTORCH_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_BASE}

ARG COMFYUI_REF=50e5270b86765bac2da70248d61050abba72b19f
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV COMFYUI_ROOT=/opt/ComfyUI

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFYUI_ROOT" \
    && cd "$COMFYUI_ROOT" \
    && git checkout "$COMFYUI_REF" \
    && pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY docker/comfyui/kura_comfy_prepare.py /opt/kura_comfy_prepare.py

WORKDIR /workspace
EXPOSE 8188
CMD ["python", "/opt/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"]
