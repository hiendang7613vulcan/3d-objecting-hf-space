FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential g++ \
    libgl1-mesa-glx libglib2.0-0 libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- micromamba (py311) ----
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /opt bin/micromamba \
  && ln -s /opt/bin/micromamba /usr/local/bin/micromamba

SHELL ["/bin/bash", "-lc"]

RUN micromamba create -y -n sam3d -c conda-forge python=3.11 pip \
  && micromamba clean -a -y

# IMPORTANT: pin setuptools FIRST
RUN micromamba run -n sam3d python -m pip install --force-reinstall "setuptools==81.0.0"

# Torch CUDA 12.1 (matches MV-SAM3D common setup)
RUN micromamba run -n sam3d python -m pip install \
  torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# UI + fal + core libs
RUN micromamba run -n sam3d python -m pip install \
  gradio==5.49.0 fal-client huggingface-hub requests Pillow numpy

# Kaolin (fail-soft)
RUN micromamba run -n sam3d python -m pip install -U kaolin \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html \
  || true

# Git deps (fail-soft)
RUN micromamba run -n sam3d python -m pip install \
  "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
  "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
  || true

# MV-SAM3D full requirements
COPY MV-SAM3D/my_requirements.txt /tmp/my_requirements.txt
RUN micromamba run -n sam3d pip install -r /tmp/my_requirements.txt || true

# PyTorch3D + gsplat from prebuilt wheels
COPY MV-SAM3D/wheels /tmp/wheels
RUN micromamba run -n sam3d pip install --no-deps /tmp/wheels/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl
RUN micromamba run -n sam3d pip install /tmp/wheels/gsplat-1.5.3-cp311-cp311-linux_x86_64.whl
RUN rm -rf /tmp/wheels

# ---- copy MV-SAM3D repo into image ----
COPY MV-SAM3D /mv_sam3d

# ---- runtime env ----
ENV HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf \
    TORCH_HOME=/data/torch \
    XDG_CACHE_HOME=/data/cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONPATH=/mv_sam3d:/mv_sam3d/notebook

WORKDIR /workspace
COPY src /workspace/src
COPY app.py /workspace/app.py
COPY MV-SAM3D/my_requirements.txt /workspace/my_requirements.txt
COPY README.md /workspace/README.md

EXPOSE 7860
CMD ["bash", "-lc", "micromamba run -n sam3d python /workspace/app.py"]
