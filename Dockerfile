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

# MV-SAM3D runtime deps (minimum practical set)
RUN micromamba run -n sam3d python -m pip install \
  omegaconf hydra-core loguru timm easydict astor==0.8.1 opencv-python trimesh \
  lightning==2.3.3 spconv-cu121==2.3.8 \
  && true

# Kaolin (fail-soft)
RUN micromamba run -n sam3d python -m pip install -U kaolin \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html \
  || true

# Git deps (fail-soft)
RUN micromamba run -n sam3d python -m pip install \
  "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
  "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
  || true

# PyTorch3D + gsplat (can be heavy; fail-soft, replace with wheels later if needed)
# TORCH_CUDA_ARCH_LIST covers: T4(7.5) A100(8.0) A10G(8.6) L4(8.9) H100(9.0)
RUN micromamba run -n sam3d bash -lc "\
  python -c 'import pytorch3d' 2>/dev/null && echo 'pytorch3d exists' || ( \
    export FORCE_CUDA=1 && export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0' && \
    pip install --no-build-isolation --no-deps \
      'git+https://github.com/facebookresearch/pytorch3d.git@stable' \
  )" || true

RUN micromamba run -n sam3d bash -lc "\
  python -c 'import gsplat' 2>/dev/null && echo 'gsplat exists' || ( \
    export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0' && \
    pip install --no-build-isolation \
      'git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7' \
  )" || true

# ---- clone repos ----
RUN git clone https://github.com/devinli123/MV-SAM3D.git /mv_sam3d \
 && git clone https://github.com/facebookresearch/sam-3d-objects.git /sam3d_upstream

# Patch missing visualization module in MV-SAM3D
RUN mkdir -p /mv_sam3d/sam3d_objects/utils \
 && cp -r /sam3d_upstream/sam3d_objects/utils/visualization /mv_sam3d/sam3d_objects/utils/visualization

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
COPY requirements.txt /workspace/requirements.txt
COPY README.md /workspace/README.md

EXPOSE 7860
CMD ["bash", "-lc", "micromamba run -n sam3d python /workspace/app.py"]

