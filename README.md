---
title: MV-SAM3D + fal SAM-3
emoji: 🧊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: l4x1
suggested_storage: small
pinned: false
---

# MV-SAM3D + fal SAM-3 (Main Object Mask) — Hugging Face Space

This Space:
- Upload multiple images (multi-view)
- Calls fal SAM-3 to extract main object mask per image
- Runs MV-SAM3D multi-view inference
- Outputs GLB (preview) + PLY (gaussian splat) + params.npz (download)

## Required Secrets (HF Space → Settings → Secrets)
- FAL_KEY: fal.ai API key
- HF_TOKEN: Hugging Face token (only if facebook/sam-3d-objects checkpoints are gated)

## Persistent Storage (recommended)
Enable "Persistent Storage" in the Space settings so `/data` persists.
This caches checkpoints and torch/hf caches to make restart faster.

## Local Run (optional)
You need a GPU + CUDA compatible environment; Docker build is recommended.
