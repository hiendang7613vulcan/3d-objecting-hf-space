import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Persisted (if HF persistent storage enabled)
CKPT_DIR = Path("/data/checkpoints/hf")
PIPELINE = CKPT_DIR / "pipeline.yaml"
SSGEN = CKPT_DIR / "ss_generator.yaml"


def ensure_sam3d_checkpoints():
    """
    Ensure pipeline.yaml + ss_generator.yaml exist in /data/checkpoints/hf.
    If missing, download from facebook/sam-3d-objects (repo_type=model).
    For gated repos, set HF_TOKEN in Space secrets.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if PIPELINE.exists() and SSGEN.exists():
        return

    token = os.environ.get("HF_TOKEN") or None

    local = snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        allow_patterns=["checkpoints/*"],
        token=token,
    )

    src = Path(local) / "checkpoints"
    if not src.exists():
        raise RuntimeError("Downloaded snapshot missing 'checkpoints/' folder")

    # Copy every file from checkpoints/ into CKPT_DIR
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, CKPT_DIR / p.name)

    if not PIPELINE.exists():
        raise RuntimeError("pipeline.yaml still missing after download")
    if not SSGEN.exists():
        raise RuntimeError("ss_generator.yaml still missing after download")


def link_mv_sam3d_checkpoints(mv_root: Path):
    """
    MV-SAM3D expects:
      /mv_sam3d/checkpoints/hf/pipeline.yaml
      /mv_sam3d/checkpoints/hf/ss_generator.yaml
    Create symlinks to the /data copies.
    """
    dst_dir = mv_root / "checkpoints" / "hf"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in ["pipeline.yaml", "ss_generator.yaml"]:
        src = CKPT_DIR / name
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

