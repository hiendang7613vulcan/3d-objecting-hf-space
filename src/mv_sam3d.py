import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MVResult:
    outdir: Path
    viewer_path: str
    glb_path: Optional[str]
    ply_path: Optional[str]
    npz_path: Optional[str]
    log_tail: str


def run_mv_sam3d_inference(
    mv_root: Path,
    input_dir: Path,
    mask_prompt: str,
    image_names: str,
    stage1_weighting: bool,
    stage2_weighting: bool,
    stage2_weight_source: str,
    da3_npz_path: Optional[Path] = None,
) -> MVResult:
    """
    Runs:
      python run_inference_weighted.py --input_path <input_dir> --mask_prompt <mask_prompt> ...
    Parses the output directory from logs and returns paths.
    """
    args = [
        "python", "run_inference_weighted.py",
        "--input_path", str(input_dir),
        "--mask_prompt", mask_prompt,
    ]

    if image_names and image_names.strip():
        args += ["--image_names", image_names.strip()]

    if not stage1_weighting:
        args += ["--no_stage1_weighting"]

    if not stage2_weighting:
        args += ["--no_stage2_weighting"]
    else:
        args += ["--stage2_weight_source", stage2_weight_source]
        if stage2_weight_source in ("visibility", "mixed"):
            if da3_npz_path is None:
                raise RuntimeError("visibility/mixed requires DA3 output .npz")
            args += ["--da3_output", str(da3_npz_path)]

    proc = subprocess.run(
        args,
        cwd=str(mv_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = proc.stdout
    tail = log[-12000:]

    if proc.returncode != 0:
        raise RuntimeError("MV-SAM3D failed:\n\n" + tail)

    # Try parse "All output files saved to: <dir>"
    m = re.search(r"All output files saved to:\s*(.+)\s*", log)
    if m:
        outdir = Path(m.group(1).strip())
    else:
        # fallback: parse GLB path
        m2 = re.search(r"GLB file saved to:\s*(.+result\.glb)", log)
        if not m2:
            raise RuntimeError("Cannot locate output paths in logs:\n\n" + tail)
        outdir = Path(m2.group(1)).parent

    # FIX: Output path is relative to mv_root (subprocess cwd), make it absolute
    if not outdir.is_absolute():
        outdir = mv_root / outdir

    glb = outdir / "result.glb"
    ply = outdir / "result.ply"
    npz = outdir / "params.npz"

    viewer = str(glb) if glb.exists() else (str(ply) if ply.exists() else "")
    if not viewer:
        raise RuntimeError(f"Output missing in {outdir}\n\n" + tail)

    return MVResult(
        outdir=outdir,
        viewer_path=viewer,
        glb_path=str(glb) if glb.exists() else None,
        ply_path=str(ply) if ply.exists() else None,
        npz_path=str(npz) if npz.exists() else None,
        log_tail=tail,
    )

