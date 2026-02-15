import os
import tempfile
from pathlib import Path
import gradio as gr

from src.checkpoints import ensure_sam3d_checkpoints, link_mv_sam3d_checkpoints
from src.fal_sam3 import extract_main_object_alpha
from src.mv_sam3d import run_mv_sam3d_inference
from src.utils import (
    natural_key,
    load_rgb_image,
    save_rgb_png,
    make_rgba_with_alpha,
    save_rgba_png,
)

MV_ROOT = Path("/mv_sam3d")


def build_mv_input_from_uploads(files, mask_prompt: str, prompt: str | None, pick_mode: str, max_masks: int, progress: gr.Progress):
    """
    Create MV-SAM3D input folder:
      input/
        images/0.png ...
        <mask_prompt>/0.png ...  (RGBA with alpha mask)
    """
    if not files:
        raise gr.Error("Upload at least 2 images (multi-view).")

    # Sort by filename (natural sort: 1,2,10)
    files_sorted = sorted(files, key=lambda f: natural_key(Path(f).name))

    workdir = Path(tempfile.mkdtemp(prefix="mv_input_"))
    input_dir = workdir / "input"
    images_dir = input_dir / "images"
    masks_dir = input_dir / mask_prompt
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    n = len(files_sorted)
    for i, fp in enumerate(files_sorted):
        progress((i + 1) / max(n, 1), desc=f"SAM-3 masking view {i+1}/{n}")

        img = load_rgb_image(fp)

        # fal SAM-3 main object alpha mask (L, 0..255)
        alpha = extract_main_object_alpha(
            image_path=fp,
            prompt=(prompt.strip() if prompt and prompt.strip() else None),
            pick_mode=pick_mode,
            max_masks=max_masks,
        )

        # Save RGB image to images/{i}.png
        rgb_path = images_dir / f"{i}.png"
        save_rgb_png(img, rgb_path)

        # Save RGBA mask image to <mask_prompt>/{i}.png (alpha channel stores mask)
        rgba = make_rgba_with_alpha(img, alpha)
        mask_path = masks_dir / f"{i}.png"
        save_rgba_png(rgba, mask_path)

    return input_dir, [str((images_dir / f"{i}.png")) for i in range(n)]


def run_pipeline(
    files,
    mask_prompt,
    sam3_prompt,
    pick_mode,
    max_masks,
    image_names,
    stage1_weighting,
    stage2_weighting,
    stage2_weight_source,
    da3_npz,
    progress=gr.Progress(),
):
    # 1) ensure checkpoints in /data and link into MV repo
    progress(0.02, desc="Ensuring SAM-3D checkpoints...")
    ensure_sam3d_checkpoints()
    link_mv_sam3d_checkpoints(MV_ROOT)

    # 2) build input folder using fal SAM-3 masks
    progress(0.05, desc="Preparing multi-view input...")
    input_dir, preview_imgs = build_mv_input_from_uploads(
        files=files,
        mask_prompt=mask_prompt,
        prompt=sam3_prompt,
        pick_mode=pick_mode,
        max_masks=int(max_masks),
        progress=progress,
    )

    # 3) run MV-SAM3D inference
    progress(0.75, desc="Running MV-SAM3D inference (GPU)...")
    out = run_mv_sam3d_inference(
        mv_root=MV_ROOT,
        input_dir=input_dir,
        mask_prompt=mask_prompt,
        image_names=image_names,
        stage1_weighting=stage1_weighting,
        stage2_weighting=stage2_weighting,
        stage2_weight_source=stage2_weight_source,
        da3_npz_path=(Path(da3_npz) if da3_npz else None),
    )

    # 4) outputs
    progress(1.0, desc="Done!")
    return (
        out.viewer_path,
        out.glb_path,
        out.ply_path,
        out.npz_path,
        out.log_tail,
        preview_imgs,
    )


with gr.Blocks() as demo:
    gr.Markdown(
        """
# MV-SAM3D + fal SAM-3 (Main Object Mask)
Upload multi-view images → fal SAM-3 extracts main object mask per view → MV-SAM3D outputs GLB + PLY.

**Secrets required:** `FAL_KEY` and (maybe) `HF_TOKEN`.
"""
    )

    with gr.Row():
        files = gr.Files(label="Multi-view images (PNG/JPG)", file_types=["image"])
        da3_npz = gr.File(label="(Optional) DA3 output (.npz) for visibility/mixed", file_types=[".npz"])

    with gr.Row():
        mask_prompt = gr.Textbox(label="mask_prompt folder name", value="object")
        sam3_prompt = gr.Textbox(label="SAM-3 prompt (optional, e.g. 'stuffed toy' / 'person')", value="")

    with gr.Row():
        pick_mode = gr.Dropdown(label="Pick main object mode", choices=["largest", "best_score"], value="largest")
        max_masks = gr.Slider(label="SAM-3 max_masks", minimum=1, maximum=10, value=5, step=1)

    gr.Markdown("### MV-SAM3D params")

    with gr.Row():
        image_names = gr.Textbox(label="image_names (comma-separated, optional)", placeholder="0,1,2,3,4,5,6,7")

    with gr.Row():
        stage1_weighting = gr.Checkbox(label="Stage 1 weighting", value=False)
        stage2_weighting = gr.Checkbox(label="Stage 2 weighting", value=False)
        stage2_weight_source = gr.Dropdown(
            label="Stage 2 weight source",
            choices=["entropy", "visibility", "mixed"],
            value="entropy",
        )

    run_btn = gr.Button("Run", variant="primary")

    with gr.Row():
        viewer = gr.Model3D(label="Preview (GLB)")
        preview_gallery = gr.Gallery(label="Prepared inputs (images/*.png)", columns=4, height=240)

    with gr.Row():
        glb_dl = gr.File(label="result.glb")
        ply_dl = gr.File(label="result.ply")
        npz_dl = gr.File(label="params.npz")

    log_box = gr.Textbox(label="Log tail", lines=18)

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            files,
            mask_prompt,
            sam3_prompt,
            pick_mode,
            max_masks,
            image_names,
            stage1_weighting,
            stage2_weighting,
            stage2_weight_source,
            da3_npz,
        ],
        outputs=[viewer, glb_dl, ply_dl, npz_dl, log_box, preview_gallery],
    )

if __name__ == "__main__":
    demo.launch()

