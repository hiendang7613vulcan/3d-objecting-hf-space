import io
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import requests
from PIL import Image
import fal_client

PickMode = Literal["largest", "best_score"]


def _download_png(url: str, timeout: int = 120) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def _mask_to_alpha(mask_img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Convert returned mask PNG into binary alpha (L: 0 or 255) resized to original size.
    """
    m = mask_img.convert("L").resize(size, Image.NEAREST)
    arr = np.array(m)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def extract_main_object_alpha(
    image_path: str | Path,
    prompt: Optional[str] = None,
    pick_mode: PickMode = "largest",
    max_masks: int = 5,
) -> Image.Image:
    """
    Calls fal SAM-3 segmentation and returns alpha mask (L, 0..255) for the main object.
    Requires env var FAL_KEY to be set.
    """
    image_path = str(image_path)
    img = Image.open(image_path)
    W, H = img.size

    # Upload file to fal CDN
    image_url = fal_client.upload_file(image_path)

    args: dict = {
        "image_url": image_url,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": int(max_masks),
        "include_scores": True,
        "include_boxes": True,
    }
    if prompt:
        args["prompt"] = prompt

    # SAM-3 endpoint (fal) — use subscribe for queue-based robustness
    out = fal_client.subscribe("fal-ai/sam-3/image", arguments=args)

    masks = out.get("masks", [])  # list of {url: "...", width: ..., height: ...}
    scores = out.get("scores", None)

    if not masks:
        raise RuntimeError("fal SAM-3 returned no masks")

    candidates = []
    for i, m in enumerate(masks):
        mask_img = _download_png(m["url"])
        alpha = _mask_to_alpha(mask_img, (W, H))
        area = int(np.array(alpha).sum() // 255)
        score = None
        if isinstance(scores, list) and i < len(scores):
            try:
                score = float(scores[i])
            except Exception:
                score = None
        candidates.append((alpha, area, score))

    if pick_mode == "best_score" and any(c[2] is not None for c in candidates):
        alpha, area, score = max(candidates, key=lambda x: x[2] if x[2] is not None else -1.0)
    else:
        # Default: pick largest mask by pixel area
        alpha, area, score = max(candidates, key=lambda x: x[1])

    return alpha

