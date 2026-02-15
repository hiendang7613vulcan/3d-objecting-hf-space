import re
from pathlib import Path
from PIL import Image


def natural_key(s: str):
    """Natural sort key: '10.png' sorts after '2.png' correctly."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_rgb_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def save_rgb_png(img: Image.Image, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def make_rgba_with_alpha(rgb_img: Image.Image, alpha_mask: Image.Image) -> Image.Image:
    """
    rgb_img: RGB PIL
    alpha_mask: L PIL, same size or will be resized
    Returns RGBA PIL where alpha channel is the mask.
    """
    if rgb_img.mode != "RGB":
        rgb_img = rgb_img.convert("RGB")
    if alpha_mask.mode != "L":
        alpha_mask = alpha_mask.convert("L")
    if alpha_mask.size != rgb_img.size:
        alpha_mask = alpha_mask.resize(rgb_img.size, Image.NEAREST)
    rgba = rgb_img.copy()
    rgba.putalpha(alpha_mask)
    return rgba


def save_rgba_png(img: Image.Image, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")

