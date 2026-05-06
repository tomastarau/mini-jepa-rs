from pathlib import Path

import numpy as np
from PIL import Image


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.65) -> Image.Image:
    image = image.convert("RGB")
    image_data = np.array(image, dtype=np.float32)

    if mask.shape != image_data.shape[:2]:
        raise ValueError("mask shape must match image size")

    overlay = image_data.copy()
    overlay[mask] = [255, 32, 32]
    blended = image_data * (1 - alpha) + overlay * alpha

    return Image.fromarray(blended.astype(np.uint8))


def save_visualization(image: Image.Image, mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_mask(image, mask).save(path)
