from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")

    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)

    return image


def create_demo_image(width: int, height: int) -> Image.Image:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    red = (255 * x).astype(np.uint8)
    green = (255 * y).astype(np.uint8)
    blue = (255 * (1 - x * y)).astype(np.uint8)

    image = np.stack(
        [
            np.broadcast_to(red, (height, width)),
            np.broadcast_to(green, (height, width)),
            blue,
        ],
        axis=-1,
    )

    return Image.fromarray(image)


def save_image(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
