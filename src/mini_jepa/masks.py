from pathlib import Path

import numpy as np
from PIL import Image


LCG_MULTIPLIER = 6364136223846793005
U64_MASK = (1 << 64) - 1


def generate_block_mask(
    width: int,
    height: int,
    block_size: int,
    blocks: int,
    seed: int,
) -> np.ndarray:
    validate_mask_args(width, height, block_size)

    mask = np.zeros((height, width), dtype=bool)
    max_x = width - block_size + 1
    max_y = height - block_size + 1

    for _ in range(blocks):
        seed = next_random(seed)
        x = seed % max_x
        seed = next_random(seed)
        y = seed % max_y
        mask[y : y + block_size, x : x + block_size] = True

    return mask


def load_mask_png(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    data = np.array(image)
    return data == 0


def save_mask_png(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.where(mask, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def next_random(seed: int) -> int:
    return (seed * LCG_MULTIPLIER + 1) & U64_MASK


def validate_mask_args(width: int, height: int, block_size: int) -> None:
    if width <= 0:
        raise ValueError("width must be greater than 0")

    if height <= 0:
        raise ValueError("height must be greater than 0")

    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")

    if block_size > width or block_size > height:
        raise ValueError("block_size must fit inside the mask")
