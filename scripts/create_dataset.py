from pathlib import Path

import numpy as np
from PIL import Image


SIZE = 128


def save(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def gradient_h() -> np.ndarray:
    x = np.linspace(0, 255, SIZE, dtype=np.float32)
    r = x[None, :].repeat(SIZE, axis=0)
    g = np.zeros((SIZE, SIZE), dtype=np.float32)
    b = 255 - r
    return np.stack([r, g, b], axis=-1)


def gradient_v() -> np.ndarray:
    y = np.linspace(0, 255, SIZE, dtype=np.float32)
    g = y[:, None].repeat(SIZE, axis=1)
    r = np.zeros((SIZE, SIZE), dtype=np.float32)
    b = 255 - g
    return np.stack([r, g, b], axis=-1)


def gradient_diagonal() -> np.ndarray:
    x = np.linspace(0, 1, SIZE, dtype=np.float32)
    y = np.linspace(0, 1, SIZE, dtype=np.float32)
    d = (x[None, :] + y[:, None]) / 2
    r = (255 * d).clip(0, 255)
    g = (255 * (1 - d)).clip(0, 255)
    b = (255 * np.abs(d - 0.5) * 2).clip(0, 255)
    return np.stack([r, g, b], axis=-1)


def gradient_radial() -> np.ndarray:
    cx, cy = SIZE / 2, SIZE / 2
    x = np.arange(SIZE, dtype=np.float32)
    y = np.arange(SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = (dist / dist.max()).clip(0, 1)
    r = (255 * dist).clip(0, 255)
    g = (255 * (1 - dist)).clip(0, 255)
    b = (128 * np.ones((SIZE, SIZE), dtype=np.float32))
    return np.stack([r, g, b], axis=-1)


def checkerboard() -> np.ndarray:
    block = 16
    x = np.arange(SIZE) // block
    y = np.arange(SIZE) // block
    pattern = ((x[None, :] + y[:, None]) % 2).astype(np.float32)
    r = 255 * pattern
    g = 128 * np.ones((SIZE, SIZE), dtype=np.float32)
    b = 255 * (1 - pattern)
    return np.stack([r, g, b], axis=-1)


def stripes_h() -> np.ndarray:
    stripe = 16
    y = np.arange(SIZE, dtype=np.float32)
    pattern = ((y // stripe) % 2)[:, None].repeat(SIZE, axis=1)
    r = 255 * pattern
    g = 100 * np.ones((SIZE, SIZE), dtype=np.float32)
    b = 200 * (1 - pattern)
    return np.stack([r, g, b], axis=-1)


def stripes_v() -> np.ndarray:
    stripe = 16
    x = np.arange(SIZE, dtype=np.float32)
    pattern = ((x // stripe) % 2)[None, :].repeat(SIZE, axis=0)
    r = 50 * np.ones((SIZE, SIZE), dtype=np.float32)
    g = 255 * pattern
    b = 200 * (1 - pattern)
    return np.stack([r, g, b], axis=-1)


def rings() -> np.ndarray:
    cx, cy = SIZE / 2, SIZE / 2
    x = np.arange(SIZE, dtype=np.float32)
    y = np.arange(SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    pattern = (dist // 12 % 2).astype(np.float32)
    r = 200 * pattern
    g = 100 * np.ones((SIZE, SIZE), dtype=np.float32)
    b = 255 * (1 - pattern)
    return np.stack([r, g, b], axis=-1)


def noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)).astype(np.float32)


def cross() -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    thickness = 20
    mid = SIZE // 2
    img[mid - thickness // 2 : mid + thickness // 2, :] = [255, 200, 0]
    img[:, mid - thickness // 2 : mid + thickness // 2] = [0, 180, 255]
    return img


IMAGES = [
    ("gradient_h", gradient_h()),
    ("gradient_v", gradient_v()),
    ("gradient_diagonal", gradient_diagonal()),
    ("gradient_radial", gradient_radial()),
    ("checkerboard", checkerboard()),
    ("stripes_h", stripes_h()),
    ("stripes_v", stripes_v()),
    ("rings", rings()),
    ("noise", noise(seed=42)),
    ("cross", cross()),
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "images"
    for name, image in IMAGES:
        save(image, out / f"{name}.png")
        print(f"created {out / name}.png")


if __name__ == "__main__":
    main()
