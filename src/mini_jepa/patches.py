import numpy as np
import torch
from PIL import Image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    data = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(data).permute(2, 0, 1)


def patchify_image(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    validate_image_tensor(image, patch_size)
    channels, height, width = image.shape
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, channels * patch_size * patch_size)


def mask_to_patch_mask(mask: np.ndarray, patch_size: int) -> torch.Tensor:
    validate_mask(mask, patch_size)
    height, width = mask.shape
    patch_rows = height // patch_size
    patch_cols = width // patch_size
    patch_mask = mask.reshape(patch_rows, patch_size, patch_cols, patch_size)
    patch_mask = patch_mask.any(axis=(1, 3))
    return torch.from_numpy(patch_mask.reshape(-1))


def validate_image_tensor(image: torch.Tensor, patch_size: int) -> None:
    if image.ndim != 3:
        raise ValueError("image tensor must have shape [channels, height, width]")

    _, height, width = image.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image size must be divisible by patch_size")


def validate_mask(mask: np.ndarray, patch_size: int) -> None:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [height, width]")

    height, width = mask.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("mask size must be divisible by patch_size")

