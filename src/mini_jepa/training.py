from pathlib import Path

import torch
from torch.nn import functional as F

from mini_jepa.images import load_rgb_image
from mini_jepa.masks import load_mask_png
from mini_jepa.model import MiniJepa
from mini_jepa.patches import image_to_tensor, mask_to_patch_mask, patchify_image


def train_on_image(
    image_path: str | Path,
    mask_path: str | Path,
    patch_size: int = 16,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> list[float]:
    image = load_rgb_image(image_path)
    mask = load_mask_png(mask_path)
    image_tensor = image_to_tensor(image)
    patches = patchify_image(image_tensor, patch_size)
    hidden_patch_mask = mask_to_patch_mask(mask, patch_size)

    model = MiniJepa(
        patch_dim=patches.shape[1],
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_patches=patches.shape[0],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(epochs):
        optimizer.zero_grad()
        predictions, targets = model(patches, hidden_patch_mask)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    return losses

