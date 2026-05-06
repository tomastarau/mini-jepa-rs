from pathlib import Path

import torch
from torch.nn import functional as F

from mini_jepa.images import load_images_from_dir, load_rgb_image
from mini_jepa.masks import generate_block_mask, load_mask_png
from mini_jepa.model import MiniJepa
from mini_jepa.patches import image_to_tensor, mask_to_patch_mask, patchify_image


def train_on_dir(
    image_dir: str | Path,
    patch_size: int = 16,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    image_size: int = 128,
    block_size: int = 24,
    blocks: int = 6,
    seed: int = 42,
) -> list[float]:
    images = load_images_from_dir(image_dir, size=(image_size, image_size))

    patch_samples = []
    for image in images:
        image_tensor = image_to_tensor(image)
        patch_samples.append(patchify_image(image_tensor, patch_size))

    patch_dim = patch_samples[0].shape[1]
    max_patches = patch_samples[0].shape[0]
    model = MiniJepa(patch_dim=patch_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, max_patches=max_patches)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    mask_seed = seed

    for _ in range(epochs):
        epoch_loss = 0.0
        for patches in patch_samples:
            mask = generate_block_mask(image_size, image_size, block_size, blocks, mask_seed)
            hidden_patch_mask = mask_to_patch_mask(mask, patch_size)
            mask_seed += 1
            optimizer.zero_grad()
            predictions, targets = model(patches, hidden_patch_mask)
            loss = F.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / len(patch_samples))

    return losses


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

