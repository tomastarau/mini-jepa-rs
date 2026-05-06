import torch
from torch import nn


class MiniJepa(nn.Module):
    def __init__(self, patch_dim: int, embed_dim: int, hidden_dim: int, max_patches: int) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.position_embedding = nn.Embedding(max_patches, embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def forward(self, patches: torch.Tensor, hidden_patch_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context_embeddings = self.context_encoder(patches)
        hidden_indices = hidden_patch_mask.nonzero(as_tuple=True)[0]
        visible_embeddings = context_embeddings[~hidden_patch_mask]

        if hidden_indices.numel() == 0:
            raise ValueError("mask must hide at least one patch")

        if visible_embeddings.shape[0] == 0:
            raise ValueError("mask must leave at least one visible patch")

        context = visible_embeddings.mean(dim=0, keepdim=True)
        positions = self.position_embedding(hidden_indices)
        prediction_input = context + positions
        predictions = self.predictor(prediction_input)

        with torch.no_grad():
            targets = self.target_encoder(patches[hidden_indices])

        return predictions, targets
