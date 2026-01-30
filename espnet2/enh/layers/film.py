import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    https://arxiv.org/abs/1709.07871
    """
    def __init__(self, embed_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(embed_dim, feature_dim)
        self.beta_proj = nn.Linear(embed_dim, feature_dim)

    def init_identity(self):
        """Initialize FiLM to pass-through (gamma=1, beta=0)."""
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, embedding: torch.Tensor, feature: torch.Tensor):
        """
        Args:
            embedding: [B, embed_dim] - Spatial embedding
            feature: [B, feature_dim, T, F] - Feature to modulate
        Returns:
            modulated_feature: [B, feature_dim, T, F]
        """
        gamma = self.gamma_proj(embedding)  # [B, feature_dim]
        beta = self.beta_proj(embedding)    # [B, feature_dim]
        
        # Expand to match feature dimensions
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, feature_dim, 1, 1]
        
        # FiLM: feature * gamma + beta
        return feature * gamma + beta


class TemporalFiLM(nn.Module):
    """Temporal Feature-wise Linear Modulation (FiLM).

    This applies FiLM per time frame, broadcasting across frequency.
    """

    def __init__(self, embed_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(embed_dim, feature_dim)
        self.beta_proj = nn.Linear(embed_dim, feature_dim)

    def init_identity(self):
        """Initialize FiLM to pass-through (gamma=1, beta=0)."""
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        embedding: torch.Tensor,
        feature: torch.Tensor,
        ilens: torch.Tensor = None,
    ):
        """
        Args:
            embedding: [B, T, embed_dim] - temporal spatial embedding
            feature: [B, feature_dim, T, F] - feature to modulate
            ilens: [B] - valid lengths (optional)
        Returns:
            modulated_feature: [B, feature_dim, T, F]
        """
        if embedding.ndim != 3:
            raise ValueError(
                f"embedding must be 3D [B, T, E], but got {embedding.shape}"
            )
        if feature.ndim != 4:
            raise ValueError(
                f"feature must be 4D [B, C, T, F], but got {feature.shape}"
            )
        if embedding.shape[0] != feature.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"{embedding.shape[0]} vs {feature.shape[0]}"
            )
        if embedding.shape[1] != feature.shape[2]:
            raise ValueError(
                "Time dimension mismatch: "
                f"{embedding.shape[1]} vs {feature.shape[2]}"
            )

        gamma = self.gamma_proj(embedding)  # [B, T, C]
        beta = self.beta_proj(embedding)  # [B, T, C]

        if ilens is not None:
            ilens = ilens.to(embedding.device).long()
            t = embedding.shape[1]
            mask = (torch.arange(t, device=embedding.device)[None, :] < ilens[:, None]).to(gamma.dtype)
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            # For padded frames: gamma=1, beta=0
            gamma = gamma * mask + (1.0 - mask)
            beta = beta * mask

        gamma = gamma.permute(0, 2, 1).unsqueeze(-1)  # [B, C, T, 1]
        beta = beta.permute(0, 2, 1).unsqueeze(-1)  # [B, C, T, 1]

        return feature * gamma + beta


class ConcatConditioner(nn.Module):
    """Concatenate temporal embedding and project with 1x1 conv.

    This expects temporal embeddings and broadcasts across frequency.
    """

    def __init__(self, embed_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(feature_dim + embed_dim, feature_dim, kernel_size=1)

    def forward(self, embedding: torch.Tensor, feature: torch.Tensor):
        """
        Args:
            embedding: [B, T, E] - temporal spatial embedding
            feature: [B, C, T, F] - feature to condition
        Returns:
            conditioned_feature: [B, C, T, F]
        """
        if embedding.ndim != 3:
            raise ValueError(
                f"embedding must be 3D [B, T, E], but got {embedding.shape}"
            )
        if feature.ndim != 4:
            raise ValueError(
                f"feature must be 4D [B, C, T, F], but got {feature.shape}"
            )
        if embedding.shape[0] != feature.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"{embedding.shape[0]} vs {feature.shape[0]}"
            )
        if embedding.shape[1] != feature.shape[2]:
            raise ValueError(
                "Time dimension mismatch: "
                f"{embedding.shape[1]} vs {feature.shape[2]}"
            )

        spatial = embedding.permute(0, 2, 1).unsqueeze(-1)  # [B, E, T, 1]
        spatial = spatial.expand(-1, -1, -1, feature.shape[-1])  # [B, E, T, F]
        concat = torch.cat([feature, spatial], dim=1)  # [B, C+E, T, F]
        return self.proj(concat)
