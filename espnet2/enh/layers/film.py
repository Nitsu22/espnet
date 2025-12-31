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

