"""
ResNet2D Spatial Encoder implementation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import AbsSpatialEncoder


class RIStack(nn.Module):
    """
    複素STFTをRIスタック（実数チャネル）に変換
    MC: [B, C, F, T] (complex) -> [B, 2*C, F, T] (real)
    SC: [B, 1, F, T] (complex) -> [B, 2, F, T] (real)
    """
    def forward(self, X):
        """
        Args:
            X: [B, C, F, T] - 複素STFT
        Returns:
            X_ri: [B, 2*C, F, T] - RIスタック（実数）
        """
        # Real and Imaginary parts
        X_real = X.real  # [B, C, F, T]
        X_imag = X.imag  # [B, C, F, T]
        
        # Stack along channel dimension
        X_ri = torch.cat([X_real, X_imag], dim=1)  # [B, 2*C, F, T]
        
        return X_ri


class ChannelProjection(nn.Module):
    """
    チャネル投影層（TF-GridNet参考：Conv2D + gLU + gLN）
    """
    def __init__(self, in_channels, out_channels, eps=1e-5):
        """
        Args:
            in_channels: 入力チャネル数（RIスタック後なので2*C_in）
            out_channels: 出力チャネル数（D）
            eps: gLN用のeps
        """
        super().__init__()
        self.out_channels = out_channels
        self.eps = eps
        
        # Conv2D: in=2*C_in, out=2*D
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        # gLN用のaffine parameters
        self.gamma = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, F, T]
        Returns:
            z: [B, out_channels, F, T]
        """
        # 1) Conv2D
        h = self.conv(x)  # [B, 2*D, F, T]
        
        # 2) gLU (channel split)
        A, B = torch.chunk(h, 2, dim=1)  # each: [B, D, F, T]
        z = A * torch.sigmoid(B)  # [B, D, F, T]
        
        # 3) gLN (global LayerNorm)
        # サンプルごとに (D, F, T) 全体で正規化
        mean = z.mean(dim=(1, 2, 3), keepdim=True)  # [B, 1, 1, 1]
        var = z.var(dim=(1, 2, 3), keepdim=True, unbiased=False)  # [B, 1, 1, 1]
        z_norm = (z - mean) / torch.sqrt(var + self.eps)  # [B, D, F, T]
        
        # Affine transform
        z = self.gamma * z_norm + self.beta  # [B, D, F, T]
        
        return z


class BasicBlock2D(nn.Module):
    """
    2D ResNet用のBasicBlock
    """
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResNet18Trunk(nn.Module):
    """
    スペクトログラム向けResNet18
    周波数のみdownsample（時間方向は維持）
    """
    def __init__(self, in_channels=32):
        super().__init__()
        
        # Stage1: ch=32, blocks=2, stride=(1,1)
        self.stage1 = self._make_stage(in_channels, 24, num_blocks=1, stride=(1, 1))
        
        # Stage2: ch=64, blocks=2, stride=(2,1) - 周波数のみdownsample
        self.stage2 = self._make_stage(24, 48, num_blocks=1, stride=(2, 1))

        # Stage3: ch=64, blocks=2, stride=(2,1) - 周波数のみdownsample
        self.stage3 = self._make_stage(48, 72, num_blocks=1, stride=(2, 1))
        
        # Stage4: ch=256, blocks=2, stride=(2,1)
        self.stage4 = self._make_stage(72, 128, num_blocks=1, stride=(2, 1))
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock2D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock2D(out_channels, out_channels, stride=(1, 1)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, D, F, T] (D=32)
        Returns:
            h: [B, 128, F', T'] (F'は段階的に縮小、T'はほぼ維持)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class GlobalStatsPool(nn.Module):
    """
    Global Statistics Pooling
    周波数・時間の両方でmean/stdを計算し、concat
    """
    def forward(self, x):
        """
        Args:
            x: [B, C, F, T]
        Returns:
            g: [B, 2*C] - [mu, std]のconcat
        """
        # Mean and std over (F, T) dimensions
        mu = x.mean(dim=(2, 3))  # [B, C]
        std = x.std(dim=(2, 3), unbiased=False)  # [B, C]
        
        # Concat
        g = torch.cat([mu, std], dim=1)  # [B, 2*C]
        
        return g


class EmbeddingHead(nn.Module):
    """
    Embedding Head: Linear + L2 normalize
    """
    def __init__(self, in_dim, embed_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.fc = nn.Linear(in_dim, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_dim] (e.g., 512 from GlobalStatsPool)
        Returns:
            e: [B, embed_dim] - L2正規化済み
        """
        e = self.fc(x)  # [B, embed_dim]
        # L2 normalize
        e = e / (torch.norm(e, p=2, dim=1, keepdim=True) + self.eps)
        return e


class SpatialResNetBranch(nn.Module):
    """
    2枝構成のSpatial Embeddingモデル
    - SC枝: 1ch入力 -> 埋め込み
    - MC枝: 2ch入力 -> 埋め込み
    - TrunkとEmbeddingHeadは共有
    """
    def __init__(
        self,
        num_channels_mc=2,
        projection_dim=32,
        embed_dim=256,
        eps_gln=1e-5,
        eps_embed=1e-8
    ):
        """
        Args:
            num_channels_mc: MCのチャネル数（固定、デフォルト2）
            projection_dim: チャネル投影のD（デフォルト32）
            embed_dim: 埋め込み次元（デフォルト256）
            eps_gln: gLN用eps（デフォルト1e-5）
            eps_embed: Embedding正規化用eps（デフォルト1e-8）
        """
        super().__init__()
        
        self.num_channels_mc = num_channels_mc
        self.projection_dim = projection_dim
        self.embed_dim = embed_dim
        
        # 前処理モジュール（共有）
        self.ri_stack = RIStack()
        
        # チャネル投影層（別々）
        # SC: in_channels=2 (1ch -> RIスタック後2ch)
        self.proj_sc = ChannelProjection(in_channels=2, out_channels=projection_dim, eps=eps_gln)
        
        # MC: in_channels=2*num_channels_mc (2ch -> RIスタック後4ch)
        self.proj_mc = ChannelProjection(in_channels=2*num_channels_mc, out_channels=projection_dim, eps=eps_gln)
        
        # Trunk（共有）
        self.trunk = ResNet18Trunk(in_channels=projection_dim)
        
        # Pooling + Embedding Head（共有）
        self.pooling = GlobalStatsPool()
        # Trunk出力は128チャネル、GlobalStatsPool後は256次元
        self.embedding_head = EmbeddingHead(in_dim=256, embed_dim=embed_dim, eps=eps_embed)
    
    def forward_sc(self, X_sc):
        """
        SC枝のforward
        Args:
            X_sc: [B, T, F] - STFT spectrum (complex)
        Returns:
            e_sc: [B, embed_dim] - 埋め込み（L2正規化済み）
        """
        # 入力形式変換: [B, T, F] -> [B, 1, F, T]
        # X_sc: [B, T, F] (complex)
        X = X_sc.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
        X = X.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]
        
        # RIスタック
        X_ri = self.ri_stack(X)  # [B, 2, F, T] (real)
        
        # チャネル投影
        z = self.proj_sc(X_ri)  # [B, D, F, T] (D=projection_dim)
        
        # Trunk
        h = self.trunk(z)  # [B, 128, F', T']
        
        # Pooling + Embedding
        g = self.pooling(h)  # [B, 256]
        e = self.embedding_head(g)  # [B, embed_dim]
        
        return e
    
    def forward_mc(self, X_mc):
        """
        MC枝のforward
        Args:
            X_mc: [B, T, C_mc, F] - STFT spectrum (complex, C_mc=2)
        Returns:
            e_mc: [B, embed_dim] - 埋め込み（L2正規化済み）
        """
        # 入力形式変換: [B, T, C_mc, F] -> [B, C_mc, F, T]
        # X_mc: [B, T, C_mc, F] (complex)
        X = X_mc.permute(0, 2, 3, 1)  # [B, T, C_mc, F] -> [B, C_mc, F, T]
        
        # RIスタック
        X_ri = self.ri_stack(X)  # [B, 2*C_mc, F, T] (real)
        
        # チャネル投影
        z = self.proj_mc(X_ri)  # [B, D, F, T] (D=projection_dim)
        
        # Trunk
        h = self.trunk(z)  # [B, 128, F', T']
        
        # Pooling + Embedding
        g = self.pooling(h)  # [B, 256]
        e = self.embedding_head(g)  # [B, embed_dim]
        
        return e
    
    def forward(self, X, num_channels=None):
        """
        デフォルトforward
        SCかMCかを自動判定（チャネル数で判定）
        Args:
            X: [B, T, F] or [B, T, C, F] - STFT spectrum (complex)
            num_channels: Number of channels (1 for SC, num_channels_mc for MC).
                         If None, automatically determined from input shape.
        Returns:
            e: [B, embed_dim] - 埋め込み
        """
        if num_channels is not None:
            if num_channels == 1:
                return self.forward_sc(X)
            elif num_channels == self.num_channels_mc:
                return self.forward_mc(X)
            else:
                raise ValueError(
                    f"Unexpected num_channels: {num_channels}. "
                    f"Expected 1 (SC) or {self.num_channels_mc} (MC)"
                )
        
        # Auto-detect from input shape
        if X.ndim == 3:
            # [B, T, F] -> SC
            return self.forward_sc(X)
        elif X.ndim == 4:
            # [B, T, C, F] -> MC
            C = X.shape[2]
            if C == 1:
                # [B, T, 1, F] -> squeeze to [B, T, F] and treat as SC
                return self.forward_sc(X.squeeze(2))
            elif C == self.num_channels_mc:
                return self.forward_mc(X)
            else:
                raise ValueError(
                    f"Unexpected number of channels: {C}. "
                    f"Expected 1 (SC) or {self.num_channels_mc} (MC)"
                )
        else:
            raise ValueError(f"Unexpected input dim: {X.ndim}. Expected 3 or 4")


class ResNet2DSpatialEncoder(AbsSpatialEncoder):
    """Spatial ResNet Branch Encoder"""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_channels_mc: int = 2,
    ):
        """Initialize ResNet2D Spatial Encoder.

        Args:
            embedding_dim: Embedding dimension
            num_channels_mc: Number of channels for multi-channel input (default: 2)
        """
        super().__init__()
        self.model = SpatialResNetBranch(
            num_channels_mc=num_channels_mc,
            projection_dim=32,  # Default from user's original code
            embed_dim=embedding_dim,
        )
        self.num_channels_mc = num_channels_mc
    
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        num_channels: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input: [B, T, F] or [B, T, C, F] - STFT spectrum (complex)
            ilens: [B] - input lengths (not used in this implementation)
            num_channels: Number of input channels (1 for SC, num_channels_mc for MC).
                         If None, automatically determined from input shape.
                         If specified, use this value to switch between SC/MC.
        Returns:
            embedding: [B, embed_dim] - spatial embedding
        """
        return self.model(input, num_channels=num_channels)
