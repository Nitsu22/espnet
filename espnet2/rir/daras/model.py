import math
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _build_mamba_block(dim: int, d_state: int, d_conv: int) -> nn.Module:
    try:
        from mamba_ssm import Mamba
    except Exception as e:
        raise ImportError(
            "DARAS MASS-BRPE requires mamba_ssm in the runtime environment."
        ) from e
    return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)


class GammatoneFeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        num_bands: int = 20,
        f_min: float = 50.0,
        f_max: float = 2000.0,
        filter_length: int = 401,
        hop_length: int = 160,
        order: int = 4,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.num_bands = int(num_bands)
        self.hop_length = int(hop_length)
        self.filter_length = int(filter_length)
        if self.filter_length % 2 == 0:
            self.filter_length += 1
        real, imag = self._build_filters(
            sample_rate=self.sample_rate,
            num_bands=self.num_bands,
            f_min=float(f_min),
            f_max=float(f_max),
            filter_length=self.filter_length,
            order=int(order),
        )
        self.register_buffer("real_filters", real)
        self.register_buffer("imag_filters", imag)

    @staticmethod
    def _hz_to_erb_rate(freq: torch.Tensor) -> torch.Tensor:
        return 21.4 * torch.log10(4.37e-3 * freq + 1.0)

    @staticmethod
    def _erb_rate_to_hz(erb: torch.Tensor) -> torch.Tensor:
        return (torch.pow(10.0, erb / 21.4) - 1.0) / 4.37e-3

    @classmethod
    def _build_filters(
        cls,
        sample_rate: int,
        num_bands: int,
        f_min: float,
        f_max: float,
        filter_length: int,
        order: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        erb_min = cls._hz_to_erb_rate(torch.tensor(float(f_min)))
        erb_max = cls._hz_to_erb_rate(torch.tensor(float(f_max)))
        centers = cls._erb_rate_to_hz(torch.linspace(erb_min, erb_max, num_bands))
        t = torch.arange(filter_length, dtype=torch.float32) / float(sample_rate)
        real_filters = []
        imag_filters = []
        for center in centers:
            erb_bw = 24.7 * (4.37e-3 * center + 1.0)
            bandwidth = 1.019 * erb_bw
            envelope = torch.pow(t.clamp_min(1.0e-6), order - 1)
            envelope = envelope * torch.exp(-2.0 * math.pi * bandwidth * t)
            phase = 2.0 * math.pi * center * t
            real = envelope * torch.cos(phase)
            imag = -envelope * torch.sin(phase)
            real = real - real.mean()
            imag = imag - imag.mean()
            real = real / real.norm().clamp_min(1.0e-8)
            imag = imag / imag.norm().clamp_min(1.0e-8)
            real_filters.append(real.flip(0))
            imag_filters.append(imag.flip(0))
        real = torch.stack(real_filters, dim=0).unsqueeze(1)
        imag = torch.stack(imag_filters, dim=0).unsqueeze(1)
        return real, imag

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        if speech.dim() == 3:
            speech = speech[..., 0]
        x = speech.unsqueeze(1).float()
        pad = self.filter_length // 2
        real = F.conv1d(
            x, self.real_filters.to(dtype=x.dtype), stride=self.hop_length, padding=pad
        )
        imag = F.conv1d(
            x, self.imag_filters.to(dtype=x.dtype), stride=self.hop_length, padding=pad
        )
        mag = torch.log1p(torch.sqrt(real.pow(2) + imag.pow(2) + 1.0e-8))
        phase = torch.atan2(imag, real)
        dphase = F.pad(phase[:, 1:] - phase[:, :-1], (0, 0, 1, 0))
        return torch.cat([mag, phase / math.pi, dphase / math.pi], dim=1)


class DeepAudioEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        ch = in_channels
        for idx in range(4):
            stride = 2 if idx == 0 else 1
            layers.extend(
                [
                    nn.Conv1d(ch, out_channels, kernel_size, stride=stride, padding=pad),
                    nn.BatchNorm1d(out_channels),
                    nn.PReLU(out_channels),
                ]
            )
            ch = out_channels
        self.net = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class DeepAudioEncoder(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 128)):
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(DeepAudioEncoderBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(in_channels, channels[-1], 1)
        self.output_dim = int(channels[-1])

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        if speech.dim() == 3:
            speech = speech[..., 0]
        x = speech.unsqueeze(1).float()
        return self.proj(self.blocks(x))


class BiMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fwd = _build_mamba_block(dim, d_state, d_conv)
        self.bwd = _build_mamba_block(dim, d_state, d_conv)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y_f = self.fwd(y)
        y_b = torch.flip(self.bwd(torch.flip(y, dims=[1])), dims=[1])
        x = x + self.drop(0.5 * (y_f + y_b))
        return x + self.drop(self.ffn(x))


class MASSBRPE(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        num_bands: int = 20,
        f_min: float = 50.0,
        f_max: float = 2000.0,
        filter_length: int = 401,
        hop_length: int = 160,
        patch_size: int = 16,
        max_patches: int = 1024,
        feature_dim: int = 128,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        room_feature_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gammatone = GammatoneFeatureExtractor(
            sample_rate=sample_rate,
            num_bands=num_bands,
            f_min=f_min,
            f_max=f_max,
            filter_length=filter_length,
            hop_length=hop_length,
        )
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Linear(self.patch_size * self.patch_size, feature_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, int(max_patches), feature_dim))
        self.blocks = nn.ModuleList(
            [BiMambaBlock(feature_dim, d_state, d_conv, dropout) for _ in range(num_layers)]
        )
        self.param_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        self.volume_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, room_feature_dim),
            nn.GELU(),
        )
        self.rt60_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, room_feature_dim),
            nn.GELU(),
        )

    def forward(self, speech: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.gammatone(speech)
        x = self._patchify(feat)
        x = self.patch_embed(x)
        pos = self._positional_embedding(x.shape[1], x.device, x.dtype)
        x = x + pos
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.param_head(pooled), self.volume_head(pooled), self.rt60_head(pooled)

    def _patchify(self, feat: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        pad_freq = (-feat.shape[1]) % patch
        pad_time = (-feat.shape[2]) % patch
        feat = F.pad(feat, (0, pad_time, 0, pad_freq))
        bsz, nfreq, ntime = feat.shape
        feat = feat.view(bsz, nfreq // patch, patch, ntime // patch, patch)
        feat = feat.permute(0, 1, 3, 2, 4).contiguous()
        return feat.view(bsz, -1, patch * patch)

    def _positional_embedding(
        self, num_patches: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = self.pos_embedding
        if num_patches > pos.shape[1]:
            pos = F.interpolate(
                pos.transpose(1, 2),
                size=num_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = pos[:, :num_patches]
        return pos.to(device=device, dtype=dtype)


class HybridPathCrossAttention(nn.Module):
    def __init__(
        self,
        audio_dim: int = 128,
        room_feature_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.room_to_audio = nn.Linear(room_feature_dim * 2, audio_dim)
        self.prefuse = nn.Sequential(nn.Linear(audio_dim * 2, audio_dim), nn.GELU())
        self.attn = nn.MultiheadAttention(
            embed_dim=audio_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(audio_dim)
        self.ffn_audio = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, audio_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(audio_dim * 4, audio_dim),
        )
        self.ffn_room = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, audio_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(audio_dim * 4, audio_dim),
        )
        self.out = nn.Sequential(nn.Linear(audio_dim * 2, audio_dim), nn.GELU())

    def forward(
        self, audio_feat: torch.Tensor, volume_feat: torch.Tensor, rt60_feat: torch.Tensor
    ) -> torch.Tensor:
        audio = audio_feat.transpose(1, 2)
        room = self.room_to_audio(torch.cat([volume_feat, rt60_feat], dim=-1))
        room = room.unsqueeze(1).expand(-1, audio.shape[1], -1)
        prefused = self.prefuse(torch.cat([room, audio], dim=-1))
        attn_out, _ = self.attn(query=prefused, key=audio, value=audio, need_weights=False)
        audio_enh = self.norm1(audio + attn_out)
        audio_enh = audio_enh + self.ffn_audio(audio_enh)
        room_enh = room + self.ffn_room(room)
        return self.out(torch.cat([audio_enh, room_enh], dim=-1)).transpose(1, 2)


class DATDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        rir_length: int = 8192,
        init_length: int = 128,
        num_late_filters: int = 8,
        fir_length: int = 129,
        bp_min: int = 1,
        bp_max: int = 8191,
    ):
        super().__init__()
        self.rir_length = int(rir_length)
        self.init_length = int(init_length)
        self.num_late_filters = int(num_late_filters)
        self.bp_min = int(bp_min)
        self.bp_max = int(bp_max)
        self.latent = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim * self.init_length),
        )
        up_layers = []
        channels = input_dim
        out_channels = input_dim
        length = self.init_length
        while length < self.rir_length:
            up_layers.extend(
                [
                    nn.ConvTranspose1d(
                        channels, out_channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.PReLU(out_channels),
                ]
            )
            channels = out_channels
            length *= 2
        self.upsample = nn.Sequential(*up_layers)
        self.head = nn.Conv1d(channels, 1 + self.num_late_filters, 1)
        self.fir = nn.Conv1d(
            self.num_late_filters,
            self.num_late_filters,
            fir_length,
            padding=fir_length // 2,
            groups=self.num_late_filters,
            bias=False,
        )
        noise = torch.randn(1, self.num_late_filters, self.rir_length)
        self.register_buffer("noise", noise)
        self.mix = nn.Conv1d(1 + self.num_late_filters, 1, 1)

    def forward(self, fused: torch.Tensor, log_params: torch.Tensor) -> torch.Tensor:
        pooled = fused.mean(dim=-1)
        x = self.latent(pooled).view(pooled.shape[0], fused.shape[1], self.init_length)
        x = self.upsample(x)
        x = x[..., : self.rir_length]
        heads = self.head(x)
        early = heads[:, :1]
        masks = heads[:, 1:]

        bp = torch.exp(log_params[:, 2] * math.log(10.0)).clamp(
            self.bp_min, self.bp_max
        )
        pos = torch.arange(self.rir_length, device=fused.device).view(1, 1, -1)
        early_mask = (pos < bp.view(-1, 1, 1)).to(dtype=early.dtype)
        late_mask = 1.0 - early_mask
        early = early * early_mask

        noise = self.noise.to(device=fused.device, dtype=fused.dtype).expand(
            fused.shape[0], -1, -1
        )
        late_noise = self.fir(noise)[..., : self.rir_length]
        late = late_noise * torch.sigmoid(masks) * late_mask
        return self.mix(torch.cat([early, late], dim=1)).squeeze(1)


class DARAS(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        rir_length: int = 8192,
        audio_channels: Sequence[int] = (64, 128, 256, 512),
        feature_dim: int = 128,
        room_feature_dim: int = 64,
        num_heads: int = 8,
        mamba_layers: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        num_gammatone_bands: int = 20,
        gammatone_f_min: float = 50.0,
        gammatone_f_max: float = 2000.0,
        gammatone_filter_length: int = 401,
        gammatone_hop_length: int = 160,
        gammatone_patch_size: int = 16,
        gammatone_max_patches: int = 1024,
        dat_init_length: int = 128,
        dat_num_late_filters: int = 8,
        dat_fir_length: int = 129,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.audio_encoder = DeepAudioEncoder(audio_channels)
        if self.audio_encoder.output_dim != feature_dim:
            self.audio_proj = nn.Conv1d(self.audio_encoder.output_dim, feature_dim, 1)
        else:
            self.audio_proj = nn.Identity()
        self.mass_brpe = MASSBRPE(
            sample_rate=sample_rate,
            num_bands=num_gammatone_bands,
            f_min=gammatone_f_min,
            f_max=gammatone_f_max,
            filter_length=gammatone_filter_length,
            hop_length=gammatone_hop_length,
            patch_size=gammatone_patch_size,
            max_patches=gammatone_max_patches,
            feature_dim=feature_dim,
            num_layers=mamba_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            room_feature_dim=room_feature_dim,
            dropout=dropout,
        )
        self.fusion = HybridPathCrossAttention(
            audio_dim=feature_dim,
            room_feature_dim=room_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = DATDecoder(
            input_dim=feature_dim,
            rir_length=rir_length,
            init_length=dat_init_length,
            num_late_filters=dat_num_late_filters,
            fir_length=dat_fir_length,
            bp_max=rir_length - 1,
        )

    def forward(self, speech: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        audio = self.audio_proj(self.audio_encoder(speech))
        log_params, volume_feat, rt60_feat = self.mass_brpe(speech)
        fused = self.fusion(audio, volume_feat, rt60_feat)
        rir = self.decoder(fused, log_params)
        return rir, {"log_params": log_params}
