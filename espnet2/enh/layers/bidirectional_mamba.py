"""Offline BiMamba using two independent official Mamba-1 modules.

CUDA uses mamba_ssm's kernels. The explicit torch backend uses the official
selective_scan_ref for CPU correctness checks, not efficient training.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BidirectionalMamba(nn.Module):
    """Map [batch, time, dim] to the same shape, without cached state."""

    def __init__(
        self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0, backend="auto"
    ):
        super().__init__()
        if min(dim, d_state, d_conv, expand) <= 0:
            raise ValueError("Mamba dimensions must be positive")
        if backend not in ("auto", "cuda", "torch"):
            raise ValueError("Mamba backend must be auto, cuda or torch")
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError(
                "TF-Locoformer-BiMamba requires mamba_ssm and its matching "
                "CUDA extensions in the runtime environment."
            ) from error
        self.backend = backend
        self.forward_mamba = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.backward_mamba = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.fusion = nn.Linear(2 * dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # ESPnet Xavier initialization ignores Mamba's _no_reinit markers.
        # Preserve the official discretization and state initialization.
        for direction in ("forward", "backward"):
            module = getattr(self, direction + "_mamba")
            for name, parameter in (
                ("dt_weight", module.dt_proj.weight),
                ("dt_bias", module.dt_proj.bias),
                ("A_log", module.A_log),
                ("D", module.D),
            ):
                self.register_buffer(
                    "_initial_" + direction + "_" + name,
                    parameter.detach().clone(),
                    persistent=False,
                )

    def espnet_initialization_fn(self):
        with torch.no_grad():
            for direction in ("forward", "backward"):
                module = getattr(self, direction + "_mamba")
                for name, parameter in (
                    ("dt_weight", module.dt_proj.weight),
                    ("dt_bias", module.dt_proj.bias),
                    ("A_log", module.A_log),
                    ("D", module.D),
                ):
                    parameter.copy_(
                        getattr(self, "_initial_" + direction + "_" + name)
                    )

    @staticmethod
    def _torch_forward(module, sequence):
        """Official Mamba projections/convolution with its reference scan."""
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

        x, z = module.in_proj(sequence).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(module.conv1d(x)[..., : sequence.shape[1]])
        dt, B, C = module.x_proj(x.transpose(1, 2)).split(
            (module.dt_rank, module.d_state, module.d_state), dim=-1
        )
        dt = F.linear(dt, module.dt_proj.weight).transpose(1, 2)
        y = selective_scan_ref(
            x,
            dt,
            -module.A_log.float().exp(),
            B.transpose(1, 2),
            C.transpose(1, 2),
            module.D.float(),
            z=z,
            delta_bias=module.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return module.out_proj(y.transpose(1, 2))

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1] == 0:
            raise ValueError("BiMamba expects nonempty [batch, time, dim]")
        backend = self.backend
        if backend == "auto":
            backend = "cuda" if sequence.is_cuda else "torch"
        if backend == "cuda" and not sequence.is_cuda:
            raise ValueError("Mamba CUDA backend requires CUDA input")
        run = self._torch_forward if backend == "torch" else lambda m, x: m(x)
        forward = run(self.forward_mamba, sequence.contiguous())
        backward = run(
            self.backward_mamba, sequence.flip(1).contiguous()
        ).flip(1)
        return self.dropout(self.fusion(torch.cat((forward, backward), dim=-1)))
