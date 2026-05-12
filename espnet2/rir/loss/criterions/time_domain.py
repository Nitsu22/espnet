import torch

from espnet2.enh.loss.criterions.time_domain import (
    MultiResL1SpecLoss,
    TimeDomainL1,
    TimeDomainLoss,
)


class RIRCorrelationLoss(TimeDomainLoss):
    def __init__(self, eps: float = 1.0e-8, name=None, only_for_test=False):
        _name = "rir_corr_loss" if name is None else name
        super().__init__(_name, only_for_test=only_for_test)
        self.eps = eps

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        ref = ref.reshape(ref.size(0), -1)
        inf = inf.reshape(inf.size(0), -1)
        ref = ref - ref.mean(dim=1, keepdim=True)
        inf = inf - inf.mean(dim=1, keepdim=True)
        denom = torch.linalg.vector_norm(ref, dim=1) * torch.linalg.vector_norm(
            inf, dim=1
        )
        corr = (ref * inf).sum(dim=1) / denom.clamp_min(self.eps)
        return 1.0 - corr


class RIRMultiTaskLoss(TimeDomainLoss):
    def __init__(
        self,
        window_sz=(512,),
        hop_sz=None,
        time_domain_weight: float = 0.5,
        normalize_variance: bool = False,
        reduction: str = "sum",
        mrstft_weight: float = 1.0,
        l1_weight: float = 0.1,
        corr_weight: float = 0.1,
        corr_eps: float = 1.0e-8,
        name=None,
        only_for_test: bool = False,
    ):
        _name = "rir_multitask_loss" if name is None else name
        super().__init__(_name, only_for_test=only_for_test)
        self.mrstft_weight = float(mrstft_weight)
        self.l1_weight = float(l1_weight)
        self.corr_weight = float(corr_weight)
        self.mrstft = MultiResL1SpecLoss(
            window_sz=window_sz,
            hop_sz=hop_sz,
            time_domain_weight=time_domain_weight,
            normalize_variance=normalize_variance,
            reduction=reduction,
        )
        self.l1 = TimeDomainL1()
        self.corr = RIRCorrelationLoss(eps=corr_eps)

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        mrstft_loss = self.mrstft(ref, inf)
        l1_loss = self.l1(ref, inf)
        corr_loss = self.corr(ref, inf)
        loss = (
            self.mrstft_weight * mrstft_loss
            + self.l1_weight * l1_loss
            + self.corr_weight * corr_loss
        )
        self.stats = {
            "rir_mrstft_loss": mrstft_loss.detach(),
            "rir_l1_loss": l1_loss.detach(),
            "rir_corr_loss": corr_loss.detach(),
        }
        return loss
