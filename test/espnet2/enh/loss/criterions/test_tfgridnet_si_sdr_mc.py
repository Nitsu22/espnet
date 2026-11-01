from itertools import permutations

import numpy as np
import torch

from espnet2.enh.loss.criterions.tfgridnet_si_sdr_mc import (
    TFGridNetSISDRMCPIT,
    TFGridNetSISDRSE,
)
from espnet2.enh.separator.tfgridnet_separator import TFGridNet


def objective(ref, inf, lengths=None, scale=None):
    batch, _, samples = ref.shape
    return TFGridNetSISDRMCPIT(TFGridNetSISDRSE())(
        list(ref.unbind(1)),
        list(inf.unbind(1)),
        {
            "tfgridnet_lengths": lengths if lengths is not None else torch.full((batch,), samples),
            "tfgridnet_mix_std": scale if scale is not None else torch.ones(batch, 1),
        },
    )


def numpy_costs(ref, inf):
    # Independent least-squares oracle for equations (9)-(10).
    costs, se_costs = [], []
    for order in permutations(range(len(ref))):
        projected, se = [], 0.0
        for target, estimate in zip(ref, inf[list(order)]):
            alpha = np.linalg.lstsq(estimate[:, None], target, rcond=None)[0][0]
            projected.append(alpha * estimate)
            se -= 20 * np.log10(np.linalg.norm(target) / np.linalg.norm(projected[-1] - target))
        se_costs.append(se)
        costs.append(se + np.abs(np.sum(projected, axis=0) - ref.sum(0)).mean())
    return np.asarray(costs), np.asarray(se_costs)


def test_equation_and_joint_permutation():
    rng = np.random.RandomState(4)
    # Large normalized targets expose cases where MC changes the PIT assignment.
    for _ in range(1000):
        ref, inf = rng.randn(2, 13) * 30, rng.randn(2, 13)
        costs, se_costs = numpy_costs(ref, inf)
        if costs.argmin() != se_costs.argmin():
            break
    else:
        raise AssertionError("No joint-PIT counterexample found")
    loss, stats, other = objective(torch.tensor(ref)[None], torch.tensor(inf)[None])
    np.testing.assert_allclose(loss.item(), costs.min(), atol=1e-10)
    expected_order = list(list(permutations(range(2)))[costs.argmin()])
    assert other["perm"].tolist() == [expected_order]
    torch.testing.assert_close(loss, stats["si_sdr_se_loss"] + stats["mixture_constraint"])


def test_padding_batch_and_scaling():
    torch.manual_seed(3)
    ref, inf = torch.randn(2, 2, 41, dtype=torch.float64), torch.randn(2, 2, 41, dtype=torch.float64)
    lengths = torch.tensor([41, 23])
    loss = objective(ref, inf, lengths)[0]
    expected = (objective(ref[:1], inf[:1])[0] + objective(ref[1:, :, :23], inf[1:, :, :23])[0]) / 2
    torch.testing.assert_close(loss, expected)
    changed_ref, changed_inf = ref.clone(), inf.clone()
    changed_ref[1, :, 23:] = 1e6
    changed_inf[1, :, 23:] = -1e6
    torch.testing.assert_close(loss, objective(changed_ref, changed_inf, lengths)[0])
    torch.testing.assert_close(loss, objective(ref, inf[:, [1, 0]], lengths)[0])
    torch.testing.assert_close(loss, objective(ref, inf * torch.tensor([2.0, -3.0])[None, :, None], lengths)[0])
    torch.testing.assert_close(loss, objective(ref * 5, inf * 5, lengths, torch.full((2, 1), 5.0))[0])


def test_gradients_and_silence():
    torch.manual_seed(5)
    ref = torch.randn(1, 2, 17, dtype=torch.float64)
    inf = torch.randn(1, 2, 17, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: objective(ref, x)[0], (inf,))
    for target, estimate in [(torch.zeros_like(ref), inf), (ref, torch.zeros_like(ref)), (ref, ref)]:
        estimate = estimate.detach().requires_grad_()
        loss = objective(target, estimate)[0]
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(estimate.grad).all()


def test_length_aware_mixture_normalization():
    torch.manual_seed(7)
    model = TFGridNet(input_dim=0, n_layers=1, lstm_hidden_units=8, emb_dim=8,
                      n_fft=32, stride=8, window="sqrt_hann", emb_ks=2,
                      return_loss_metadata=True, length_aware_normalization=True)
    mixture = torch.randn(2, 128)
    mixture[1, 80:] = 0
    output, _, metadata = model(mixture, torch.tensor([128, 80]))
    expected = torch.stack([mixture[0].std(), mixture[1, :80].std()])[:, None]
    torch.testing.assert_close(metadata["tfgridnet_mix_std"], expected)
    assert all(torch.isfinite(x).all() for x in output)
