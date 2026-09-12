import torch

from espnet2.enh.separator.tflocoformer_separator_nocashe_rir_cmha_film_all import (
    RIRCMHAFiLMBlock,
)
from espnet2.torch_utils.initialize import initialize


def test_film_identity_survives_espnet_initialization():
    block = RIRCMHAFiLMBlock(channels=8, n_heads=2)

    initialize(block, "xavier_uniform")

    torch.testing.assert_close(
        block.gamma_proj.weight,
        torch.zeros_like(block.gamma_proj.weight),
    )
    torch.testing.assert_close(
        block.gamma_proj.bias,
        torch.ones_like(block.gamma_proj.bias),
    )
    torch.testing.assert_close(
        block.beta_proj.weight,
        torch.zeros_like(block.beta_proj.weight),
    )
    torch.testing.assert_close(
        block.beta_proj.bias,
        torch.zeros_like(block.beta_proj.bias),
    )

    speech = torch.randn(2, 8, 3, 5)
    rir = torch.randn(2, 8, 4, 5)
    output = block(speech, rir)
    torch.testing.assert_close(output, speech)

    output.square().mean().backward()
    for projection in (block.gamma_proj, block.beta_proj):
        for parameter in projection.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
