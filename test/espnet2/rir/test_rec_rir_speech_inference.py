from types import SimpleNamespace

import pytest
import torch

from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.speech_inference import estimate_speech


@pytest.mark.parametrize("length", [8000, 8193])
@pytest.mark.parametrize("normalize", [False, True])
def test_heads_restore_gain_and_keep_final_partial_hop(length, normalize):
    torch.manual_seed(41)
    mixture = torch.randn(length) * 7

    def heads(features):
        # Independent known spectral gains identify both branches/speakers.
        clean = torch.stack((features * 0.25, features * 0.75), dim=1)
        reverb = torch.stack((features * -0.5, features * 1.5), dim=1)
        return clean, None, reverb

    model = SimpleNamespace(
        transforms=RecRIRTransforms(8000, 256, 128, "sqrthann", 256),
        normalize_by_mix=normalize,
        rec_rir=heads,
        _postprocess_multi=lambda x: torch.complex(x[:, :, 0], x[:, :, 1]),
    )
    result = estimate_speech(model, mixture)
    for branch, gains in (("clean", (0.25, 0.75)), ("reverb", (-0.5, 1.5))):
        expected = torch.stack([mixture * gain for gain in gains])
        assert result[branch].shape == (2, length)
        torch.testing.assert_close(result[branch], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            result[branch][:, -1], expected[:, -1], atol=1e-5, rtol=1e-5
        )
