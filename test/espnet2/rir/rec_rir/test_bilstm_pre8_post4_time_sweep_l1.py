"""Full BiLSTM pre8/post4 topology with the shared waveform Sweep objective."""
from unittest.mock import patch

import torch

from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)


def test_pre8_post4_combined_loss_gradients_and_paired_pit():
    with patch('espnet2.rir.rec_rir.pooled_bimamba._build_mamba',
               side_effect=AssertionError('BiLSTM model must not construct Mamba')):
        model = ESPnetPooledBiMambaSweepV2PITModel(
            sr=8000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
            ctf_taps=5, pim_sweep_duration=0.1, time_sweep_loss_weight=1.0,
            predictor_conf=dict(time_module='bilstm', pre_frequency_module='light',
                                post_frequency_module='bilstm', pre_layers=8,
                                post_layers=4, emb_dim=8, freq_bottleneck=4,
                                freq_rank=4, pooling_hidden=4, n_heads=2,
                                ffn_dim=8, head_dim=8, local_kernel=3))
    assert len(model.ctf_predictor.time_blocks) == 8
    assert len(model.ctf_predictor.post_blocks) == 4
    batch = dict(speech_mix=torch.randn(2, 256),
                 speech_mix_lengths=torch.tensor([256, 256]))
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = float(speaker)
        reverb = direct.clone()
        reverb[:, 145 + speaker] = 0.3 * speaker
        batch[f'rir_direct{speaker}'] = direct
        batch[f'rir_ref{speaker}'] = reverb
    loss, stats, _ = model(**batch)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss.detach(),
                               stats['loss_sweep'] + stats['loss_time_sweep_l1'])
    assert stats['loss_sweep_min'] <= stats['loss_sweep'] + 1e-6
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    swapped = dict(batch)
    for prefix in ('rir_direct', 'rir_ref'):
        swapped[prefix + '1'], swapped[prefix + '2'] = batch[prefix + '2'], batch[prefix + '1']
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch)[0], model(**swapped)[0])
        rirs = model.estimate_rir(batch['speech_mix'][0], rir_length=256)
    assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()
