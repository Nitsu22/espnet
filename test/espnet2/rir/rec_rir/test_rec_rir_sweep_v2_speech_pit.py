"""Joint-PIT, synchronized targets and loss-ablation regressions on CPU.

Only Mamba kernels are replaced here; the lab check uses actual CUDA Mamba.
"""
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from espnet2.rir.rec_rir.rec_rir_sweep_v2_speech_pit import ESPnetRecRIRSweepV2SpeechPITModel
from espnet2.tasks.rir import RIRTask
from espnet2.train.preprocessor_rec_rir_sweep_v2_pit import RecRIRSweepV2PITPreprocessor
from espnet2.train.preprocessor_rec_rir_sweep_v2_speech_pit import RecRIRSweepV2SpeechPITPreprocessor
from test.espnet2.rir.rec_rir.test_rec_rir_sweep_v2_pit import COMMON, SMALL, cpu_kernel, make_model


def joint_model(**kwargs):
    with patch('espnet2.rir.rec_rir.model._build_mamba', cpu_kernel):
        return ESPnetRecRIRSweepV2SpeechPITModel(**COMMON, network_conf=SMALL, **kwargs)


def test_one_joint_permutation_per_utterance_and_weighting():
    model = joint_model()
    sweep = torch.tensor([[[0., 10.], [10., 0.]]] * 2, requires_grad=True)
    speech = torch.tensor([[[3., 0.], [0., 3.]], [[20., 0.], [0., 20.]]], requires_grad=True)
    loss, best, stats = model.select_joint_costs(sweep, speech)
    torch.testing.assert_close(best, torch.tensor([0, 1]))
    torch.testing.assert_close(loss, torch.tensor(6.5))
    torch.testing.assert_close(stats['loss_sweep'], torch.tensor(5.))
    torch.testing.assert_close(stats['loss_speech_rec'], torch.tensor(1.5))
    loss.backward()
    for gradient in (sweep.grad, speech.grad):
        torch.testing.assert_close(gradient, torch.tensor([
            [[.25, 0.], [0., .25]], [[0., .25], [.25, 0.]]]))
    model.speech_loss_weight = 4.0
    loss, best, stats = model.select_joint_costs(sweep, speech)
    assert torch.equal(best, torch.tensor([1, 1]))
    torch.testing.assert_close(loss, torch.tensor(10.))
    # Independently minimizing the two losses would incorrectly return zero.
    assert loss > 0


def test_zero_weight_ablation_matches_existing_sweep_and_speech_losses_and_gradients():
    torch.manual_seed(3)
    model = joint_model()
    direct = torch.zeros(2, 2, 256)
    direct[:, 0, 20] = 1; direct[:, 1, 30] = -2
    reverb = direct.clone()
    reverb[:, 0, 220] = .4; reverb[:, 1, 150] = -.7
    direct_spec = torch.randn(2, 2, 33, 9, dtype=torch.complex64)
    reverb_spec = torch.randn_like(direct_spec)
    for only_sweep in (True, False):
        model.sweep_loss_weight = float(only_sweep)
        model.speech_loss_weight = float(not only_sweep)
        ctf = torch.randn(2, 2, 33, 5, dtype=torch.complex64, requires_grad=True)
        actual, best, _ = model.paired_joint_loss(ctf, direct, reverb, direct_spec, reverb_spec)
        expected, perm = (model.paired_pit_loss(ctf, direct, reverb) if only_sweep
                          else model._pit_rec_loss(ctf, direct_spec, reverb_spec))
        torch.testing.assert_close(actual, expected)
        assert torch.equal(best, perm)
        grad = torch.autograd.grad(actual, ctf, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, ctf)[0]
        torch.testing.assert_close(grad, expected_grad)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_synchronized_crop_and_unchanged_full_physical_rir_pairs():
    mixture = np.arange(512, dtype=np.float64) + 1
    data = {'speech_mix': np.stack((mixture, -mixture), -1)}
    for speaker in (1, 2):
        data[f'speech_direct{speaker}'] = data['speech_mix'] * speaker
        data[f'speech_reverb{speaker}'] = data['speech_mix'] * (speaker + 2)
        direct = np.zeros((1024, 2)); direct[20 + speaker, 0] = -speaker
        reverb = direct.copy(); reverb[700 + speaker, 0] = .5 * speaker
        data[f'rir_direct{speaker}'], data[f'rir_ref{speaker}'] = direct, reverb
    proc = RecRIRSweepV2SpeechPITPreprocessor(True, speech_segment=256, rir_length=1024)
    old = RecRIRSweepV2PITPreprocessor(True, speech_segment=256, rir_length=1024)
    np.random.seed(12); prepared = proc('test', data)
    np.random.seed(12); existing = old('test', data)
    np.testing.assert_array_equal(prepared['speech_mix'], existing['speech_mix'])
    for speaker in (1, 2):
        np.testing.assert_array_equal(prepared[f'speech_direct{speaker}'], prepared['speech_mix'] * speaker)
        np.testing.assert_array_equal(prepared[f'speech_reverb{speaker}'], prepared['speech_mix'] * (speaker + 2))
        for prefix in ('rir_direct', 'rir_ref'):
            np.testing.assert_array_equal(prepared[f'{prefix}{speaker}'], existing[f'{prefix}{speaker}'])
        assert prepared[f'rir_ref{speaker}'][700 + speaker] == .5
        assert prepared[f'rir_direct{speaker}'][20 + speaker] == -1
    validation = RecRIRSweepV2SpeechPITPreprocessor(False, speech_segment=256, rir_length=1024)
    assert len(validation('test', data)['speech_mix']) == 512


def test_train_joint_pit_and_unchanged_network_checkpoint_and_pim():
    torch.manual_seed(0); model = joint_model()
    original = make_model()
    original.load_state_dict(model.state_dict(), strict=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters())
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 192]))
    batch['speech_mix'][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256); direct[:, 20 + speaker] = speaker
        reverb = direct.clone(); reverb[:, 160 + speaker] = speaker * .2
        batch[f'rir_direct{speaker}'], batch[f'rir_ref{speaker}'] = direct, reverb
        for prefix in ('speech_direct', 'speech_reverb'):
            wave = torch.randn(2, 256); wave[1, 192:] = 0
            batch[f'{prefix}{speaker}'] = wave
    loss, stats, _ = model(**batch)
    torch.testing.assert_close(loss.detach(), stats['loss_sweep'] + stats['loss_speech_rec'])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    model.eval(); original.eval()
    swapped = {key: value.clone() for key, value in batch.items()}
    for prefix in ('rir_direct', 'rir_ref', 'speech_direct', 'speech_reverb'):
        swapped[prefix + '1'][0] = batch[prefix + '2'][0]
        swapped[prefix + '2'][0] = batch[prefix + '1'][0]
    with torch.no_grad():
        torch.testing.assert_close(model(**swapped)[0], model(**batch)[0])
        torch.testing.assert_close(model.estimate_ctf(batch['speech_mix'][0]), original.estimate_ctf(batch['speech_mix'][0]))
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        clone = joint_model().eval(); clone.load_state_dict(torch.load(buffer), strict=True)
        rir = clone.estimate_rir(batch['speech_mix'][0], rir_length=256)
        assert rir.shape == (2, 256) and torch.isfinite(rir).all()
        torch.testing.assert_close(rir, original.estimate_rir(batch['speech_mix'][0], rir_length=256))


def test_weights_and_task_configuration():
    for a, b in [(-1, 1), (float('nan'), 1), (1, float('inf')), (0, 0)]:
        try:
            joint_model(sweep_loss_weight=a, speech_loss_weight=b)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid weights accepted')
    root = Path(__file__).resolve().parents[4]
    recipe = root / 'egs2/whamr/rir_2spk'
    old = yaml.safe_load((recipe / 'conf/tuning/train_rec_rir_2spk_nf_8k_sweep_v2.yaml').read_text())
    path = recipe / 'conf/tuning/train_rec_rir_2spk_nf_8k_sweep_v2_speech_rec.yaml'
    new = yaml.safe_load(path.read_text())
    old['rir_model_type'] = old['preprocessor'] = 'rec_rir_sweep_v2_speech_pit'
    old['model_conf'].update(sweep_loss_weight=1., speech_loss_weight=1.)
    assert old == new
    args = RIRTask.get_parser().parse_args(['--config', str(path), '--output_dir', 'unused'])
    args.model_conf['network_conf'] = SMALL
    with patch('espnet2.rir.rec_rir.model._build_mamba', cpu_kernel):
        model = RIRTask.build_model(args)
    assert isinstance(model, ESPnetRecRIRSweepV2SpeechPITModel)
    assert isinstance(RIRTask.build_preprocess_fn(args, True), RecRIRSweepV2SpeechPITPreprocessor)


if __name__ == '__main__':
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for key, value in list(globals().items()) if key.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
