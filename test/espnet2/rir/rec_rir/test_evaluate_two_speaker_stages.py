"""CPU checks of split inference/scoring and complete-set guards."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    'two_speaker_evaluation', ROOT / 'egs2/whamr/rir_2spk/local/evaluate_two_speaker.py')
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)
REAL_RUN = subprocess.run


def fixture(root):
    experiment, data, output = root / 'experiment', root / 'data', root / 'output'
    experiment.mkdir()
    data.mkdir()
    (experiment / 'config.yaml').write_text('placeholder: true\n')
    (experiment / 'valid.loss.best.pth').write_bytes(b'checkpoint-fixture')
    waveform = np.zeros(32000, dtype=np.float32)
    waveform[80] = 1
    waveform[81:] = .01 * np.exp(-np.arange(31919) / 2000)
    audio = root / 'rir.wav'
    sf.write(audio, waveform, 16000, subtype='FLOAT')
    for name in ('wav', 'rir_ref1', 'rir_ref2'):
        (data / f'{name}.scp').write_text(f'u0 {audio}\nu1 {audio}\n')
    return experiment, data, output, audio


def invoke(experiment, data, output, *flags):
    arguments = ['evaluate', '--experiment', str(experiment), '--data', str(data),
                 '--output', str(output), *flags]
    with patch.object(sys, 'argv', arguments), \
            patch.object(EVALUATE.subprocess, 'check_output', return_value='test-commit\n'):
        EVALUATE.main()


def fake_inference(audio):
    def run(command, **kwargs):
        if '-m' not in command:
            return REAL_RUN(command, **kwargs)
        output = Path(command[command.index('--output_dir') + 1])
        output.mkdir()
        rows = []
        ids = Path(command[command.index('--wav_scp') + 1]).read_text().splitlines()
        for row in ids:
            uid = row.split()[0]
            copies = []
            for speaker in (1, 2):
                path = output / f'{uid}_{speaker}.wav'
                shutil.copy2(audio, path)
                copies.append(str(path))
            rows.append(f'{uid} {copies[0]} {copies[1]}\n')
        (output / 'rir.scp').write_text(''.join(rows))
        return subprocess.CompletedProcess(command, 0)
    return run


def test_split_scoring_uses_no_inference_and_preserves_metric_modes():
    with tempfile.TemporaryDirectory() as temporary:
        experiment, data, output, audio = fixture(Path(temporary))
        with patch.object(EVALUATE.subprocess, 'run', side_effect=fake_inference(audio)) as calls:
            invoke(experiment, data, output, '--stage', 'inference')
            assert calls.call_count == 1
        assert (output / 'inference_complete.json').exists()
        assert not (output / 'complete.json').exists()
        with patch.object(EVALUATE.subprocess, 'run', wraps=REAL_RUN) as calls:
            invoke(experiment, data, output, '--stage', 'score', '--also_polarity_aligned')
            assert calls.call_count == 2
            assert all('-m' not in call.args[0] for call in calls.call_args_list)
        for name, polarity in (('score', False), ('score_polarity_aligned', True)):
            report = json.loads((output / name / 'summary.json').read_text())
            assert report['num_utts'] == 2 and report['num_source_pairs'] == 4
            assert report['align'] == 'peak' and report['scale_mode'] == 'peak'
            assert report['pit_metric'] == 'rmse' and report['polarity_align'] == polarity
            assert report['rmse_mean'] == 0
        assert json.loads((output / 'complete.json').read_text())['count'] == 2


def test_scoring_rejects_missing_predictions_and_changed_checkpoint():
    with tempfile.TemporaryDirectory() as temporary:
        experiment, data, output, audio = fixture(Path(temporary))
        with patch.object(EVALUATE.subprocess, 'run', side_effect=fake_inference(audio)):
            invoke(experiment, data, output, '--stage', 'inference')
        predictions = output / 'inference/rir.scp'
        rows = predictions.read_text()
        predictions.write_text(rows.splitlines()[0] + '\n')
        with patch.object(EVALUATE.subprocess, 'run') as calls:
            with unittest.TestCase().assertRaisesRegex(ValueError, 'Incomplete predictions'):
                invoke(experiment, data, output, '--stage', 'score')
            calls.assert_not_called()
        predictions.write_text(rows)
        (output / 'valid.loss.best.pth').write_bytes(b'changed')
        with unittest.TestCase().assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            invoke(experiment, data, output, '--stage', 'score')


def test_auxiliary_checkpoint_selection_and_provenance():
    with tempfile.TemporaryDirectory() as temporary:
        experiment, data, output, audio = fixture(Path(temporary))
        name = 'valid.loss_sweep.best.pth'
        (experiment / name).write_bytes(b'sweep-best-fixture')
        with patch.object(EVALUATE.subprocess, 'run', side_effect=fake_inference(audio)):
            invoke(experiment, data, output, '--stage', 'inference', '--checkpoint', name)
        assert (output / 'valid.loss.best.pth').read_bytes() == b'sweep-best-fixture'
        assert json.loads((output / 'metadata.json').read_text())['checkpoint'] == name
        with unittest.TestCase().assertRaisesRegex(ValueError, 'checkpoint'):
            invoke(experiment, data, output, '--stage', 'score')
        with patch.object(EVALUATE.subprocess, 'run', wraps=REAL_RUN):
            invoke(experiment, data, output, '--stage', 'score', '--checkpoint', name)
        assert (output / 'complete.json').exists()


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for name, value in list(globals().items()) if name.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
