"""Noisy index selection, unchanged RIR teachers and batching metadata."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / 'egs2/whamr/rir_2spk'
PREPARE = RECIPE / 'local/prepare_two_speaker_dump.py'


def source_tree(root, short_noise=False):
    source = root / 'source'
    source.mkdir()
    (source / 'generation_complete.json').write_text('{}')
    arrays = {
        's1_reverb': np.full((64, 2), .1),
        's2_reverb': np.full((64, 2), .2),
        's1_anechoic': np.full((64, 2), .05),
        's2_anechoic': np.full((64, 2), .08),
        'noise': np.full((32 if short_noise else 64, 2), .01),
        'mix_clean_reverb': np.full((64, 2), .3),
        'mix_both_reverb': np.full((64, 2), .31),
    }
    for speaker in (1, 2):
        for kind in ('anechoic', 'reverb'):
            arrays[f'rir{speaker}_{kind}'] = np.eye(8, 2) * .1
    for split in ('tr', 'cv', 'tt'):
        manifest = source / 'manifests' / f'{split}_min_16k'
        audio = source / 'wav16k/min' / split
        manifest.mkdir(parents=True)
        audio.mkdir(parents=True)
        for name, array in arrays.items():
            path = audio / f'{name}.wav'
            sf.write(path, array, 16000, subtype='FLOAT')
            (manifest / f'{name}.scp').write_text(f'u {path}\n')
        (audio / 'utterances.jsonl').write_text(json.dumps(
            dict(utterance_id='u', source1='/speaker1/a.wav',
                 source2='/speaker2/b.wav')) + '\n')
    return source


def prepare(source, output, noisy=False, check=True):
    args = [sys.executable, str(PREPARE), '--source', str(source),
            '--output', str(output), '--workers', '1']
    if noisy:
        args.append('--noise')
    return subprocess.run(args, check=check, capture_output=True, text=True)


def test_noisy_and_nf_keep_same_rir_teachers_and_use_matching_shapes(tmp_path):
    source = source_tree(tmp_path)
    nf, noisy = tmp_path / 'nf', tmp_path / 'noisy'
    prepare(source, nf)
    prepare(source, noisy, noisy=True)
    for split in ('tr', 'cv', 'tt'):
        clean_dir = nf / 'raw' / f'{split}_rir_2spk_nf_min_16k'
        noisy_dir = noisy / 'raw' / f'{split}_rir_2spk_noisy_min_16k'
        assert 'mix_both_reverb.wav' in (noisy_dir / 'wav.scp').read_text()
        assert 'mix_clean_reverb.wav' in (clean_dir / 'wav.scp').read_text()
        assert (noisy_dir / 'noise_ref.scp').is_file()
        for name in ('rir_ref1', 'rir_ref2', 'rir_direct1', 'rir_direct2'):
            assert (clean_dir / f'{name}.scp').read_bytes() == (noisy_dir / f'{name}.scp').read_bytes()
    assert not list(noisy.rglob('*.wav'))
    prepare(source, noisy, noisy=True)
    assert prepare(source, noisy, check=False).returncode != 0
    stats = tmp_path / 'stats'
    config = RECIPE / 'conf/tuning/train_pooled_bimamba_2spk_noisy_16k_sweep_v2.yaml'
    subprocess.run([sys.executable, str(RECIPE / 'local/prepare_input_shapes.py'),
                    '--config', str(config), '--dump', str(noisy),
                    '--output', str(stats)], check=True)
    assert (stats / 'train/speech_mix_shape').read_text() == 'u 64\n'
    shape = noisy / 'raw/tr_rir_2spk_noisy_min_16k/utt2num_samples'
    shape.write_text('u 65\n')
    failed = subprocess.run([sys.executable, str(RECIPE / 'local/prepare_input_shapes.py'),
                             '--config', str(config), '--dump', str(noisy),
                             '--output', str(stats)], capture_output=True)
    assert failed.returncode != 0


def test_rejects_noisy_mixture_with_wrong_noise_length(tmp_path):
    source = source_tree(tmp_path, short_noise=True)
    output = tmp_path / 'bad'
    failed = prepare(source, output, noisy=True, check=False)
    assert failed.returncode != 0
    assert 'noise and mixture lengths differ' in failed.stderr
    assert not output.exists()
