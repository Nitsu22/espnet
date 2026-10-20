import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / 'egs2/whamr/rir_2spk'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, RECIPE / 'local' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'generation_complete.json').write_text('{}')
    names = {'s1_anechoic': .1, 's2_anechoic': .2, 's1_reverb': .1,
             's2_reverb': .2, 'noise': .05, 'mix_clean_reverb': .3,
             'mix_both_reverb': .35, 'rir1_anechoic': .1, 'rir2_anechoic': .1,
             'rir1_reverb': .2, 'rir2_reverb': .2}
    for split in ('tr', 'cv', 'tt'):
        manifest = root / 'manifests' / f'{split}_min_8k'
        manifest.mkdir(parents=True)
        metadata = root / 'wav8k/min' / split
        metadata.mkdir(parents=True)
        rows = []
        for index in range(2):
            uid = f'{split}_{index}'
            rows.append({'utterance_id': uid, 'source1': '/a/one/a.wav',
                         'source2': '/a/two/b.wav'})
        (metadata / 'utterances.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows))
        for name, amplitude in names.items():
            entries = []
            for row in rows:
                path = metadata / f'{row["utterance_id"]}_{name}.wav'
                sf.write(path, np.full((800, 2), amplitude), 8000, subtype='FLOAT')
                entries.append(f'{row["utterance_id"]} {path}\n')
            (manifest / f'{name}.scp').write_text(''.join(entries))
    return root


def prepare(monkeypatch, source, output, noise):
    args = ['prepare', '--source', str(source), '--output', str(output),
            '--sample-rate', '8k'] + (['--noise'] if noise else [])
    monkeypatch.setattr(sys, 'argv', args)
    load_script('prepare_two_speaker_dump').main()


def test_pairing_no_audio_copy_and_rerun(source, tmp_path, monkeypatch):
    clean, noisy = tmp_path / 'clean', tmp_path / 'noisy'
    prepare(monkeypatch, source, clean, False)
    prepare(monkeypatch, source, noisy, True)
    before = {str(p.relative_to(noisy)): p.read_bytes()
              for p in noisy.rglob('*') if p.is_file()}
    prepare(monkeypatch, source, noisy, True)
    assert before == {str(p.relative_to(noisy)): p.read_bytes()
                      for p in noisy.rglob('*') if p.is_file()}
    assert not list(noisy.rglob('*.wav'))
    monkeypatch.syspath_prepend(str(RECIPE / 'local'))
    report = load_script('verify_noisy_two_speaker_dump').verify(noisy, clean)
    assert report['counts'] == {'tr': 2, 'cv': 2, 'tt': 2}
    # Same output cannot silently switch back to noise-free input.
    with pytest.raises(ValueError, match='different source/settings'):
        prepare(monkeypatch, source, noisy, False)


def test_noisy_shapes_and_tampering(source, tmp_path, monkeypatch):
    dump, shapes = tmp_path / 'noisy', tmp_path / 'shapes'
    prepare(monkeypatch, source, dump, True)
    monkeypatch.setattr(sys, 'argv', ['shapes', '--config', str(RECIPE / 'conf/tuning/'
        'train_pooled_bimamba_2spk_noisy_8k_sweep_v2.yaml'), '--dump', str(dump),
        '--output', str(shapes)])
    script = load_script('prepare_input_shapes')
    script.main()
    assert (shapes / 'train/speech_mix_shape').read_text() == 'tr_0 800\ntr_1 800\n'
    path = dump / 'raw/tr_rir_2spk_noisy_min_8k/wav.scp'
    path.write_text(path.read_text() + 'bad nonexistent.wav\n')
    with pytest.raises(AssertionError):
        script.main()


def test_mismatched_ids_rejected(source, tmp_path, monkeypatch):
    (source / 'manifests/tr_min_8k/noise.scp').write_text('')
    with pytest.raises(ValueError, match='Mismatched source IDs'):
        prepare(monkeypatch, source, tmp_path / 'bad', True)
    assert not (tmp_path / 'bad').exists()


def test_noise_length_rejected(source, tmp_path, monkeypatch):
    path = source / 'wav8k/min/tr/tr_0_noise.wav'
    sf.write(path, np.zeros((700, 2)), 8000, subtype='FLOAT')
    with pytest.raises(ValueError, match='noise and mixture lengths differ'):
        prepare(monkeypatch, source, tmp_path / 'bad', True)


def test_config_exactly_matches_native_8k_baseline():
    folder = RECIPE / 'conf/tuning'
    assert yaml.safe_load((folder / 'train_pooled_bimamba_2spk_nf_8k_sweep_v2.yaml').read_text()) == yaml.safe_load(
        (folder / 'train_pooled_bimamba_2spk_noisy_8k_sweep_v2.yaml').read_text())
