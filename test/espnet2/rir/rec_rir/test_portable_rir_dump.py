"""Byte-preserving relocation, sampler hashes and corruption detection."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    'portable', ROOT / 'egs2/whamr/rir_2spk/local/portable_rir_dump.py')
portable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable)


def test_nf_roundtrip_and_corruption():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        source = root / 'whamr/rir_2spk/dump'
        audio = root / 'whamr/enh_rir/data_rir_plus/wav16k/min/tr'
        audio.mkdir(parents=True)
        wave = audio / 'mix.wav'
        rir = audio / 'rir.wav'
        sf.write(wave, np.ones((64, 2)) * .1, 16000, subtype='FLOAT')
        sf.write(rir, np.eye(8, 2), 16000, subtype='FLOAT')
        folder = source / 'raw/tr_rir_2spk_nf_min_16k'
        folder.mkdir(parents=True)
        for key in ('wav', 'speech_mix', 'speech_direct1', 'speech_direct2',
                    'speech_reverb1', 'speech_reverb2', 'rir_direct1',
                    'rir_direct2', 'rir_ref1', 'rir_ref2'):
            path = rir if key.startswith('rir') else wave
            (folder / f'{key}.scp').write_text(f'example {path}\n')
        (folder / 'utt2num_samples').write_text('example 64\n')
        prep = dict(spec=dict(num_spk=2, sample_rate='16k', length='min', noise=False),
                    counts=dict(tr=1), output_hashes={})
        (source / 'preparation.json').write_text(json.dumps(prep))
        staging, destination = root / 'stage', root / 'published'
        portable.prepare(source, staging, destination, 'nf', 2)
        portable.verify(staging, 2)
        staging.rename(destination)
        manifest = portable.verify_metadata(destination)
        assert manifest['audio_files'] == 2
        # Stage 1 must reuse a published dump even without data_rir_plus.
        subprocess.run([sys.executable, str(ROOT / 'egs2/whamr/rir_2spk/local/prepare_two_speaker_dump.py'),
                        '--source', str(root / 'missing_source'), '--output', str(destination)], check=True)
        saved = json.loads((destination / 'preparation.json').read_text())
        for filename, expected in saved['output_hashes'].items():
            assert portable.digest(destination / filename) == expected
        scp = destination / 'raw/tr_rir_2spk_nf_min_16k/wav.scp'
        relocated = Path(scp.read_text().split()[1])
        assert relocated.read_bytes() == wave.read_bytes()
        assert sf.info(relocated).channels == 2
        original = scp.read_text()
        scp.write_text(original.replace('example', 'wrong'))
        with unittest.TestCase().assertRaisesRegex(ValueError, 'Metadata checksum'):
            portable.verify_metadata(destination)
        scp.write_text(original)
        with relocated.open('ab') as stream:
            stream.write(b'corruption')
        with unittest.TestCase().assertRaisesRegex(ValueError, 'Audio checksum'):
            portable.verify(destination, 2)


def test_but_relocates_metadata_and_preserves_original_rir():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        source = root / 'but'
        folder = source / 'raw/tt_but_2spk_clean_reverb_min_16k'
        folder.mkdir(parents=True)
        original = source / 'original_rir/rir.wav'
        original.parent.mkdir()
        sf.write(original, np.arange(8) * .1, 16000, subtype='FLOAT')
        wave = source / 'audio/16000/mix.wav'
        wave.parent.mkdir(parents=True)
        sf.write(wave, np.arange(64) * .001, 16000, subtype='FLOAT')
        for key, path in [('wav', wave), ('rir_ref1', original), ('rir_ref2', original)]:
            (folder / f'{key}.scp').write_text(f'example {path}\n')
        (folder / 'utt2num_samples').write_text('example 64\n')
        (source / 'generation_complete.json').write_text('{}')
        (source / 'rir_metadata.json').write_text(json.dumps([dict(path=str(original))]))
        destination, staging = root / 'published', root / 'stage'
        portable.prepare(source, staging, destination, 'but', 2)
        portable.verify(staging, 2)
        staging.rename(destination)
        portable.verify_metadata(destination)
        assert (destination / 'original_rir/rir.wav').read_bytes() == original.read_bytes()
        meta = json.loads((destination / 'rir_metadata.json').read_text())
        assert meta[0]['path'] == str(destination / 'original_rir/rir.wav')
        original_meta = json.loads((destination / 'provenance/midgar/rir_metadata.json').read_text())
        assert original_meta[0]['path'] == str(original)


if __name__ == '__main__':
    tests = [unittest.FunctionTestCase(v) for k, v in list(globals().items()) if k.startswith('test_')]
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests)).wasSuccessful())
