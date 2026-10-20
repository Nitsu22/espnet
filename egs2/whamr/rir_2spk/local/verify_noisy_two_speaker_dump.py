"""Audit noisy/NF index pairing and sampled acoustic mixture identities."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_two_speaker_dump import read_scp


def verify(noisy, clean):
    metadata = json.loads((noisy / 'preparation.json').read_text())
    clean_metadata = json.loads((clean / 'preparation.json').read_text())
    assert metadata['spec']['noise'] is True
    assert clean_metadata['spec']['noise'] is False
    assert metadata['counts'] == clean_metadata['counts']
    sample_rate = metadata['spec']['sample_rate']
    assert sample_rate == clean_metadata['spec']['sample_rate'] == '8k'
    # Check every recorded output, not just the sampled waveforms.
    for relative, expected in metadata['output_hashes'].items():
        assert hashlib.sha256((noisy / relative).read_bytes()).hexdigest() == expected
    report = {'counts': metadata['counts'], 'sample_rate': 8000, 'splits': {}}
    paired_fields = ['utt2spk', 'utt2num_samples'] + [
        f'{prefix}{speaker}.scp'
        for prefix in ('rir_ref', 'rir_direct', 'speech_direct', 'speech_reverb')
        for speaker in (1, 2)
    ]
    for split in ('tr', 'cv', 'tt'):
        nf = clean / 'raw' / f'{split}_rir_2spk_nf_min_8k'
        folder = noisy / 'raw' / f'{split}_rir_2spk_noisy_min_8k'
        for name in paired_fields:
            assert (folder / name).read_bytes() == (nf / name).read_bytes(), name
        maps = {name: read_scp(folder / f'{name}.scp') for name in
                ('wav', 'speech_reverb1', 'speech_reverb2', 'noise_ref')}
        clean_mix = read_scp(nf / 'wav.scp')
        ids = sorted(clean_mix)
        assert all(set(values) == set(ids) for values in maps.values())
        assert len(ids) == metadata['counts'][split]
        assert all(maps['wav'][uid] != clean_mix[uid] for uid in ids)
        sampled = []
        for index in sorted({0, len(ids) // 3, 2 * len(ids) // 3, len(ids) - 1}):
            uid = ids[index]
            waves = {}
            for name, values in maps.items():
                waves[name], rate = sf.read(values[uid])
                assert rate == 8000 and waves[name].ndim == 2
            expected = waves['speech_reverb1'] + waves['speech_reverb2']
            original, rate = sf.read(clean_mix[uid])
            assert rate == 8000
            np.testing.assert_allclose(expected, original, rtol=0, atol=2e-6)
            assert np.any(waves['noise_ref'][:, 0] != 0)
            np.testing.assert_allclose(expected + waves['noise_ref'], waves['wav'],
                                       rtol=0, atol=2e-6)
            sampled.append(uid)
        report['splits'][split] = {'paired_utterances': len(ids), 'sampled_uids': sampled}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noisy', type=Path, required=True)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.noisy, args.clean)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
