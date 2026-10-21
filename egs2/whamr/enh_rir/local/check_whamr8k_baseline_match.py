"""Require the saved noisy Baseline's optimizer, architecture and shape series."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from check_bimamba_conditioning_data import read_scp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(config_path, baseline_path, stats, inputs):
    new, baseline = [yaml.safe_load(path.read_text()) for path in (config_path, baseline_path)]
    for key in new.keys() - {'separator', 'separator_conf', 'model_conf'}:
        if new[key] != baseline.get(key):
            raise ValueError(f'Baseline training setting differs: {key}')
    trunk = {key: value for key, value in new['separator_conf'].items()
             if key not in ('ctf_taps', 'random_ctf_swap')}
    if trunk != baseline['separator_conf'] or baseline['separator'] != 'tflocoformer_nocashe':
        raise ValueError('Baseline separator architecture differs')
    if baseline['model_conf'] != {'normalize_variance': True}:
        raise ValueError('Baseline waveform normalization differs')
    if baseline['ngpu'] != 1 or baseline['accum_grad'] != 1:
        raise ValueError('Use the existing one-GPU accum-grad-one Baseline')
    if baseline['fold_length'] != [200000] * 3:
        raise ValueError('Baseline fold thresholds differ')
    source = Path(new['model_conf']['bimamba_checkpoint'])
    if digest(source) != new['model_conf']['bimamba_sha256']:
        raise ValueError('Wrong pretrained noisy BiMamba checkpoint')
    prepared = json.loads((inputs / 'preparation.json').read_text())
    if prepared['spec'].get('condition') != 'whamr':
        raise ValueError('Use noisy WHAMR, not NF-WHAMR')
    for name, expected in prepared['output_hashes'].items():
        if digest(inputs / name) != expected:
            raise ValueError(f'Prepared data changed: {name}')
    report = dict(config_sha256=digest(config_path), baseline_config_sha256=digest(baseline_path),
                  preparation_sha256=digest(inputs / 'preparation.json'),
                  checkpoint_sha256=digest(source), counts=prepared['counts'], shapes={})
    for split, phase in (('tr', 'train'), ('cv', 'valid')):
        lengths = read_scp(inputs / 'raw' / f'{split}_mix_both_reverb_min_8k/utt2num_samples')
        for field in ('speech_mix', 'speech_ref1', 'speech_ref2'):
            path = stats / phase / f'{field}_shape'
            original = read_scp(path)
            original = {uid: str(int(shape.split(',')[0])) for uid, shape in original.items()}
            if lengths != original:
                raise ValueError(f'Baseline IDs/lengths differ: {phase}/{field}')
            report['shapes'][f'{phase}/{field}'] = digest(path)
    report['status'] = 'passed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--baseline-stats', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.config, args.baseline_config, args.baseline_stats, args.inputs)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
