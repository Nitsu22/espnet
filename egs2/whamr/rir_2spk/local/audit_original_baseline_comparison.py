"""Verify exact original audio bytes and all non-head training settings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import yaml


BASELINE_CONFIG_SHA256 = 'd4e3461ca974c7c41ae643b4ba4aa9479007cea7902b77c9002884bbff34137f'
NAMES = ('speech_mix', 'speech_direct1', 'speech_direct2',
         'speech_reverb1', 'speech_reverb2')
RELOCATED = ('train_data_path_and_name_and_type', 'valid_data_path_and_name_and_type',
             'train_shape_file', 'valid_shape_file')
RUNTIME = {'config', 'print_config', 'required', 'version', 'distributed',
           'output_dir', 'dist_init_method', 'dist_world_size', 'dist_rank',
           'local_rank', 'dist_master_addr', 'dist_master_port', 'dist_launcher',
           'resume', 'rir_model_type', 'model_conf'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def config_path(variant):
    return Path(f'conf/tuning/train_rec_rir_original_baseline_{variant}.yaml')


def audit_configs(inputs):
    baseline_file = inputs / 'baseline_config.yaml'
    if digest(baseline_file) != BASELINE_CONFIG_SHA256:
        raise ValueError('Not the selected original trained baseline config')
    baseline = yaml.safe_load(baseline_file.read_text())
    manifest = json.loads((inputs / 'preparation.json').read_text())
    assert manifest['baseline_config_sha256'] == BASELINE_CONFIG_SHA256
    assert manifest['no_audio_conversion'] is True
    expected_model = dict(baseline['model_conf'])
    taps = expected_model.pop('dim_output_CTF') // (2 * expected_model['num_spk'])
    result = {}
    for variant in ('nb', 'full'):
        path = config_path(variant)
        proposal = yaml.safe_load(path.read_text())
        for key, value in baseline.items():
            if key in RUNTIME or key in RELOCATED:
                continue
            assert proposal.get(key) == value, f'{variant}: changed {key}'
        allowed = set(baseline) | {'model_conf', 'rir_model_type'}
        assert set(proposal) <= allowed
        model = dict(proposal['model_conf'])
        assert model.pop('ctf_length') == taps == 60
        assert model.pop('ctf_block_type') == variant
        assert model.pop('relation_checkpoint') is True
        assert model.pop('length_aware') is False
        assert model == expected_model, f'{variant}: non-head model settings differ'
        assert proposal['rir_model_type'] == 'rec_rir_pit_lag_relation'
        assert proposal['init_param'] == [] and proposal['freeze_param'] == []
        assert proposal['valid_batch_size'] is None
        assert proposal['fold_length'] == [32000] * 5
        destination = Path(manifest['destination'])
        for mode in ('train', 'valid'):
            expected_specs = [
                [str(Path('exp/original_baseline_inputs') / Path(index).relative_to(destination)),
                 name, kind]
                for index, name, kind in manifest['data_specs'][mode]
            ]
            assert proposal[f'{mode}_data_path_and_name_and_type'] == expected_specs
            assert [spec[1] for spec in expected_specs] == list(NAMES)
            expected_shapes = [
                str(Path('exp/original_baseline_inputs') / Path(shape).relative_to(destination))
                for shape in manifest['shape_files'][mode]
            ]
            assert proposal[f'{mode}_shape_file'] == expected_shapes
        result[variant] = digest(path)
    return dict(baseline_config_sha256=BASELINE_CONFIG_SHA256,
                config_sha256=result, preparation_sha256=digest(inputs / 'preparation.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('exp/original_baseline_inputs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    report = audit_configs(args.inputs)
    manifest = json.loads((args.inputs / 'preparation.json').read_text())
    for field in ('destination_scp_sha256', 'destination_shape_sha256'):
        for relative, expected in manifest[field].items():
            assert digest(args.inputs / relative) == expected, relative
    inventory = args.inputs / 'audio_manifest.jsonl'
    assert digest(inventory) == manifest['audio_manifest_sha256']
    records = [json.loads(line) for line in inventory.read_text().splitlines()]
    assert len({row['package_path'] for row in records}) == len(records)

    def verify(row):
        relative = Path(row['package_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        wave = args.inputs / relative
        assert wave.stat().st_size == row['bytes'], relative
        assert digest(wave) == row['sha256'] == row['staged_sha256'], relative
        return row['bytes']

    if not args.metadata_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            total_bytes = sum(pool.map(verify, records))
        report.update(audio_bytes=total_bytes, exact_audio_bytes_verified=True)
    else:
        report['exact_audio_bytes_verified'] = False
    report.update(audio_files=len(records), counts=manifest['counts'],
                  original_baseline_retrained=False, shared_speech_initializer=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
