"""Compare fixed Rec-RIR checkpoints on identical FLOAT WHAMR inputs."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_index(path):
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines()]
    result = dict(rows)
    if len(rows) != len(result) or len(rows) != 3000:
        raise ValueError(f'Expected 3000 distinct utterances: {path}')
    return result


def run(arguments, pythonpath=None):
    environment = os.environ.copy()
    if pythonpath is not None:
        environment['PYTHONPATH'] = str(pythonpath)
    subprocess.run([sys.executable, '-B', *map(str, arguments)], check=True, env=environment)


def validate_prediction(folder, ids):
    rows = [line.split() for line in (folder / 'rir.scp').read_text().splitlines()]
    if len(rows) != len(ids) or {row[0] for row in rows} != set(ids):
        raise ValueError('Incomplete or duplicate prediction IDs')
    for row in rows:
        if len(row) != 3:
            raise ValueError('Expected two predicted RIRs per mixture')
        for name in row[1:]:
            info = sf.info(name)
            if (info.samplerate != 8000 or info.frames != 16000
                    or info.channels != 1 or info.subtype != 'FLOAT'):
                raise ValueError(f'Invalid output format: {name}')
            wave, _ = sf.read(name)
            if not np.isfinite(wave).all() or not np.any(wave):
                raise ValueError(f'Invalid waveform: {name}')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ls-code', type=Path, required=True)
    args = parser.parse_args()
    recipe, output, ls_code = args.recipe.resolve(), args.output.resolve(), args.ls_code.resolve()
    data = recipe / 'dump_nf_2spk_8k_min/raw/tt_rir_2spk_nf_min_8k'
    reference = recipe / 'dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k'
    preparation = json.loads((reference / 'preparation.json').read_text())
    assert preparation['full_dataset'] and preparation['count'] == 3000
    assert preparation['config']['sample_rate'] == 8000
    assert preparation['config']['rir_length'] == 16000 and preparation['channel'] == 0
    indexes = {}
    for name in ('wav.scp', 'rir_ref1.scp', 'rir_ref2.scp'):
        assert digest(data / name) == preparation['source_index_sha256'][name]
        indexes[name] = read_index(data / name)
    ids = list(indexes['wav.scp'])
    assert all(set(index) == set(ids) for index in indexes.values())
    inputs = output / 'input'
    inputs.mkdir(exist_ok=False)
    shutil.copy2(data / 'wav.scp', inputs / 'wav.scp')
    (inputs / 'smoke.scp').write_text(''.join(
        f"{uid} {indexes['wav.scp'][uid]}\n" for uid in ids[:3]))
    physical = inputs / 'physical_rir.scp'
    physical.write_text(''.join(
        f"{uid} {indexes['rir_ref1.scp'][uid]} {indexes['rir_ref2.scp'][uid]}\n" for uid in ids))
    sweep_source = recipe / 'exp/rir_train_rec_rir_2spk_nf_8k_sweep_v2'
    training = recipe / 'exp/rec_rir_sweep_v2_training_20261017'
    assert (training / 'exit_status').read_text().strip() == '0'
    sweep_provenance = json.loads((training / 'training_complete.json').read_text())
    original_source = recipe / 'exp/rec_rir_ls8k_20261015'
    original_provenance = json.loads((original_source / 'metadata.json').read_text())
    models = {}
    for label, source, provenance, kind in (
        ('rec_rir_sweep_v2_best45', sweep_source, sweep_provenance, 'rec_rir_sweep_v2_pit'),
        ('rec_rir_3loss_best97', original_source, original_provenance, 'rec_rir_pit'),
    ):
        assert digest(source / 'valid.loss.best.pth') == provenance['checkpoint_sha256']
        assert digest(source / 'config.yaml') == provenance['config_sha256']
        config = yaml.safe_load((source / 'config.yaml').read_text())
        assert config['sample_rate'] == config['model_conf']['sr'] == 8000
        assert config['rir_model_type'] == kind
        assert config['model_conf']['num_spk'] == 2
        if kind == 'rec_rir_pit':
            assert all(config['model_conf'][key] == 1 for key in (
                'loss_w_cln', 'loss_w_rvb', 'loss_w_rec'))
        folder = output / 'models' / label
        folder.mkdir(parents=True, exist_ok=False)
        for filename in ('config.yaml', 'valid.loss.best.pth'):
            shutil.copy2(source / filename, folder / filename)
        metadata = dict(inference='ordinary sweep PIM', sample_rate=8000,
            rir_length=16000, count=3000, input='full utterance, channel 0, identical canonical FLOAT NF mixture',
            experiment=str(source), checkpoint=Path(provenance['checkpoint']).name,
            checkpoint_sha256=digest(folder / 'valid.loss.best.pth'),
            config_sha256=digest(folder / 'config.yaml'),
            input_index_sha256=digest(inputs / 'wav.scp'))
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        models[label] = folder
    for label, folder in models.items():
        print(f'Smoke inference: {label}', flush=True)
        run(['-m', 'espnet2.bin.rec_rir_pit_inference',
            '--train_config', folder / 'config.yaml', '--model_file', folder / 'valid.loss.best.pth',
            '--wav_scp', inputs / 'smoke.scp', '--output_dir', folder / 'smoke',
            '--sample_rate', '8000', '--rir_length', '16000', '--output_subtype', 'FLOAT', '--device', 'cuda:0'])
        validate_prediction(folder / 'smoke', ids[:3])
    (output / 'smoke_complete.json').write_text(json.dumps(dict(models=list(models), count=3)) + '\n')
    scorer = ls_code / 'egs2/whamr/rir_2spk/local/score_rir_pit.py'
    for label, folder in models.items():
        print(f'Full inference: {label}', flush=True)
        run(['-m', 'espnet2.bin.rec_rir_pit_inference',
            '--train_config', folder / 'config.yaml', '--model_file', folder / 'valid.loss.best.pth',
            '--wav_scp', inputs / 'wav.scp', '--output_dir', folder / 'inference',
            '--sample_rate', '8000', '--rir_length', '16000', '--output_subtype', 'FLOAT', '--device', 'cuda:0'])
        validate_prediction(folder / 'inference', ids)
        print(f'Physical WHAMR scoring: {label}', flush=True)
        run([scorer, '--ref_rir_scp', physical,
            '--pred_rir_scp', folder / 'inference/rir.scp', '--out_dir', folder / 'physical',
            '--sample_rate', '8000', '--rir_length', '16000', '--align', 'peak',
            '--scale_mode', 'peak', '--pit_metric', 'rmse'])
    ls_arguments = [ls_code / 'egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py',
                    '--reference_dir', reference, '--out_dir', output / 'ls', '--no_comparison_csv']
    for label, folder in models.items():
        ls_arguments.extend(['--prediction', label, folder / 'inference/rir.scp'])
    run(ls_arguments, pythonpath=ls_code)
    ls_report = json.loads((output / 'ls/comparison.json').read_text())
    assert ls_report['count'] == 3000 and ls_report['source_pairs'] == 6000
    assert not (output / 'ls/comparison.csv').exists()
    result = dict(count=3000, source_pairs=6000, sample_rate=8000, rir_length=16000,
        inference='ordinary sweep PIM', align='peak', scale_mode='peak', pit_metric='rmse',
        polarity_align=False, input_index_sha256=digest(inputs / 'wav.scp'),
        physical_reference_sha256=digest(physical),
        ls_reference_preparation_sha256=ls_report['reference_preparation_sha256'], models={})
    for label, folder in models.items():
        physical_summary = json.loads((folder / 'physical/summary.json').read_text())
        assert physical_summary['num_utts'] == 3000 and physical_summary['num_source_pairs'] == 6000
        with (folder / 'physical/per_source.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        counts = {key: sum(math.isfinite(float(row[key])) for row in rows)
                  for key in ('rmse', 'rmse_direct_50ms', 'corr', 'rt60_err', 'drr_err', 'c50_err')}
        assert all(counts[key] == 6000 for key in ('rmse', 'rmse_direct_50ms', 'corr'))
        result['models'][label] = dict(metadata=json.loads((folder / 'metadata.json').read_text()),
            physical=physical_summary, physical_valid_counts=counts,
            ls=ls_report['models'][label]['metrics'],
            ls_valid_counts=ls_report['models'][label]['valid_counts'])
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'evaluation_complete.json').write_text(json.dumps(dict(
        finished_at=datetime.datetime.now().astimezone().isoformat(), count=3000,
        source_pairs=6000, models=list(models), comparison_sha256=digest(output / 'comparison.json'),
        ls_comparison_sha256=digest(output / 'ls/comparison.json')), indent=2) + '\n')


if __name__ == '__main__':
    main()
