"""Evaluate pooled Rec-RIR and BiMamba; reuse verified matched Rec-RIR scores."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import shutil

import yaml

from evaluate_rec_rir_sweep_v2_8k import (
    digest, read_index, run, validate_prediction,
)


def validate_context(recipe):
    data = recipe / 'dump_nf_2spk_8k_min/raw/tt_rir_2spk_nf_min_8k'
    reference = recipe / 'dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k'
    prior = recipe / 'exp/rec_rir_sweep_v2_comparison_20261020'
    complete = json.loads((prior / 'evaluation_complete.json').read_text())
    assert (prior / 'exit_status').read_text().strip() == '0'
    assert (prior / 'final_phase').read_text().strip() == 'complete'
    assert digest(prior / 'comparison.json') == complete['comparison_sha256']
    assert digest(prior / 'ls/comparison.json') == complete['ls_comparison_sha256']
    result = json.loads((prior / 'comparison.json').read_text())
    prior_ls = json.loads((prior / 'ls/comparison.json').read_text())
    prep = json.loads((reference / 'preparation.json').read_text())
    assert prep['full_dataset'] and prep['count'] == 3000 and prep['channel'] == 0
    assert prep['config']['sample_rate'] == 8000 and prep['config']['rir_length'] == 16000
    assert digest(reference / 'preparation.json') == result['ls_reference_preparation_sha256']
    assert digest(data / 'wav.scp') == result['input_index_sha256']
    assert (result['count'], result['source_pairs'], result['sample_rate'], result['rir_length']) == (3000, 6000, 8000, 16000)
    assert result['inference'] == 'ordinary sweep PIM'
    assert result['align'] == result['scale_mode'] == 'peak'
    assert result['pit_metric'] == 'rmse' and result['polarity_align'] is False
    indexes = {}
    for key in ('wav.scp', 'rir_ref1.scp', 'rir_ref2.scp'):
        assert digest(data / key) == prep['source_index_sha256'][key]
        indexes[key] = read_index(data / key)
    ids = list(indexes['wav.scp'])
    assert all(set(index) == set(ids) for index in indexes.values())
    physical = ''.join(f"{uid} {indexes['rir_ref1.scp'][uid]} {indexes['rir_ref2.scp'][uid]}\n" for uid in ids)
    assert (prior / 'input/physical_rir.scp').read_text() == physical
    assert digest(prior / 'input/physical_rir.scp') == result['physical_reference_sha256']
    assert set(result['models']) == {'rec_rir_sweep_v2_best45', 'rec_rir_3loss_best97'}
    for label, record in result['models'].items():
        folder = prior / 'models' / label
        metadata = record['metadata']
        assert metadata['input_index_sha256'] == result['input_index_sha256']
        assert digest(folder / 'config.yaml') == metadata['config_sha256']
        assert digest(folder / 'valid.loss.best.pth') == metadata['checkpoint_sha256']
        assert digest(folder / 'inference/rir.scp') == prior_ls['models'][label]['prediction_index_sha256']
        validate_prediction(folder / 'inference', ids)
        assert json.loads((folder / 'physical/summary.json').read_text()) == record['physical']
        assert prior_ls['models'][label]['metrics'] == record['ls']
        record['reused_from'] = str(folder)
    pooled_run = recipe / 'exp/rec_rir_pooled_sweep_v2_training_20261020'
    assert (pooled_run / 'exit_status').read_text().strip() == '0'
    assert (pooled_run / 'final_phase').read_text().strip() == 'complete'
    pooled_provenance = json.loads((pooled_run / 'training_complete.json').read_text())
    baseline_comparison = json.loads((recipe / 'exp/rec_rir_ls8k_20261015/comparison/comparison.json').read_text())
    baseline_provenance = baseline_comparison['models']['BiMamba_8k_best']['prediction_metadata']
    specs = [
        ('rec_rir_pooled_sweep_v2_best44', recipe / 'exp/rir_train_rec_rir_pooled_2spk_nf_8k_sweep_v2',
         pooled_provenance, 'rec_rir_pooled_sweep_v2_pit', 44),
        ('bimamba_baseline_sweep_v2_best45', recipe / 'exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2',
         baseline_provenance, 'pooled_bimamba_sweep_v2_pit', 45),
    ]
    for label, source, provenance, kind, epoch in specs:
        assert digest(source / 'valid.loss.best.pth') == provenance['checkpoint_sha256']
        assert digest(source / 'config.yaml') == provenance['config_sha256']
        assert (source / 'valid.loss.best.pth').resolve().name == f'{epoch}epoch.pth'
        config = yaml.safe_load((source / 'config.yaml').read_text())
        assert config['rir_model_type'] == kind
        assert config['sample_rate'] == config['model_conf']['sr'] == 8000
        assert config['model_conf']['num_spk'] == 2 and config['model_conf']['ctf_taps'] == 60
        assert config['model_conf'].get('drr_loss_weight', 0) == 0
        if kind == 'pooled_bimamba_sweep_v2_pit':
            assert '# Ended (code 0)' in (source / 'train.log').read_text()
    return result, specs, indexes, ids, physical, reference, prior


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ls-code', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    recipe, output, ls_code = args.recipe.resolve(), args.output.resolve(), args.ls_code.resolve()
    result, specs, indexes, ids, physical, reference, prior = validate_context(recipe)
    if args.validate_only:
        print(json.dumps(dict(count=len(ids), new_models=[s[0] for s in specs],
                              reused_models=list(result['models']),
                              input_sha256=result['input_index_sha256'],
                              ls_reference_sha256=result['ls_reference_preparation_sha256']), indent=2))
        return
    inputs = output / 'input'
    inputs.mkdir(exist_ok=False)
    (inputs / 'wav.scp').write_text(''.join(f'{uid} {indexes["wav.scp"][uid]}\n' for uid in ids))
    assert digest(inputs / 'wav.scp') == result['input_index_sha256']
    (inputs / 'smoke.scp').write_text(''.join(f'{uid} {indexes["wav.scp"][uid]}\n' for uid in ids[:3]))
    (inputs / 'physical_rir.scp').write_text(physical)
    folders = {}
    for label, source, provenance, kind, epoch in specs:
        folder = output / 'models' / label
        folder.mkdir(parents=True, exist_ok=False)
        for filename in ('config.yaml', 'valid.loss.best.pth'):
            shutil.copy2(source / filename, folder / filename)
        metadata = dict(inference='ordinary sweep PIM', sample_rate=8000, rir_length=16000,
            count=3000, input='full utterance, channel 0, identical canonical FLOAT NF mixture',
            experiment=str(source), checkpoint=f'{epoch}epoch.pth',
            checkpoint_sha256=digest(folder / 'valid.loss.best.pth'),
            config_sha256=digest(folder / 'config.yaml'), input_index_sha256=result['input_index_sha256'])
        assert metadata['checkpoint_sha256'] == provenance['checkpoint_sha256']
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        folders[label] = folder
    for label, folder in folders.items():
        print(f'Smoke inference: {label}', flush=True)
        run(['-m', 'espnet2.bin.rec_rir_pit_inference',
             '--train_config', folder / 'config.yaml', '--model_file', folder / 'valid.loss.best.pth',
             '--wav_scp', inputs / 'smoke.scp', '--output_dir', folder / 'smoke',
             '--sample_rate', '8000', '--rir_length', '16000', '--output_subtype', 'FLOAT', '--device', 'cuda:0'])
        validate_prediction(folder / 'smoke', ids[:3])
    (output / 'smoke_complete.json').write_text(json.dumps(dict(models=list(folders), count=3)) + '\n')
    for label, folder in folders.items():
        print(f'Full inference: {label}', flush=True)
        run(['-m', 'espnet2.bin.rec_rir_pit_inference',
             '--train_config', folder / 'config.yaml', '--model_file', folder / 'valid.loss.best.pth',
             '--wav_scp', inputs / 'wav.scp', '--output_dir', folder / 'inference',
             '--sample_rate', '8000', '--rir_length', '16000', '--output_subtype', 'FLOAT', '--device', 'cuda:0'])
        validate_prediction(folder / 'inference', ids)
        print(f'Physical WHAMR scoring: {label}', flush=True)
        run([ls_code / 'egs2/whamr/rir_2spk/local/score_rir_pit.py',
             '--ref_rir_scp', inputs / 'physical_rir.scp', '--pred_rir_scp', folder / 'inference/rir.scp',
             '--out_dir', folder / 'physical', '--sample_rate', '8000', '--rir_length', '16000',
             '--align', 'peak', '--scale_mode', 'peak', '--pit_metric', 'rmse'])
    ls_arguments = [ls_code / 'egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py',
                    '--reference_dir', reference, '--out_dir', output / 'ls', '--no_comparison_csv']
    for label, folder in folders.items():
        ls_arguments.extend(['--prediction', label, folder / 'inference/rir.scp'])
    run(ls_arguments, pythonpath=ls_code)
    ls_report = json.loads((output / 'ls/comparison.json').read_text())
    assert ls_report['reference_preparation_sha256'] == result['ls_reference_preparation_sha256']
    assert (ls_report['count'], ls_report['source_pairs']) == (3000, 6000)
    assert not (output / 'ls/comparison.csv').exists()
    for label, folder in folders.items():
        summary = json.loads((folder / 'physical/summary.json').read_text())
        assert summary['num_utts'] == 3000 and summary['num_source_pairs'] == 6000
        with (folder / 'physical/per_source.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        counts = {key: sum(math.isfinite(float(row[key])) for row in rows)
                  for key in ('rmse', 'rmse_direct_50ms', 'corr', 'rt60_err', 'drr_err', 'c50_err')}
        assert all(counts[key] == 6000 for key in ('rmse', 'rmse_direct_50ms', 'corr'))
        assert all(ls_report['models'][label]['valid_counts'][key] == 6000
                   for key in ('rmse', 'rmse_direct_50ms', 'corr'))
        result['models'][label] = dict(metadata=json.loads((folder / 'metadata.json').read_text()),
            physical=summary, physical_valid_counts=counts,
            ls=ls_report['models'][label]['metrics'], ls_valid_counts=ls_report['models'][label]['valid_counts'])
    result['prior_comparison'] = str(prior / 'comparison.json')
    result['prior_comparison_sha256'] = digest(prior / 'comparison.json')
    result['physical_rt60_note'] = ('The unchanged physical-reference RT60 scorer has a known '
        '356.032 s reference outlier. Physical RT60 ranking is withheld; no sample was removed.')
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'evaluation_complete.json').write_text(json.dumps(dict(
        finished_at=datetime.datetime.now().astimezone().isoformat(), count=3000, source_pairs=6000,
        models=list(result['models']), comparison_sha256=digest(output / 'comparison.json'),
        ls_comparison_sha256=digest(output / 'ls/comparison.json')), indent=2) + '\n')


if __name__ == '__main__':
    main()
