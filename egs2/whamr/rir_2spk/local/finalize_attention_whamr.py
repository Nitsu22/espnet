#!/usr/bin/env python3
"""Verify completed WHAMR scoring and persist counts and checkpoint provenance."""

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finalize(training, output, expected_utts):
    require(expected_utts > 0, 'Expected utterance count must be positive')
    require(not (output/'evaluation_complete.json').exists(), 'Evaluation already finalized')
    for marker in ('smoke_exit_status', 'inference_exit_status', 'scoring_exit_status'):
        require((output/marker).read_text().strip() == '0', f'Unsuccessful {marker}')
    require((training/'training_exit_status').read_text().strip() == '0', 'Training failed')
    started = read_json(output/'started.json')
    trained = read_json(training/'training_complete.json')
    smoke = read_json(output/'smoke_whamr/complete.json')
    require(smoke['count'] == 2, 'Expected successful two-example smoke evaluation')
    root = output/'whamr'
    metadata = read_json(root/'metadata.json')
    inference = read_json(root/'inference_complete.json')
    complete = read_json(root/'complete.json')
    require(metadata['count'] == inference['count'] == complete['count'] == expected_utts,
            'Incomplete WHAMR inference/scoring')
    require(metadata['git_commit'] == trained['commit'] == started['commit'],
            'Training and evaluation code commits differ')
    checkpoint_hash = hashlib.sha256((root/'valid.loss.best.pth').read_bytes()).hexdigest()
    require(checkpoint_hash == metadata['checkpoint_sha256'] == trained['checkpoint_sha256'],
            'Evaluated checkpoint differs from the trained best checkpoint')
    require(hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest()
            == trained['trained_config_sha256'], 'Evaluated model configuration differs')
    require(metadata['experiment'] == trained['model_directory'] == started['model_directory'],
            'Evaluated experiment directory differs')
    require(metadata['data'] == started['test_data'], 'Evaluated test data differs')
    require(metadata['checkpoint'] == 'valid.loss.best.pth', 'Wrong checkpoint selection')
    score = root/'score'
    summary = read_json(score/'summary.json')
    require(summary['num_utts'] == expected_utts and summary['num_source_pairs'] == 2*expected_utts,
            'Incomplete WHAMR summary')
    expected_protocol = dict(sample_rate=16000, rir_length=32000, align='peak',
                             scale_mode='peak', pit_metric='rmse', polarity_align=False,
                             direct_index_source='reference_peak')
    for key, value in expected_protocol.items():
        require(summary[key] == value, f'WHAMR protocol differs: {key}')
    with (score/'per_source.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2*expected_utts, 'Incomplete source-pair CSV')
    counts_by_id = Counter(row['utt_id'] for row in rows)
    require(len(counts_by_id) == expected_utts and set(counts_by_id.values()) == {2},
            'Expected exactly two scored sources per utterance')
    prediction_rows = [line.split(maxsplit=1) for line in (root/'inference/rir.scp').read_text().splitlines()
                       if line.strip()]
    require(len(prediction_rows) == expected_utts
            and {row[0] for row in prediction_rows} == set(counts_by_id),
            'Scored utterances differ from inferred utterances')
    source_pairs = {(row['utt_id'], row['pred_index'], row['ref_index']) for row in rows}
    require(len(source_pairs) == len(rows), 'Duplicate scored source pairs')
    by_id = defaultdict(list)
    for row in rows:
        by_id[row['utt_id']].append(row)
    for uid, pairs in by_id.items():
        require({row['pred_index'] for row in pairs} == {'0', '1'}
                and {row['ref_index'] for row in pairs} == {'0', '1'},
                f'Incomplete PIT source assignment: {uid}')
    metric_map = dict(rmse='rmse_mean', rmse_50ms='rmse_50ms_mean',
                      rmse_direct_50ms='rmse_direct_50ms_mean', corr='corr_mean',
                      corr_direct_50ms='corr_direct_50ms_mean',
                      rt60_err='rt60_mae', drr_err='drr_mae', c50_err='c50_mae')
    valid_counts = {}
    for metric, aggregate in metric_map.items():
        values = [float(row[metric]) for row in rows]
        valid = [value for value in values if math.isfinite(value)]
        valid_counts[metric] = len(valid)
        require(bool(valid), f'No finite values for {metric}')
        transformed = [abs(value) for value in valid] if metric.endswith('_err') else valid
        calculated = math.fsum(transformed)/len(transformed)
        require(math.isclose(calculated, summary[aggregate], rel_tol=1e-9, abs_tol=1e-12),
                f'Summary disagrees with source CSV: {aggregate}')
    for metric in ('rmse', 'rmse_50ms', 'rmse_direct_50ms'):
        require(valid_counts[metric] == 2*expected_utts, f'Nonfinite waveform error: {metric}')
    now = datetime.datetime.now().astimezone()
    evaluation_start = datetime.datetime.fromisoformat(started['started_at'])
    report = dict(finished_at=now.isoformat(),
                  evaluation_elapsed_seconds=(now-evaluation_start).total_seconds(),
                  commit=trained['commit'], checkpoint_sha256=checkpoint_hash,
                  selected_checkpoint=trained['selected_checkpoint'],
                  source_config_sha256=trained['source_config_sha256'],
                  trained_config_sha256=trained['trained_config_sha256'],
                  inference_seconds=complete['inference_seconds'],
                  scoring_seconds=complete['scoring_seconds'],
                  whamr=dict(summary=summary, valid_counts=valid_counts))
    (score/'valid_metric_counts.json').write_text(json.dumps(valid_counts, indent=2)+'\n')
    (output/'evaluation_complete.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-utts', type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(finalize(args.training, args.output, args.expected_utts), indent=2))


if __name__ == '__main__':
    main()
