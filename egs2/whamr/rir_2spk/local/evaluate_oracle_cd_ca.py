#!/usr/bin/env python3
"""Evaluate oracle-CD CA reconstruction against unprocessed physical CA RIRs."""
import argparse
import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import socket
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from espnet2.rir.rec_rir.ca_evaluation import CAReconstructor

SCORER_COMMIT = '4e8a0f5b9b0bc5d73e3e5ace5b42d4b29b300f72'
VARIANTS = {
    'bimamba': ('rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_scratch_plateau_200ep',
                'baseline_scratch_plateau_200ep_training_20261022', '165epoch.pth'),
    'bilstm': ('rir_train_pooled_bilstm_2spk_nf_8k_sweep_v2_pre8_post4_time_l1_plateau_200ep',
               'bilstm_pre8_post4_time_l1_training_20261023', '141epoch.pth'),
    'rec_rir': (None, 'rec_rir_ls8k_20261015', '97epoch.pth'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_index(path):
    pairs = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError(f'Duplicate keys in {path}')
    return result


def select_pairs(candidates, references, score_pair, sr=8000, length=16000):
    """Score four reference-conditioned candidates, then use one utterance PIT."""
    matrix = [[score_pair(candidates[p, r], references[r], sr, length,
                          'peak', 'peak', 2.5, None, False)
               for r in range(2)] for p in range(2)]
    permutations = ((0, 1), (1, 0))
    costs = [sum(matrix[p][r]['rmse'] for p, r in enumerate(perm)) / 2
             for perm in permutations]
    if not np.isfinite(costs).all():
        raise ValueError('Nonfinite PIT costs')
    best = int(np.argmin(costs))
    return best, permutations[best], matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    import subprocess
    from espnet2.tasks.rir import RIRTask
    recipe, output = args.recipe.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    code_root = Path(__file__).resolve().parents[4]
    scorer_root = recipe / 'exp/frequency_ls_eval_code_20261014'
    assert subprocess.check_output(['git', '-C', str(scorer_root), 'rev-parse', 'HEAD'], text=True).strip() == SCORER_COMMIT
    assert not subprocess.check_output(['git', '-C', str(scorer_root), 'status', '--porcelain'], text=True).strip()
    sys.path.insert(0, str(scorer_root / 'egs2/whamr/rir_2spk/local'))
    scoring = importlib.import_module('score_rir_pit')
    model_name, control_name, checkpoint_name = VARIANTS[args.variant]
    model_dir = (recipe / 'exp' / model_name if model_name else
                 recipe.parent / 'rir/exp/rir_train_rec_rir_2spk_pit_clean_8192')
    checkpoint = model_dir / checkpoint_name
    control = recipe / 'exp' / control_name
    if args.variant == 'rec_rir':
        completed = json.loads((control / 'metadata.json').read_text())
        assert json.loads((control / 'complete.json').read_text())['count'] == 3000
    else:
        assert (control / 'exit_status').read_text().strip() == '0'
        assert (control / 'final_phase').read_text().strip() == 'complete'
        completed = json.loads((control / 'training_complete.json').read_text())
        assert Path(completed['checkpoint']).resolve() == checkpoint.resolve()
    assert digest(checkpoint) == completed['checkpoint_sha256']
    assert digest(model_dir / 'config.yaml') == completed['config_sha256']
    config = yaml.safe_load((model_dir / 'config.yaml').read_text())
    assert config['sample_rate'] == config['model_conf']['sr'] == 8000
    assert config['model_conf']['pim_sweep_duration'] == 8.192
    data = recipe / 'dump_nf_2spk_8k_min/raw/tt_rir_2spk_nf_min_8k'
    names = ('wav.scp', 'rir_direct1.scp', 'rir_direct2.scp', 'rir_ref1.scp', 'rir_ref2.scp')
    indexes = {name: read_index(data / name) for name in names}
    keys = sorted(indexes['wav.scp'])
    assert len(keys) == 3000 and all(set(i) == set(keys) for i in indexes.values())
    canonical = json.loads((recipe / 'dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k/preparation.json').read_text())
    for name, sha in canonical['source_index_sha256'].items():
        assert digest(data / name) == sha, name
    if args.limit:
        assert args.limit > 0
        keys = keys[:args.limit]
    output.mkdir(parents=True)
    start = time.monotonic()
    model, _ = RIRTask.build_model_from_file(str(model_dir / 'config.yaml'), str(checkpoint), device=args.device)
    model.eval()
    reconstructor = CAReconstructor(model.pim, model.transforms, 16000)
    metadata = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
        host=socket.gethostname(), gpu_index=int(__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '-1')),
        variant=args.variant, model_directory=str(model_dir.resolve()), checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=digest(checkpoint), config_sha256=digest(model_dir / 'config.yaml'),
        parameters=sum(p.numel() for p in model.parameters()),
        commit=subprocess.check_output(['git', '-C', str(code_root), 'rev-parse', 'HEAD'], text=True).strip(),
        scoring_commit=SCORER_COMMIT, data_index_sha256={n:digest(data/n) for n in names},
        evaluation='oracle true CD, original-sweep inverse, raw CA ground truth',
        sample_rate=8000, rir_length=16000, input='full utterance, channel0, canonical FLOAT NF mixture',
        inverse_calibration='fixed known sweep gain and delay; no prediction gain/peak fit',
        reference_sweep_processing=False, align='peak', scale_mode='peak', polarity_align=False,
        pit='one global permutation per utterance, paired reference CD and CA, minimum full RIR RMSE',
        limit=args.limit)
    (output/'started.json').write_text(json.dumps(metadata, indent=2)+'\n')
    all_rows, utterances, strict_rows = [], [], []
    def load(name, uid, is_rir):
        path = indexes[name][uid]
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        assert sr == 8000 and np.isfinite(audio).all()
        audio = audio[:, 0]
        if is_rir:
            if len(audio) > 16000 and np.any(audio[16000:] != 0):
                raise ValueError('Reference exceeds 2 seconds')
            audio = np.pad(audio[:16000], (0, max(0, 16000-len(audio))))
        elif sf.info(path).subtype != 'FLOAT':
            raise ValueError('Expected canonical FLOAT mixture')
        return audio
    with (output/'per_source.jsonl').open('w') as details, (output/'rir.scp').open('w') as index:
        for number, uid in enumerate(keys, 1):
            wav = load('wav.scp', uid, False)
            direct = np.stack([load(f'rir_direct{i}.scp', uid, True) for i in (1, 2)])
            references = [load(f'rir_ref{i}.scp', uid, True) for i in (1, 2)]
            if any(not np.any(r) for r in references):
                raise ValueError('Zero CA RIR')
            with torch.no_grad():
                ctf = model.estimate_ctf(torch.as_tensor(wav, device=args.device))
                assert ctf.shape == (2, 129, 60)
                candidates = reconstructor.candidate_rirs(ctf, torch.as_tensor(direct, device=args.device)).cpu().numpy()
            if not np.isfinite(candidates).all():
                raise ValueError('Nonfinite CA predictions')
            best, perm, matrix = select_pairs(candidates, references, scoring.score_pair)
            utterances.append(best)
            paths = [None, None]
            for p, r in enumerate(perm):
                row = dict(utt_id=uid, pred_index=p, ref_index=r, perm_index=best, **matrix[p][r])
                all_rows.append(row)
                details.write(json.dumps(row, allow_nan=True)+'\n')
                # Same PIT assignment, additionally report unaligned and unscaled
                # absolute reconstruction metrics; selection remains shape RMSE.
                strict_rows.append(scoring.score_pair(candidates[p, r], references[r],
                    8000, 16000, 'none', 'none', 2.5, None, False))
                folder = output / f'ca_ref{r+1}'
                folder.mkdir(exist_ok=True)
                path = folder / f'{uid}.wav'
                sf.write(path, candidates[p, r], 8000, subtype='FLOAT')
                paths[r] = str(path)
            index.write(f'{uid} {paths[0]} {paths[1]}\n')
            if number % 100 == 0 or number == len(keys):
                details.flush(); index.flush()
                print(f'{args.variant}: evaluated {number}/{len(keys)}', flush=True)
    def summarize(rows):
        summary = scoring.aggregate_pair_metrics(rows)
        for metric in ('rt60', 'drr', 'c50'):
            summary[metric+'_pearson'] = scoring.finite_pearson(
                (r[metric+'_pred'] for r in rows), (r[metric+'_ref'] for r in rows))
        summary.update(num_utts=len(keys), num_source_pairs=len(rows),
            valid_counts={k:sum(math.isfinite(float(row[k])) for row in rows)
                          for k in ('rmse','rmse_direct_50ms','corr','rt60_err','drr_err','c50_err')})
        return summary
    summary = summarize(all_rows)
    assert summary['valid_counts']['rmse'] == 2*len(keys)
    assert summary['valid_counts']['corr'] == 2*len(keys)
    report = dict(finished_at=datetime.datetime.now().astimezone().isoformat(),
                  elapsed_seconds=time.monotonic()-start, metadata=metadata,
                  summary=summary, unaligned_unscaled_same_pit=summarize(strict_rows),
                  perm0_ratio=float(np.mean(np.array(utterances)==0)))
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'evaluation_complete.json').write_text(json.dumps(dict(
        finished_at=report['finished_at'], count=len(keys), source_pairs=2*len(keys),
        summary_sha256=digest(output/'summary.json')), indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
