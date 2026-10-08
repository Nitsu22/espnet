#!/usr/bin/env python3
"""Submit one-GPU ablations from midgar and continue only timed-out jobs."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

RECIPE = '/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk'
VARIANTS = ('ffn_only', 'freq_before_pool', '2blocks', 'bilstm')
PROFILE = 'exp/pooled_bimamba_ablations_profile/job_8934455'
PYTHON = '/gs/bs/tga-shinoda/nitsu/anaconda3/envs/tf-locoformer/bin/python'


def remote(command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                           'tsubame', command], check=True, capture_output=True,
                          text=True, timeout=60).stdout.strip()


def remote_python(code, *args):
    return remote(f'cd {shlex.quote(RECIPE)} && ' +
                  shlex.join([PYTHON, '-c', code, *args]))


def estimate(report):
    # Measurements omit disk loading and ESPnet reporting/checkpoint overhead.
    # Explicit allowance: 25% plus 30 s/epoch. 55 epochs is the existing
    # baseline's observed stopping epoch, not a promised convergence time.
    for key in ('steady_update_seconds', 'estimated_valid_step_seconds'):
        if not math.isfinite(report[key]) or report[key] <= 0:
            raise ValueError(f'Invalid measurement: {key}')
    epoch_seconds = 1.25 * (5109 * report['steady_update_seconds'] +
                            5000 * report['estimated_valid_step_seconds']) + 30
    wall_hours = min(24, math.ceil(100 * epoch_seconds / 3600 * 1.1))
    return dict(epoch_seconds=epoch_seconds,
                expected_hours_55_epochs=55 * epoch_seconds / 3600,
                projected_hours_100_epochs=100 * epoch_seconds / 3600,
                wall_hours=wall_hours)


def save(state, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def submit(variant, wall_hours, resume, commit, expected_hours):
    wall = f'{wall_hours:02d}:00:00'
    args = ['qsub', '-terse', '-g', 'tga-shinoda', '-l', f'h_rt={wall}',
            '-N', f'bi_{variant}', 'qsub/pooled_bimamba_ablation_train_1gpu.sh',
            variant, PROFILE, str(resume).lower(), str(wall_hours * 3600)]
    command = (f'cd {shlex.quote(RECIPE)} && '
               f'test "$(git rev-parse HEAD)" = {shlex.quote(commit)} && ' +
               shlex.join(args))
    job_id = remote(command)
    if not job_id.isdecimal():
        raise RuntimeError(f'Unrecognized submission result: {job_id}')
    report = dict(job_id=int(job_id), command=shlex.join(args), commit=commit,
                  resources='gpu_1=1', priority=-5, wall_hours=wall_hours,
                  runtime_assumption_hours=expected_hours,
                  expected_points=.2 * (.7 * expected_hours + .1 * wall_hours),
                  upper_points=.2 * .8 * wall_hours, resume=resume)
    print(json.dumps(dict(variant=variant, submission=report)), flush=True)
    return report


def inspect_jobs(jobs):
    # Queue inspection and small metadata/log tails only; no login-node model
    # construction, checkpoint loading, or audio/statistics processing.
    code = '''
import json, pathlib, re, subprocess, sys
result = {}
for variant, job_id in json.loads(sys.argv[1]).items():
    folder = pathlib.Path(f'exp/pooled_bimamba_ablation_train_jobs/job_{job_id}')
    active = subprocess.run(['qstat', '-j', str(job_id)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    status = folder / 'exit_status'
    log = pathlib.Path(f'exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_{variant}/train.log')
    epochs = []
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 131072))
            epochs = re.findall(r'(\\d+)epoch results:', stream.read().decode(errors='replace'))
    result[variant] = dict(active=active, exit_status=int(status.read_text()) if status.exists() else None,
                           resume_ready=(folder/'resume_ready').exists(),
                           completed_epoch=int(epochs[-1]) if epochs else 0)
print(json.dumps(result))
'''
    return json.loads(remote_python(code, json.dumps(jobs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state['commit'] != args.commit:
            raise ValueError('State belongs to a different commit')
        if set(state['variants']) != set(VARIANTS):
            raise RuntimeError('Incomplete initial launch requires queue reconciliation')
    else:
        code = ('import json,pathlib; p=pathlib.Path(' + repr(PROFILE) + '); '
                'assert (p/"exit_status").read_text().strip()=="0"; '
                'print(json.dumps({s:json.loads((p/(s+".json")).read_text()) '
                'for s in ' + repr(VARIANTS) + '}))')
        profiles = json.loads(remote_python(code))
        state = dict(commit=args.commit, profile_directory=PROFILE,
                     estimate_assumption='5109 train/5000 valid steps; measured GPU time *1.25 +30s/epoch; baseline stopped at55',
                     variants={})
        save(state, args.state)
        for variant in VARIANTS:
            plan = estimate(profiles[variant])
            # Write before submission; never silently repeat an uncertain qsub.
            state['variants'][variant] = dict(plan=plan, phase='submitting', submissions=[])
            save(state, args.state)
            submission = submit(variant, plan['wall_hours'], False, args.commit,
                                plan['expected_hours_55_epochs'])
            state['variants'][variant].update(phase='running', submissions=[submission])
            save(state, args.state)
    if any(v['phase'] == 'submitting' for v in state['variants'].values()):
        raise RuntimeError('Submission needs queue reconciliation before any retry')
    missing_checks = {}
    while any(v['phase'] == 'running' for v in state['variants'].values()):
        jobs = {k:v['submissions'][-1]['job_id'] for k,v in state['variants'].items()
                if v['phase'] == 'running'}
        try:
            observations = inspect_jobs(jobs)
        except (subprocess.SubprocessError, OSError) as error:
            print(f'Queue check failed; retry later: {error}', flush=True)
            time.sleep(60)
            continue
        for variant, observation in observations.items():
            entry = state['variants'][variant]
            entry['last_observation'] = observation
            if observation['active']:
                missing_checks[variant] = 0
                continue
            if observation['exit_status'] == 0:
                entry['phase'] = 'complete'
            elif observation['exit_status'] == 124 and observation['resume_ready']:
                if observation['completed_epoch'] <= entry.get('last_resume_epoch', -1):
                    entry['phase'] = 'failed'
                    print(f'{variant}: no epoch progress across a segment; no retry', flush=True)
                    continue
                entry['last_resume_epoch'] = observation['completed_epoch']
                remaining = max(1, 100 - observation['completed_epoch'])
                hours = min(24, max(1, math.ceil(remaining * entry['plan']['epoch_seconds'] / 3600 * 1.1)))
                entry['phase'] = 'submitting'
                save(state, args.state)
                entry['submissions'].append(submit(variant, hours, True, args.commit,
                                                   min(hours, remaining * entry['plan']['epoch_seconds'] / 3600)))
                entry['phase'] = 'running'
            elif observation['exit_status'] is not None:
                entry['phase'] = 'failed'
                print(f'{variant} failed: {observation}; no automatic retry', flush=True)
            else:
                missing_checks[variant] = missing_checks.get(variant, 0) + 1
                if missing_checks[variant] >= 3:
                    entry['phase'] = 'failed'
                    print(f'{variant}: job disappeared without an exit marker; inspect accounting', flush=True)
        save(state, args.state)
        if any(v['phase'] == 'running' for v in state['variants'].values()):
            time.sleep(60)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
