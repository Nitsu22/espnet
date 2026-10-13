#!/usr/bin/env python3
"""Replace a paused owned verifier only after an identical full check succeeds.

Failure before handover preserves all data and resumes the original processes.
Publication and scratch cleanup happen only after the full four-worker check.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time


def command(pid):
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid():
        raise ValueError('Process owner changed')
    return [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]


def write_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-pid', required=True, type=int)
    parser.add_argument('--partial', required=True, type=Path)
    parser.add_argument('--final', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    partial, final, scratch, results = [p.resolve() for p in
                                      (args.partial, args.final, args.scratch, args.results)]
    if (scratch.parent != Path('/tmp/nitsu')
            or not scratch.name.startswith('room-path-paired.')
            or partial != Path(str(final) + '.partial-' + scratch.name)
            or final.exists() or args.original_pid <= 1):
        raise ValueError('Unexpected private task paths or process')
    for path in (scratch, scratch / 'dump', partial, results):
        if not path.is_dir() or path.stat().st_uid != os.getuid():
            raise ValueError('Only owned task directories may be used')
    for path in (partial, final, results):
        if not str(path).startswith('/net/midgar/work2/nitsu/'):
            raise ValueError('Publication/results must use the permitted tree')
    recipe = Path(__file__).resolve().parent.parent
    code_root = recipe.parent.parent.parent
    actual = subprocess.check_output(['git', '-C', str(code_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(code_root), 'status', '--porcelain'], text=True)
    if actual != args.commit or dirty:
        raise ValueError('Use the exact clean committed checkout')
    copied = json.loads((results / 'parallel_copy.json').read_text())
    launch = json.loads((results / 'launch.json').read_text())
    complete = json.loads((scratch / 'dump/completion.json').read_text())
    if (not copied['complete'] or len(copied['copied_directories']) != 29
            or copied['source'] != str(scratch / 'dump') or copied['target'] != str(partial)
            or complete['status'] != 'complete' or not complete['checked_audio_hashes']):
        raise ValueError('All audio must already be copied and source-verified')
    original = args.original_pid
    original_cmd = command(original)
    expected = ['/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python',
                str(recipe / 'local/prepare_room_path_contrastive_dump.py'),
                '--verify-existing', str(partial)]
    if original_cmd != expected:
        raise ValueError('Original verifier command changed')
    status_lines = (Path('/proc') / str(original) / 'status').read_text().splitlines()
    driver = int(next(row.split()[1] for row in status_lines if row.startswith('PPid:')))
    driver_cmd = command(driver)
    expected_driver = ['bash', str(recipe / 'local/run_room_path_contrastive_dump_lab.sh'),
                       str(final), str(results), launch['commit']]
    if [Path(driver_cmd[0]).name] + driver_cmd[1:] != expected_driver:
        raise ValueError('Original parent driver command changed')
    originals = {driver: driver_cmd, original: original_cmd}
    print(json.dumps(dict(driver=driver, verifier=original, dry_run=args.dry_run)), flush=True)
    if args.dry_run:
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    paused, transferred, checker = [], False, None
    phase, started = 'pausing_original_verifier', time.time()
    try:
        for pid in (driver, original):
            if command(pid) != originals[pid]:
                raise ValueError('Original process identity changed before pause')
            paused.append(pid)
            os.kill(pid, signal.SIGSTOP)
        phase = 'verifying_destination_parallel'
        print(phase, flush=True)
        env = os.environ.copy()
        env.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                   NUMBA_CACHE_DIR=str(scratch / 'numba'), MPLCONFIGDIR=str(scratch / 'matplotlib'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        checker = subprocess.Popen(expected + ['--workers', '4'], env=env)
        if checker.wait() != 0:
            raise RuntimeError('Parallel full verification failed')
        verified = json.loads((partial / 'verification.json').read_text())
        if (verified['counts'] != complete['counts'] or not verified['checked_audio_hashes']
                or not verified['self_contained'] or final.exists()):
            raise ValueError('Destination verification/publication guard failed')
        if any(command(pid) != cmd for pid, cmd in originals.items()):
            raise ValueError('Original process identity changed before handover')
        # Parent must die first so its EXIT trap cannot remove the verified
        # partial dataset when the old paused child is subsequently stopped.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            os.kill(driver, signal.SIGKILL)
            transferred = True
            try:
                os.kill(original, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        phase = 'publishing'
        partial.rename(final)
        shutil.copy2(final / 'verification.json', results / 'verification.json')
        shutil.copy2(final / 'completion.json', results / 'generation_complete.json')
        phase = 'removing_scratch'
        print(phase, flush=True)
        shutil.rmtree(scratch)
        if scratch.exists():
            raise RuntimeError('Task scratch was not removed')
        phase = 'complete'
        report = dict(exit_status=0, phase=phase, host=socket.gethostname(), gpu=None,
                      scratch_directory=str(scratch), scratch_removed=True,
                      git_commit=launch['commit'], finalizer_commit=args.commit, dump=str(final),
                      previous_driver_replaced=True, elapsed_seconds=time.time() - started,
                      finished=datetime.now(timezone(timedelta(hours=9))).isoformat())
        write_json(results / 'pipeline_status.json', report)
        (results / 'exit_status').write_text('0\n')
        print(json.dumps(report), flush=True)
    except BaseException as error:
        if transferred:
            # Originals cannot be resumed after handover. Preserve any
            # remaining data and make the exact recovery state visible.
            report = dict(exit_status=1, phase=phase, host=socket.gethostname(),
                          scratch_directory=str(scratch), scratch_removed=not scratch.exists(),
                          partial_exists=partial.exists(), final_exists=final.exists(),
                          dump=str(final), finalizer_commit=args.commit,
                          error_type=type(error).__name__, error=str(error))
            write_json(results / 'pipeline_status.json', report)
            (results / 'exit_status').write_text('1\n')
            print(json.dumps(report), flush=True)
        raise
    finally:
        if checker is not None and checker.poll() is None:
            checker.terminate()
            try:
                checker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                checker.kill()
                checker.wait()
        if not transferred:
            for pid in reversed(paused):
                try:
                    if command(pid) == originals[pid]:
                        os.kill(pid, signal.SIGCONT)
                except (OSError, ValueError):
                    pass
            print('Original verifier resumed; all private data preserved', flush=True)


if __name__ == '__main__':
    main()
