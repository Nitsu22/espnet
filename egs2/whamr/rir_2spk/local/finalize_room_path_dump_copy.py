#!/usr/bin/env python3
"""Finalize a fully copied private dump without replaying an old rsync list.

Original owned processes remain paused until full verification succeeds.
Failure before publication resumes the original driver and preserves its data.
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
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--final', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, target, final, results = [p.resolve() for p in
                                    (args.source, args.target, args.final, args.results)]
    scratch = source.parent
    if (scratch.parent != Path('/tmp/nitsu')
            or not scratch.name.startswith('room-path-paired.')
            or source.name != 'dump'
            or target != Path(str(final) + '.partial-' + scratch.name)
            or final.exists()):
        raise ValueError('Unexpected private source/publication paths')
    for path in (scratch, source, target, results):
        if not path.is_dir() or path.stat().st_uid != os.getuid():
            raise ValueError('Only owned task directories may be used')
    for path in (target, final, results):
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
    complete = json.loads((source / 'completion.json').read_text())
    if (not copied['complete'] or len(copied['copied_directories']) != 29
            or copied['source'] != str(source) or copied['target'] != str(target)
            or complete['status'] != 'complete' or not complete['checked_audio_hashes']):
        raise ValueError('All audio directories must already be fully copied and verified')
    pids = copied['owned_pids']
    expected_rsync = ['rsync', '-a', str(source) + '/', str(target) + '/']
    original_commands = {}
    for pid in pids:
        cmd = command(pid)
        if [Path(cmd[0]).name] + cmd[1:] != expected_rsync:
            raise ValueError('Original transfer command changed')
        original_commands[pid] = cmd
    first_status = (Path('/proc') / str(pids[0]) / 'status').read_text().splitlines()
    driver = int(next(row.split()[1] for row in first_status if row.startswith('PPid:')))
    expected_driver = ['bash', str(recipe / 'local/run_room_path_contrastive_dump_lab.sh'),
                       str(final), str(results), launch['commit']]
    driver_cmd = command(driver)
    if [Path(driver_cmd[0]).name] + driver_cmd[1:] != expected_driver:
        raise ValueError('Original driver command changed')
    original_commands[driver] = driver_cmd
    print(json.dumps(dict(driver=driver, transfer_pids=pids, dry_run=args.dry_run)), flush=True)
    if args.dry_run:
        return
    paused, transferred = [], False
    hidden = set()
    phase = 'pausing_original_transfer'
    started = time.time()
    try:
        for pid in [driver] + pids:
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        for pid in pids:
            for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
                try:
                    path = Path(os.readlink(fd))
                    if (str(path).startswith(str(target) + '/')
                            and path.name.startswith('.') and path.is_file()
                            and not path.is_symlink() and path.stat().st_uid == os.getuid()):
                        hidden.add(path)
                except OSError:
                    pass
        phase = 'completing_metadata'
        print(phase, flush=True)
        subprocess.run(['rsync', '-a', '--ignore-existing', str(source) + '/', str(target) + '/'], check=True)
        phase = 'verifying_destination'
        print(phase, flush=True)
        env = os.environ.copy()
        env.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                   NUMBA_CACHE_DIR=str(scratch / 'numba'), MPLCONFIGDIR=str(scratch / 'matplotlib'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run(['/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python',
                        str(recipe / 'local/prepare_room_path_contrastive_dump.py'),
                        '--verify-existing', str(target)], check=True, env=env)
        verified = json.loads((target / 'verification.json').read_text())
        if (verified['counts'] != complete['counts'] or not verified['checked_audio_hashes']
                or not verified['self_contained'] or final.exists()):
            raise ValueError('Destination verification/publication guard failed')
        # Revalidate exact ownership/commands immediately before replacing the
        # old driver; killing it first prevents its EXIT cleanup from running.
        if any(command(p) != cmd for p, cmd in original_commands.items()):
            raise ValueError('Owned process identity changed before handover')
        os.kill(driver, signal.SIGKILL)
        transferred = True
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        for path in hidden:
            if path.exists() and path.stat().st_uid == os.getuid() and not path.is_symlink():
                path.unlink()
        phase = 'publishing'
        target.rename(final)
        shutil.copy2(final / 'verification.json', results / 'verification.json')
        shutil.copy2(final / 'completion.json', results / 'generation_complete.json')
        phase = 'removing_scratch'
        print(phase, flush=True)
        shutil.rmtree(scratch)
        if scratch.exists():
            raise RuntimeError('Task scratch was not removed')
        phase = 'complete'
        status = dict(exit_status=0, phase=phase, host=socket.gethostname(), gpu=None,
                      scratch_directory=str(scratch), scratch_removed=True,
                      git_commit=launch['commit'], finalizer_commit=args.commit, dump=str(final),
                      previous_driver_replaced=True, elapsed_seconds=time.time() - started,
                      finished=datetime.now(timezone(timedelta(hours=9))).isoformat())
        write_json(results / 'pipeline_status.json', status)
        (results / 'exit_status').write_text('0\n')
        print(json.dumps(status), flush=True)
    finally:
        if not transferred:
            for pid in reversed(paused):
                try:
                    if command(pid) == original_commands[pid]:
                        os.kill(pid, signal.SIGCONT)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            print('Original transfer resumed; scratch and partial dump preserved', flush=True)


if __name__ == '__main__':
    main()
