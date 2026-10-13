#!/usr/bin/env python3
"""Temporarily pause an owned rsync and copy disjoint audio directories.

The original driver resumes and performs its full verification/publication.
No source file is changed and no other user's process is signalled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if (source.parent.parent != Path('/tmp/nitsu')
            or not source.parent.name.startswith('room-path-paired.')
            or not str(target).startswith('/net/midgar/work2/nitsu/')
            or '.partial-room-path-paired.' not in target.name
            or not 1 <= args.workers <= 4):
        raise ValueError('Unexpected private copy directories or worker count')
    if source.stat().st_uid != os.getuid() or target.stat().st_uid != os.getuid():
        raise ValueError('Copy directories must be owned by the current user')
    if not str(args.report.resolve()).startswith('/net/midgar/work2/nitsu/'):
        raise ValueError('Report must remain in the permitted experiment tree')
    if args.report.exists():
        raise ValueError('Use a fresh report path')
    complete = json.loads((source / 'completion.json').read_text())
    if complete.get('status') != 'complete' or not complete.get('checked_audio_hashes'):
        raise ValueError('Source must have passed full verification')
    pids = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            command = (item / 'cmdline').read_bytes().split(b'\0')
            command = [os.fsdecode(part) for part in command if part]
            if (command and Path(command[0]).name == 'rsync'
                    and command[-2:] == [str(source) + '/', str(target) + '/']):
                pids.append(int(item.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if not 1 <= len(pids) <= 4:
        raise ValueError('Expected only the original owned rsync process group')
    directories = [p for split in ('tr', 'cv', 'tt')
                   for p in sorted((source / 'audio' / split).iterdir()) if p.is_dir()]
    # Directory targets do not overlap between concurrent workers.
    directories.sort(key=lambda p: (not p.name.startswith('speech_'), p.parent.name != 'tr', str(p)))
    if args.dry_run:
        print(json.dumps(dict(owned_pids=pids, disjoint_directories=len(directories), workers=args.workers)))
        return
    paused, finished = [], []
    start = time.time()
    try:
        for pid in pids:
            proc = Path('/proc') / str(pid)
            command = [os.fsdecode(part) for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
            if proc.stat().st_uid != os.getuid() or command[-2:] != [str(source) + '/', str(target) + '/']:
                raise RuntimeError('Transfer process changed before pause')
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)

        def copy(directory):
            destination = target / directory.relative_to(source)
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', str(directory) + '/', str(destination) + '/'], check=True)
            return str(directory.relative_to(source))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for directory in executor.map(copy, directories):
                finished.append(directory)
                print('Copied ' + directory, flush=True)
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(
            source=str(source), target=str(target), owned_pids=pids,
            copied_directories=finished, workers=args.workers,
            elapsed_seconds=time.time() - start, original_transfer_resumed=True,
            complete=len(finished) == len(directories)), indent=2) + '\n')


if __name__ == '__main__':
    main()
