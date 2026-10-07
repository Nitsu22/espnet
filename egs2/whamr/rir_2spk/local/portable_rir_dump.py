"""Package indexed RIR dumps without altering waveforms; verify relocated data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import soundfile as sf


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_scp(path):
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] in seen or any('|' in x for x in fields[1:]):
            raise ValueError(f'Invalid file-only SCP: {path}')
        seen.add(fields[0])
        rows.append((fields[0], fields[1:]))
    return rows


def prepare(source, output, destination, kind, workers):
    source, output = source.resolve(strict=True), output.absolute()
    destination = destination.absolute()
    if output.exists():
        raise FileExistsError(output)
    if kind == 'nf' and not (source / 'preparation.json').exists():
        raise ValueError('Missing NF preparation manifest')
    if kind == 'but' and not (source / 'generation_complete.json').exists():
        raise ValueError('Incomplete BUT generation')
    scps = sorted((source / 'raw').rglob('*.scp'))
    mappings = {}
    aliases = {}
    for scp in scps:
        for _, paths in read_scp(scp):
            for value in paths:
                original = Path(value).resolve(strict=True)
                if original.suffix.lower() != '.wav':
                    raise ValueError(f'Expected WAV reference: {original}')
                if original not in mappings:
                    if kind == 'but':
                        relative = original.relative_to(source)
                    else:
                        relative = Path('audio') / original.relative_to(
                            source.parents[1] / 'enh_rir/data_rir_plus')
                    mappings[original] = relative
                aliases[value] = str(destination / mappings[original])
    # Keep all BUT original RIRs as well as the resampled indexed RIRs.
    if kind == 'but':
        for original in sorted((source / 'original_rir').glob('*.wav')):
            mappings[original.resolve()] = original.relative_to(source)
    total = sum(path.stat().st_size for path in mappings)
    print(f'{kind}: {len(mappings)} unique WAVs, {total} bytes', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.portable_', dir=output.parent))

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            if value in aliases:
                return aliases[value]
            if value.startswith(str(source) + '/'):
                return str(destination / Path(value).relative_to(source))
        return value

    # Preserve original metadata verbatim as provenance, separately from SCPs.
    metadata = [p for p in source.rglob('*') if p.is_file() and
                p.suffix.lower() != '.wav' and 'audio' not in p.relative_to(source).parts]
    for path in metadata:
        relative = path.relative_to(source)
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if relative.suffix == '.scp':
            target.write_text(''.join(f'{uid} {" ".join(aliases[x] for x in values)}\n'
                                      for uid, values in read_scp(path)))
        elif relative.suffix == '.json':
            target.write_text(json.dumps(relocate(json.loads(path.read_text())), indent=2) + '\n')
        elif relative.suffix == '.jsonl':
            with target.open('w') as stream:
                for line in path.read_text().splitlines():
                    stream.write(json.dumps(relocate(json.loads(line))) + '\n')
    provenance = temporary / 'provenance/midgar'
    provenance.mkdir(parents=True, exist_ok=True)
    for name in ('preparation.json', 'generation_config.json', 'generation_complete.json',
                 'rir_metadata.json', 'utterances.jsonl', 'validation.json', 'assignment_audit.json'):
        if (source / name).exists():
            shutil.copy2(source / name, provenance / name)

    def pack(item):
        path, relative = item
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        header = sf.info(path)
        return dict(path=str(relative), sha256=digest(path), bytes=path.stat().st_size,
                    sample_rate=header.samplerate, channels=header.channels, frames=header.frames)

    with (temporary / 'audio_checksums.jsonl').open('w') as stream:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(pool.map(pack, mappings.items()), 1):
                stream.write(json.dumps(record) + '\n')
                if index % 10000 == 0 or index == len(mappings):
                    print(f'{kind}: hashed {index}/{len(mappings)} WAVs', flush=True)
    if kind == 'nf':
        prep = json.loads((temporary / 'preparation.json').read_text())
        prep['output_hashes'] = {str(p.relative_to(temporary)): digest(p)
                                 for p in sorted((temporary / 'raw').rglob('*')) if p.is_file()}
        prep['portable_source_provenance'] = 'provenance/midgar/preparation.json'
        (temporary / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    hashes = {str(p.relative_to(temporary)): digest(p) for p in sorted(temporary.rglob('*'))
              if p.is_file() and p.suffix.lower() != '.wav'}
    report = dict(schema=1, kind=kind, source_dump=str(source), destination=str(destination),
                  audio_files=len(mappings), audio_bytes=total, metadata_sha256=hashes,
                  waveform_conversion=False)
    (temporary / 'portable_dump.json').write_text(json.dumps(report, indent=2) + '\n')
    temporary.rename(output)
    print(f'Prepared: {output}', flush=True)


def verify_metadata(root, require_complete=True):
    root = Path(root)
    manifest = json.loads((root / 'portable_dump.json').read_text())
    if require_complete:
        complete = json.loads((root / 'transfer_complete.json').read_text())
        if complete['portable_manifest_sha256'] != digest(root / 'portable_dump.json'):
            raise ValueError('Transfer manifest changed since verification')
        if Path(manifest['destination']).resolve() != root.resolve():
            raise ValueError('Portable dump moved from its verified destination')
    for name, expected in manifest['metadata_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Metadata checksum mismatch: {name}')
    return manifest


def verify(root, workers):
    root = root.resolve(strict=True)
    manifest = verify_metadata(root, require_complete=False)
    destination = Path(manifest['destination'])
    records = [json.loads(line) for line in (root / 'audio_checksums.jsonl').read_text().splitlines()]
    if len(records) != manifest['audio_files'] or len({r['path'] for r in records}) != len(records):
        raise ValueError('Audio inventory count/uniqueness mismatch')

    def check(record):
        path = root / record['path']
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Audio checksum mismatch: {path}')
        header = sf.info(path)
        if (header.samplerate, header.channels, header.frames) != (
                record['sample_rate'], record['channels'], record['frames']):
            raise ValueError(f'Audio header mismatch: {path}')
        return record['path']

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(check, records), 1):
            if index % 10000 == 0 or index == len(records):
                print(f'Checked {index}/{len(records)} WAVs', flush=True)
    inventory = {r['path']: r for r in records}
    sets = {}
    for folder in sorted((root / 'raw').iterdir()):
        if not folder.is_dir():
            continue
        maps = {p.name: dict(read_scp(p)) for p in folder.glob('*.scp')}
        ids = set(maps['wav.scp'])
        lengths = dict(line.split() for line in (folder / 'utt2num_samples').read_text().splitlines())
        if set(lengths) != ids or any(set(rows) != ids for rows in maps.values()):
            raise ValueError(f'SCP ID mismatch: {folder}')
        sr = 16000 if folder.name.endswith('16k') else 8000
        for name, rows in maps.items():
            for uid, paths in rows.items():
                for value in paths:
                    relative = str(Path(value).relative_to(destination))
                    record = inventory[relative]
                    if record['sample_rate'] != sr or record['frames'] <= 0:
                        raise ValueError(f'Invalid sample rate/length: {value}')
                    if not name.startswith('rir') and record['frames'] != int(lengths[uid]):
                        raise ValueError(f'Speech lengths differ: {folder}/{uid}')
        sets[folder.name] = len(ids)
    report = dict(portable_manifest_sha256=digest(root / 'portable_dump.json'),
                  verified_audio_files=len(records), verified_audio_bytes=manifest['audio_bytes'],
                  sets=sets, sha256_verified=True, headers_verified=True)
    (root / 'transfer_complete.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pack = sub.add_parser('prepare')
    pack.add_argument('--source', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--destination', type=Path, required=True)
    pack.add_argument('--kind', choices=['nf', 'but'], required=True)
    pack.add_argument('--workers', type=int, default=4)
    check = sub.add_parser('verify')
    check.add_argument('--root', type=Path, required=True)
    check.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.output, args.destination, args.kind, args.workers)
    else:
        verify(args.root, args.workers)


if __name__ == '__main__':
    main()
