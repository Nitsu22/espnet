"""Prepare model-independent, zero-copy two-speaker WHAMR indexes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import soundfile as sf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scp(path):
    result = {}
    for line in path.read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in result:
            raise ValueError(f"Duplicate ID in {path}: {uid}")
        result[uid] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data_rir_plus'))
    parser.add_argument('--output', type=Path, default=Path('dump_nf_2spk_16k_min'))
    parser.add_argument('--sample-rate', choices=['8k', '16k'], default='16k')
    parser.add_argument('--length', choices=['min', 'max'], default='min')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if not (source / 'generation_complete.json').exists():
        raise ValueError('Source generation is incomplete')
    names = {'wav': 'mix_clean_reverb', 'speech_mix': 'mix_clean_reverb'}
    for speaker in (1, 2):
        names.update({f'speech_direct{speaker}': f's{speaker}_anechoic',
                      f'speech_reverb{speaker}': f's{speaker}_reverb',
                      f'rir_ref{speaker}': f'rir{speaker}_reverb',
                      f'rir_direct{speaker}': f'rir{speaker}_anechoic'})
    spec = dict(source=str(source), sample_rate=args.sample_rate, length=args.length,
                num_spk=2, channel=0, noise=False,
                script_sha256=digest(Path(__file__)), manifests={})
    for split in ('tr', 'cv', 'tt'):
        folder = source / 'manifests' / f'{split}_{args.length}_{args.sample_rate}'
        for name in set(names.values()):
            p = folder / f'{name}.scp'
            spec['manifests'][str(p)] = digest(p)
    if args.output.exists():
        saved = json.loads((args.output / 'preparation.json').read_text())
        if saved['spec'] != spec:
            raise ValueError('Existing dump has different source/settings; use a new output')
        for name, value in saved['output_hashes'].items():
            if digest(args.output / name) != value:
                raise ValueError(f'Existing dump changed: {name}')
        print('Existing dump verified:', args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.prepare_single_', dir=args.output.parent))
    try:
        counts = {}
        for split in ('tr', 'cv', 'tt'):
            folder = source / 'manifests' / f'{split}_{args.length}_{args.sample_rate}'
            maps = {name: read_scp(folder / f'{name}.scp') for name in set(names.values())}
            ids = sorted(maps[names['wav']])
            if not ids or any(set(m) != set(ids) for m in maps.values()):
                raise ValueError(f'Mismatched source IDs: {split}')
            sr = int(args.sample_rate.replace('k', '000'))

            def inspect(uid):
                headers = {name: sf.info(values[uid]) for name, values in maps.items()}
                if any(h.samplerate != sr or h.channels != 2 or h.frames == 0 for h in headers.values()):
                    raise ValueError(f'{uid}: expected nonempty stereo {sr} Hz data')
                n = headers[names['wav']].frames
                if any(headers[names[f'speech_{kind}{i}']].frames != n
                       for kind in ('direct', 'reverb') for i in (1, 2)):
                    raise ValueError(f'{uid}: direct/reverb lengths differ')
                return n

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                lengths = list(pool.map(inspect, ids))
            dataset = f'{split}_rir_2spk_nf_{args.length}_{args.sample_rate}'
            dest = temporary / 'raw' / dataset
            dest.mkdir(parents=True)
            for name, original in names.items():
                (dest / f'{name}.scp').write_text(''.join(f'{uid} {maps[original][uid]}\n' for uid in ids))
            metadata = source / f'wav{args.sample_rate}' / args.length / split / 'utterances.jsonl'
            speakers = {}
            for line in metadata.read_text().splitlines():
                row = json.loads(line)
                speakers[row['utterance_id']] = '_'.join(Path(row[f'source{i}']).parent.name for i in (1, 2))
            (dest / 'utt2spk').write_text(''.join(f'{u} {speakers[u]}\n' for u in ids))
            groups = {}
            for uid in ids:
                groups.setdefault(speakers[uid], []).append(uid)
            (dest / 'spk2utt').write_text(''.join(f'{s} {" ".join(us)}\n' for s, us in sorted(groups.items())))
            (dest / 'utt2num_samples').write_text(''.join(f'{u} {n}\n' for u, n in zip(ids, lengths)))
            (dest / 'feats_type').write_text('raw\n')
            (dest / 'channel').write_text('0\n')
            counts[split] = len(ids)
            print(split, len(ids), 'headers checked', flush=True)
        hashes = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
        (temporary / 'preparation.json').write_text(json.dumps(dict(spec=spec, counts=counts, output_hashes=hashes), indent=2))
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == '__main__':
    main()
