#!/usr/bin/env python3
"""Generate/validate two-speaker WHAMR tt/min with measured BUT RIRs.

The single-speaker dump is never modified. Both speech sources use the same
room, receiver setup and microphone, but different measured source positions.
"""
import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

import create_whamr_but_dump as common

RECIPE = common.RECIPE
RATES = (8000, 16000)
RESERVE = 100 * 1024**3
SIGNALS = ('source1', 'source2', 'direct1', 'direct2', 'reverb1', 'reverb2',
           'noise', 'mix_clean', 'mix_both')


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def coordinates(item, kind):
    xyz = tuple(float(item['metadata']['$Env' + kind + axis])
                for axis in ('Depth', 'Width', 'Height'))
    if not all(math.isfinite(v) for v in xyz):
        raise ValueError('Nonfinite geometry: ' + item['name'])
    return xyz


def assign_pairs(rirs, inventory, count, seed):
    """Preserve single-source assignments; balance partners for each source."""
    index = {item['name']: item for item in inventory}
    groups = defaultdict(list)
    for i, (name, _, _, _) in enumerate(rirs):
        item = index[name]
        groups[item['room'], item['microphone_setup'], item['microphone']].append(i)
    for key, indices in groups.items():
        items = [index[rirs[i][0]] for i in indices]
        if len(items) < 2 or len({coordinates(x, 'Mic1') for x in items}) != 1:
            raise ValueError('Need >=2 sources at one fixed microphone: ' + str(key))
        if len({coordinates(x, 'Spk1') for x in items}) != len(items):
            raise ValueError('Repeated physical source coordinates: ' + str(key))
    first = common.assign_rirs(rirs, count, seed)
    second = np.empty(count, dtype=int)
    rng = np.random.default_rng(seed + 1)
    for indices in groups.values():
        for i in indices:
            slots = np.flatnonzero(first == i)
            partners = np.asarray([j for j in indices if j != i])
            choices = np.arange(len(slots)) % len(partners)
            rng.shuffle(choices)
            second[slots] = partners[choices]
    return np.stack((first, second), axis=1)


def load_sources(args):
    with args.filenames.open() as f:
        rows = {r['output_filename']: r for r in csv.DictReader(f)}
    scaling = np.load(args.scaling, allow_pickle=True)
    ids = [str(x) for x in scaling['utterance_id']]
    if len(set(ids)) != len(ids) or set(ids) != set(rows):
        raise ValueError('WHAM tt CSV/scaling ID mismatch')
    return rows, scaling, ids


def scp_references(rec, condition):
    paths = rec['paths']
    refs = {'wav.scp': paths['mix_clean' if condition == 'clean' else 'mix_both'],
            'speech_reverb.scp': paths['mix_clean'], 'noise1.scp': paths['noise']}
    refs['speech_mix.scp'] = refs['wav.scp']
    for speaker in (1, 2):
        for key in ('source', 'direct', 'reverb'):
            filename = (f'source{speaker}.scp' if key == 'source'
                        else f'speech_{key}{speaker}.scp')
            refs[filename] = paths[f'{key}{speaker}']
        refs[f'spk{speaker}.scp'] = paths[f'direct{speaker}']
        refs[f'rir_ref{speaker}.scp'] = rec['rir_paths'][speaker - 1]
        refs[f'rir{speaker}.scp'] = rec['rir_paths'][speaker - 1]
    return refs


def generate(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Refusing existing output: ' + str(out))
    rows, scaling, ids = load_sources(args)
    limit = len(ids) if args.limit is None else min(args.limit, len(ids))
    headers = {}
    sample_count = 0
    for filename in ids:
        row = rows[filename]
        sources = [args.wsj_root / row[f's{s}_path'] for s in (1, 2)]
        if sources[0].parent.name == sources[1].parent.name:
            raise ValueError('Expected two distinct WSJ speakers: ' + filename)
        for path in sources + [args.noise_root / 'tt' / filename]:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in sources:
            if path not in headers:
                headers[path] = sf.info(path)
        for sr in RATES:
            sample_count += min(math.ceil(headers[p].frames * sr / headers[p].samplerate)
                                for p in sources)
    # Estimate the entire dataset even for a smoke run; never consume the reserve.
    estimated = int((sample_count * 4 * len(SIGNALS) + 512 * 1024**2) * 1.1)
    if shutil.disk_usage(out.parent).free < RESERVE + estimated:
        raise OSError('Insufficient space for estimated output plus 100 GiB reserve')
    out.mkdir(parents=True)
    save_json(out / 'capacity.json', dict(estimated_bytes=estimated, reserve_bytes=RESERVE))
    rirs = common.load_rirs(args.rir_root, out)
    inventory = json.loads((out / 'rir_inventory.json').read_text())
    assignment = assign_pairs(rirs, inventory, len(ids), args.seed)
    inventory_by_name = {r['name']: r for r in inventory}
    config = dict(
        generator_sha256=common.digest(Path(__file__)), helper_sha256=common.digest(Path(common.__file__)),
        split='tt', length_mode='min', speakers=['s1', 's2'], sample_rates=list(RATES),
        seed=args.seed, limit=args.limit, workers=args.workers,
        filenames=str(args.filenames.resolve()), filenames_sha256=common.digest(args.filenames),
        scaling=str(args.scaling.resolve()), scaling_sha256=common.digest(args.scaling),
        wsj_root=str(args.wsj_root.resolve()), noise_root=str(args.noise_root.resolve()),
        rir_root=str(args.rir_root.resolve()), inventory_sha256=common.digest(out / 'rir_inventory.json'),
        archive_sha256_record=(args.rir_root.parent / 'SHA256SUMS').read_text().strip(),
        microphone='01', microphone_setup='MicID01', rir_version='v00',
        assignment='same s1 assignment as single-speaker seed; balance distinct s2 partners per s1; shared across rates/conditions',
        rir='full distributed 1-second RIR, resampled and abs-peak normalized; retain distributed delay compensation',
        direct='one signed absolute-peak tap, not an independently verified physical direct onset',
        gains='per-speaker WHAM wsjmix gains -> int16 quantization -> shared WHAM speech gain; original noise gain/crop',
        joint_gain='one attenuation for all nine signals per utterance/rate, shared by clean/noisy',
        tail='convolve before clipping to min length; save full distributed RIR',
        snr='not fixed; no SIR matching after convolution',
    )
    save_json(out / 'generation_config.json', config)
    cached, rir_meta = {}, []
    for sr in RATES:
        for name, wave, fs, member in rirs:
            g = math.gcd(sr, fs)
            h = resample_poly(wave, sr // g, fs // g)
            gain = float(np.max(np.abs(h)))
            if gain <= 0:
                raise ValueError('Zero RIR')
            h = (h / gain).astype(np.float32).astype(np.float64)
            peak = int(np.argmax(np.abs(h)))
            path = common.write_audio(out / 'audio' / str(sr) / 'rir' / name, h, sr)
            cached[sr, name] = (h, peak, path)
            rir_meta.append(dict(sample_rate=sr, name=name, but_member=member, path=path,
                                 samples=len(h), peak_sample=peak, normalization_divisor=gain,
                                 sha256=common.digest(path)))
    save_json(out / 'rir_metadata.json', rir_meta)

    def make_utterance(i):
        if shutil.disk_usage(out).free < RESERVE:
            raise OSError('Disk reserve reached; generation incomplete')
        filename = ids[i]
        row = rows[filename]
        source_paths = [args.wsj_root / row[f's{s}_path'] for s in (1, 2)]
        names = [rirs[j][0] for j in assignment[i]]
        room = inventory_by_name[names[0]]['room']
        results = []
        for sr in RATES:
            key = f'{sr // 1000}k_min'
            original = [common.read_audio(p, sr) for p in source_paths]
            length = min(map(len, original))
            wsj_gains = [float(x) for x in scaling['scaling_wsjmix_' + key][i]]
            speech_gain = float(scaling['scaling_wham_speech_' + key][i])
            noise_gain = float(scaling['scaling_wham_noise_' + key][i])
            signals, peaks, rir_paths = {}, [], []
            for s in (1, 2):
                source = common.quantize(original[s - 1][:length] * wsj_gains[s - 1]) * speech_gain
                h, peak, path = cached[sr, names[s - 1]]
                direct = np.zeros(length)
                if peak < length:
                    direct[peak:] = source[:length - peak] * h[peak]
                signals[f'source{s}'] = source
                signals[f'direct{s}'] = direct
                signals[f'reverb{s}'] = fftconvolve(source, h)[:length]
                peaks.append(peak)
                rir_paths.append(path)
            start = int(scaling['speech_start_sample_16k'][i]) // (16000 // sr)
            noise = common.read_audio(args.noise_root / 'tt' / filename, sr)[start:start + length] * noise_gain
            if len(noise) != length:
                raise ValueError('Noise shorter than min utterance: ' + filename)
            signals['noise'] = noise
            signals['mix_clean'] = signals['reverb1'] + signals['reverb2']
            signals['mix_both'] = signals['mix_clean'] + noise
            peak = max(float(np.max(np.abs(x))) for x in signals.values())
            joint_gain = min(1.0, 0.99 / max(peak, 1e-12))
            paths = {k: common.write_audio(out / 'audio' / str(sr) / k / filename, x * joint_gain, sr)
                     for k, x in signals.items()}
            results.append(dict(
                utterance_id=Path(filename).stem, sample_rate=sr, samples=length, room=room,
                speakers=[p.parent.name for p in source_paths],
                original_sources=[str(p.resolve()) for p in source_paths],
                original_noise=str((args.noise_root / 'tt' / filename).resolve()),
                but_rirs=names, rir_paths=rir_paths, direct_peak_samples=peaks,
                wsj_gains=wsj_gains, speech_gain=speech_gain, noise_gain=noise_gain,
                noise_start_sample=start, joint_gain=joint_gain, paths=paths,
                sir_reverberant_db=common.power_ratio(signals['reverb1'], signals['reverb2']),
                snr_reverberant_db=common.power_ratio(signals['mix_clean'], noise)))
        return results

    manifests = defaultdict(lambda: defaultdict(list))
    room_counts, pair_counts = Counter(), Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (out / 'utterances.jsonl').open('w') as stream:
        for i, records in enumerate(pool.map(make_utterance, range(limit)), 1):
            for rec in records:
                stream.write(json.dumps(rec, allow_nan=False) + '\n')
                uid, sr = rec['utterance_id'], rec['sample_rate']
                if sr == 16000:
                    room_counts[rec['room']] += 1
                    pair_counts[' | '.join(rec['but_rirs'])] += 1
                for condition in ('clean', 'noisy'):
                    name = f'tt_but_2spk_{condition}_reverb_min_{sr // 1000}k'
                    files = manifests[name]
                    for key, path in scp_references(rec, condition).items():
                        files[key].append((uid, path))
                    # A mixture has two speakers: use utterance IDs for Kaldi's grouping.
                    files['utt2spk'].append((uid, uid))
                    files['spk2utt'].append((uid, uid))
                    files['utt2spk1'].append((uid, rec['speakers'][0]))
                    files['utt2spk2'].append((uid, rec['speakers'][1]))
                    files['utt2num_samples'].append((uid, str(rec['samples'])))
            if i % 100 == 0 or i == limit:
                print(f'Generated {i}/{limit} two-speaker entries at both rates', flush=True)
    for name, files in manifests.items():
        folder = out / 'raw' / name
        folder.mkdir(parents=True)
        for key, entries in files.items():
            (folder / key).write_text(''.join(f'{uid} {value}\n' for uid, value in sorted(entries)))
        (folder / 'feats_type').write_text('raw\n')
    save_json(out / 'generation_complete.json', dict(sets=sorted(manifests), utterances_per_set=limit,
              room_counts=dict(room_counts), ordered_pair_counts=dict(pair_counts), rir_count=len(rirs)))
    print('Generation complete: ' + str(out), flush=True)


def validate(out, workers=4):
    out = out.resolve()
    (out / 'validation.json').unlink(missing_ok=True)
    config = json.loads((out / 'generation_config.json').read_text())
    summary = json.loads((out / 'generation_complete.json').read_text())
    inventory = json.loads((out / 'rir_inventory.json').read_text())
    by_name = {r['name']: r for r in inventory}
    for item in inventory:
        if common.digest(Path(item['original_path'])) != item['sha256']:
            raise ValueError('Original RIR hash mismatch')
    if common.digest(out / 'rir_inventory.json') != config['inventory_sha256']:
        raise ValueError('Inventory hash mismatch')
    for key in ('filenames', 'scaling'):
        if common.digest(Path(config[key])) != config[key + '_sha256']:
            raise ValueError('Original source metadata changed')
    with Path(config['filenames']).open() as f:
        source_rows = {Path(r['output_filename']).stem: r for r in csv.DictReader(f)}
    scaling = np.load(config['scaling'], allow_pickle=True)
    id_to_index = {Path(str(uid)).stem: i for i, uid in enumerate(scaling['utterance_id'])}
    cached = {}
    for item in json.loads((out / 'rir_metadata.json').read_text()):
        info = sf.info(item['path'])
        if (info.samplerate, info.channels, info.frames, info.subtype) != (item['sample_rate'], 1, item['samples'], 'FLOAT'):
            raise ValueError('RIR header mismatch')
        if common.digest(Path(item['path'])) != item['sha256']:
            raise ValueError('RIR hash mismatch')
        cached[item['path']] = common.read_audio(item['path'], item['sample_rate'])
    records = [json.loads(line) for line in (out / 'utterances.jsonl').read_text().splitlines()]
    if len(records) != 2 * summary['utterances_per_set']:
        raise ValueError('Record count mismatch')
    by_key = {(r['sample_rate'], r['utterance_id']): r for r in records}
    if len(by_key) != len(records):
        raise ValueError('Duplicate records')

    def check_record(rec):
        sr, n, uid = rec['sample_rate'], rec['samples'], rec['utterance_id']
        left, right = [by_name[name] for name in rec['but_rirs']]
        if any(left[k] != right[k] for k in ('room', 'microphone_setup', 'microphone')):
            raise ValueError('Sources do not share a receiver')
        if coordinates(left, 'Mic1') != coordinates(right, 'Mic1') or coordinates(left, 'Spk1') == coordinates(right, 'Spk1'):
            raise ValueError('Invalid source/receiver geometry')
        if rec['room'] != left['room']:
            raise ValueError('Room label mismatch')
        pair = by_key[16000 if sr == 8000 else 8000, uid]
        if pair['but_rirs'] != rec['but_rirs']:
            raise ValueError('RIR assignment differs between rates')
        if abs(2 * by_key[8000, uid]['samples'] - by_key[16000, uid]['samples']) > 1:
            raise ValueError('Length mismatch across rates')
        signals = {}
        if set(rec['paths']) != set(SIGNALS):
            raise ValueError('Missing or extra signal types')
        for key, path in rec['paths'].items():
            info = sf.info(path)
            if (info.samplerate, info.channels, info.frames, info.subtype) != (sr, 1, n, 'FLOAT'):
                raise ValueError('Signal header mismatch: ' + path)
            signals[key] = common.read_audio(path, sr)
            if np.max(np.abs(signals[key])) > 0.990001:
                raise ValueError('Unexpected peak after common attenuation')
        errors = [np.max(np.abs(signals['mix_clean'] - signals['reverb1'] - signals['reverb2'])),
                  np.max(np.abs(signals['mix_both'] - signals['mix_clean'] - signals['noise']))]
        index = id_to_index[uid]
        suffix = f'{sr // 1000}k_min'
        originals = []
        for s in (1, 2):
            source_path = Path(config['wsj_root']) / source_rows[uid][f's{s}_path']
            if str(source_path.resolve()) != rec['original_sources'][s - 1]:
                raise ValueError('Wrong source filename')
            original = common.read_audio(source_path, sr)
            originals.append(original)
            expected = common.quantize(original[:n] * scaling['scaling_wsjmix_' + suffix][index][s - 1])
            expected *= scaling['scaling_wham_speech_' + suffix][index] * rec['joint_gain']
            errors.append(np.max(np.abs(expected - signals[f'source{s}'])))
            rir_path = rec['rir_paths'][s - 1]
            if Path(rir_path).name != rec['but_rirs'][s - 1] or Path(rir_path).parent.parent.name != str(sr):
                raise ValueError('Wrong source RIR path')
            h = cached[rir_path]
            peak = int(np.argmax(np.abs(h)))
            if peak != rec['direct_peak_samples'][s - 1]:
                raise ValueError('Wrong direct peak')
            errors.append(np.max(np.abs(fftconvolve(signals[f'source{s}'], h)[:n] - signals[f'reverb{s}'])))
            direct = np.zeros(n)
            if peak < n:
                direct[peak:] = signals[f'source{s}'][:n - peak] * h[peak]
            errors.append(np.max(np.abs(direct - signals[f'direct{s}'])))
        if min(map(len, originals)) != n:
            raise ValueError('Wrong min length')
        start = int(scaling['speech_start_sample_16k'][index]) // (16000 // sr)
        noise_path = Path(config['noise_root']) / 'tt' / source_rows[uid]['output_filename']
        noise = common.read_audio(noise_path, sr)[start:start + n]
        noise *= scaling['scaling_wham_noise_' + suffix][index] * rec['joint_gain']
        errors.append(np.max(np.abs(noise - signals['noise'])))
        worst = float(max(errors))
        if not math.isfinite(worst) or worst > 3e-6:
            raise ValueError(f'Signal identity failed for {uid}: {worst}')
        return worst

    with ThreadPoolExecutor(max_workers=workers) as pool:
        worst = max(pool.map(check_record, records))
    for name in summary['sets']:
        folder = out / 'raw' / name
        sr = 8000 if name.endswith('_8k') else 16000
        condition = 'noisy' if '_noisy_' in name else 'clean'
        subset = {uid: rec for (rate, uid), rec in by_key.items() if rate == sr}
        wanted = set(scp_references(next(iter(subset.values())), condition))
        if {p.name for p in folder.glob('*.scp')} != wanted:
            raise ValueError('Wrong SCP file schema')
        for key in wanted | {'utt2num_samples', 'utt2spk', 'spk2utt', 'utt2spk1', 'utt2spk2'}:
            entries = [line.split(maxsplit=1) for line in (folder / key).read_text().splitlines()]
            if len(entries) != len(subset) or {uid for uid, _ in entries} != set(subset):
                raise ValueError('SCP key mismatch')
            for uid, value in entries:
                rec = subset[uid]
                if key.endswith('.scp'):
                    expected = scp_references(rec, condition)[key]
                elif key == 'utt2num_samples':
                    expected = str(rec['samples'])
                elif key in ('utt2spk1', 'utt2spk2'):
                    expected = rec['speakers'][int(key[-1]) - 1]
                else:
                    expected = uid
                if value != expected:
                    raise ValueError('Incorrect mapping: ' + str(folder / key))
    rooms = Counter(r['room'] for r in records if r['sample_rate'] == 16000)
    if config['limit'] is None and (len(rooms) != 9 or max(rooms.values()) - min(rooms.values()) > 1):
        raise ValueError('Unbalanced rooms')
    report = dict(sets=len(summary['sets']), utterances_per_set=summary['utterances_per_set'],
                  audio_tuples_checked=len(records), room_counts=dict(sorted(rooms.items())),
                  rir_count=len(inventory), max_signal_identity_error=worst,
                  checks='source gains/quantization; paired rates; distinct source geometry at fixed microphone; both convolutions/direct references; mixture/noise sums; RIR hashes; all SCP mappings')
    save_json(out / 'validation.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECIPE / 'dump_but_2spk')
    parser.add_argument('--rir-root', type=Path, default=Path('/net/midgar/work2/nitsu/data/BUT_ReverbDB/rir_metadata'))
    parser.add_argument('--wsj-root', type=Path, default=RECIPE.parent / 'enh_rir/data/wsj0/wsj0_wav')
    parser.add_argument('--noise-root', type=Path, default=Path('/net/midgar/work2/nitsu/data/wsj/wham_noise'))
    parser.add_argument('--filenames', type=Path, default=RECIPE.parent / 'enh_rir/whamr_scripts/data/mix_2_spk_filenames_tt.csv')
    parser.add_argument('--scaling', type=Path, default=Path('/net/midgar/work2/nitsu/data/wsj/wham_noise/metadata/scaling_tt.npz'))
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    if not args.validate_only:
        generate(args)
    validate(args.output, args.workers)


if __name__ == '__main__':
    main()
