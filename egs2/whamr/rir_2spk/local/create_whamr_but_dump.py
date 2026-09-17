#!/usr/bin/env python3
"""Build paired, mono WHAMR tt/min BUT ReverbDB dumps (8/16 kHz), without simulation.

Requires numpy, scipy, soundfile. No GPU or pyroomacoustics is needed.
RIRs retain the distributed delay compensation and complete tails. Direct-path reference uses
one signed peak sample, as in VINP's SimACE generator. These are NOT SimACE.
"""
import argparse
import csv
import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

import shutil

MIN_FREE_BYTES = 100 * 1024**3
RECIPE = Path(__file__).resolve().parents[1]
WHAMR = RECIPE.parent


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_audio(path, sr):
    x, fs = sf.read(str(path), dtype='float64', always_2d=True)
    x = x[:, 0]  # WHAMR mono convention: left microphone
    if fs != sr:
        g = math.gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite audio: ' + str(path))
    return x


def quantize(x):
    # Original WHAMR's Matlab-compatible quantization, BEFORE WHAM speech gain.
    return np.round(32768 * x).astype(np.int16).astype(np.float64) / 32768


def write_audio(path, x, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype='FLOAT')
    return str(path.resolve())


def power_ratio(x, n):
    px, pn = float(np.dot(x, x)), float(np.dot(n, n))
    return 10 * math.log10(px / pn) if px > 0 and pn > 0 else None


def extract_rirs(root):
    """Extract only regular RIR/metadata members, never links or arbitrary paths."""
    archive = root.parent / 'BUT_ReverbDB_rel_19_06_RIR-Only.tgz'
    expected = (root.parent / 'SHA256SUMS').read_text().split()[0]
    if digest(archive) != expected:
        raise ValueError('Downloaded archive SHA256 mismatch')
    root.mkdir(parents=True, exist_ok=False)
    count = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = Path(member.name)
            if not member.isfile() or not ('RIR' in path.parts or path.suffix.lower() == '.txt'):
                continue
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive path: ' + member.name)
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as dest:
                shutil.copyfileobj(tar.extractfile(member), dest)
            count += 1
    (root / 'extraction_complete.json').write_text(json.dumps(
        dict(files=count, archive=str(archive), sha256=expected), indent=2))


def load_rirs(root, out):
    if not (root / 'extraction_complete.json').is_file():
        raise ValueError('BUT RIR extraction is incomplete')
    result, inventory, seen = [], [], set()
    for path in sorted(root.glob('*/MicID01/SpkID*/01/RIR/*.v00.wav'),
                       key=lambda p: (not p.parts[-4].endswith('_S'), str(p))):
        relative = path.relative_to(root)
        room, setup, source, mic, _, _ = relative.parts
        position = source.split('_')[0]
        key = (room, setup, position, mic)
        if key in seen:
            continue  # Prefer a dedicated RIR session (_S), then earliest session.
        seen.add(key)
        name = '__'.join(relative.parts)
        x, fs = sf.read(path, dtype='float64')
        if x.ndim != 1 or fs != 16000 or not np.isfinite(x).all() or not np.any(x):
            raise ValueError('Invalid BUT RIR: ' + str(path))
        original = out / 'original_rir' / name
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, original)
        meta_path = path.parent.parent / 'mic_meta.txt'
        metadata = dict(line.split(maxsplit=1) for line in meta_path.read_text().splitlines()
                        if line.startswith('$') and len(line.split(maxsplit=1)) == 2)
        inventory.append(dict(name=name, room=room, microphone_setup=setup,
                              microphone=mic, source_position=position, session=source,
                              source_path=str(path.resolve()), sha256=digest(path),
                              original_path=str(original.resolve()), metadata=metadata))
        result.append((name, x, fs, str(relative)))
    rooms = {r['room'] for r in inventory}
    if len(rooms) != 9:
        raise ValueError(f'Expected 9 rooms with microphone-01 v00 RIRs, got {len(rooms)}')
    (out / 'rir_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return sorted(result, key=lambda item: item[0])


def assign_rirs(rirs, count, seed):
    rooms = defaultdict(list)
    for index, (_, _, _, member) in enumerate(rirs):
        rooms[member.split('/')[0]].append(index)
    rng = np.random.default_rng(seed)
    room_names = sorted(rooms)
    room_assignment = np.arange(count) % len(room_names)
    rng.shuffle(room_assignment)
    assignment = np.empty(count, dtype=int)
    for room_index, room in enumerate(room_names):
        slots = np.flatnonzero(room_assignment == room_index)
        positions = np.arange(len(slots)) % len(rooms[room])
        rng.shuffle(positions)
        assignment[slots] = np.asarray(rooms[room])[positions]
    return assignment


def generate(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Output already exists; use a new directory: ' + str(out))
    with args.filenames.open() as f:
        rows = list(csv.DictReader(f))
    by_name = {r['output_filename']: r for r in rows}
    scaling = np.load(str(args.scaling), allow_pickle=True)
    ids = [str(x) for x in scaling['utterance_id']]
    if len(set(ids)) != len(ids) or set(ids) != set(by_name):
        raise ValueError('tt CSV and scaling IDs do not match uniquely')
    for uid in ids:
        row = by_name[uid]
        for path in (args.wsj_root / row['s1_path'], args.wsj_root / row['s2_path'],
                     args.noise_root / 'tt' / uid):
            if not path.is_file():
                raise FileNotFoundError(path)
    if shutil.disk_usage(out.parent).free < MIN_FREE_BYTES + 20 * 1024**3:
        raise OSError('Require 120 GiB free before generation (100 GiB reserve)')
    out.mkdir(parents=True)
    rirs = load_rirs(args.rir_root, out)
    assignment = assign_rirs(rirs, len(ids), args.seed)
    config = {
        'generator_sha256': digest(Path(__file__)), 'seed': args.seed,
        'split': 'tt', 'length_mode': 'min', 'source': 's1',
        'sample_rates': [8000, 16000], 'limit': args.limit,
        'rir_root': str(args.rir_root.resolve()),
        'microphone_setup': 'MicID01', 'microphone': '01', 'rir_version': 'v00',
        'repeat_selection': 'one per room/setup/source-ID/mic; prefer _S, then earliest session',
        'inventory_sha256': digest(out / 'rir_inventory.json'),
        'archive_sha256_record': (args.rir_root.parent / 'SHA256SUMS').read_text().strip(),
        'filenames': str(args.filenames.resolve()), 'filenames_sha256': digest(args.filenames),
        'scaling': str(args.scaling.resolve()), 'scaling_sha256': digest(args.scaling),
        'wsj_root': str(args.wsj_root.resolve()), 'noise_root': str(args.noise_root.resolve()),
        'assignment': 'seeded balanced rooms, then balanced source positions within room; WHAM scaling ID order; shared across rates/conditions',
        'rir': 'full distributed BUT RIR divided by absolute peak; BUT propagation-delay compensation retained; no extra alignment',
        'direct': 'single signed peak tap, not a +/-2ms direct-sound window',
        'source_gain': 'WHAMR wsjmix gain -> int16 quantization -> WHAM speech gain',
        'noise_gain': 'original WHAM noise gain and start index, NOT fixed 20dB SimACE SNR',
        'joint_gain': 'same attenuation for source/direct/reverb/noise/mix, shared clean/noisy',
        'tail': 'full RIR stored/used; observed speech clipped to WHAMR min length',
        'rir_evaluation': 'no 8192-sample truncation or evaluation alignment baked into reference',
    }
    (out / 'generation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    cached = {}
    rir_metadata = []
    for sr in (8000, 16000):
        for name, x, fs, member in rirs:
            h = resample_poly(x, sr // math.gcd(sr, fs), fs // math.gcd(sr, fs))
            peak = int(np.argmax(np.abs(h)))
            gain = float(np.max(np.abs(h)))
            if gain <= 0:
                raise ValueError('Silent RIR: ' + name)
            # Use exactly the float32 coefficients saved as the reference.
            h = (h / gain).astype(np.float32).astype(np.float64)
            path = write_audio(out / 'audio' / str(sr) / 'rir' / name, h, sr)
            cached[sr, name] = (h, peak, path)
            rir_metadata.append(dict(sample_rate=sr, but_member=member, path=path,
                                     peak_sample=peak, normalization_divisor=gain,
                                     samples=len(h), sha256=digest(path)))
    (out / 'rir_metadata.json').write_text(json.dumps(rir_metadata, indent=2) + '\n')
    manifests = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    limit = len(ids) if args.limit is None else min(args.limit, len(ids))
    with (out / 'utterances.jsonl').open('w') as metadata:
        for i, filename in enumerate(ids[:limit]):
            if shutil.disk_usage(out).free < MIN_FREE_BYTES:
                raise OSError("Disk reserve reached; generation incomplete")
            row = by_name[filename]
            uid = Path(filename).stem
            speaker = Path(row['s1_path']).parent.name
            name = rirs[int(assignment[i])][0]
            for sr in (8000, 16000):
                suffix = str(sr // 1000) + 'k_min'
                a = read_audio(args.wsj_root / row['s1_path'], sr)
                b_len = sf.info(str(args.wsj_root / row['s2_path'])).frames
                b_fs = sf.info(str(args.wsj_root / row['s2_path'])).samplerate
                length = min(len(a), math.ceil(b_len * sr / b_fs))
                wsj_gain = float(scaling['scaling_wsjmix_' + suffix][i][0])
                speech_gain = float(scaling['scaling_wham_speech_' + suffix][i])
                noise_gain = float(scaling['scaling_wham_noise_' + suffix][i])
                source = quantize(a[:length] * wsj_gain) * speech_gain
                h, peak, rir_path = cached[sr, name]
                reverb = fftconvolve(source, h)[:length]
                direct = np.zeros(length)
                if peak < length:
                    direct[peak:] = source[:length - peak] * h[peak]
                noise = read_audio(args.noise_root / 'tt' / filename, sr) * noise_gain
                start = int(scaling['speech_start_sample_16k'][i]) // (16000 // sr)
                noise = noise[start:start + length]
                if len(noise) != length:
                    raise ValueError('Noise is too short: ' + filename)
                mix = reverb + noise
                maximum = max(float(np.max(np.abs(x))) for x in
                              (source, direct, reverb, noise, mix))
                joint_gain = min(1.0, 0.99 / max(maximum, 1e-12))
                signals = dict(source=source, direct=direct, reverb=reverb, noise=noise, mix=mix)
                paths = {key: write_audio(out / 'audio' / str(sr) / key / filename,
                                         val * joint_gain, sr) for key, val in signals.items()}
                rec = dict(utterance_id=uid, sample_rate=sr, speaker=speaker, but_rir=name,
                           room=rirs[int(assignment[i])][3].split("/")[0],
                           original_s1=str((args.wsj_root / row['s1_path']).resolve()),
                           original_s2_for_min_length=str((args.wsj_root / row['s2_path']).resolve()),
                           original_noise=str((args.noise_root / 'tt' / filename).resolve()),
                           samples=length, noise_start_sample=start, wsj_gain=wsj_gain,
                           speech_gain=speech_gain, noise_gain=noise_gain, joint_gain=joint_gain,
                           direct_peak_sample=peak, rir_path=rir_path, paths=paths,
                           snr_reverberant_db=power_ratio(reverb, noise),
                           snr_direct_db=power_ratio(direct, noise))
                metadata.write(json.dumps(rec) + '\n')
                counts[str(sr) + '/' + name] += 1
                for condition in ('clean', 'noisy'):
                    setname = 'tt_but_single_' + condition + '_reverb_min_' + str(sr // 1000) + 'k'
                    m = manifests[setname]
                    input_path = paths['reverb' if condition == 'clean' else 'mix']
                    refs = {'wav.scp': input_path, 'speech_mix.scp': input_path,
                            'speech_reverb.scp': paths['reverb'], 'speech_direct.scp': paths['direct'],
                            'source.scp': paths['source'], 'rir_ref.scp': rir_path,
                            'rir.scp': rir_path, 'spk1.scp': paths['direct']}
                    for key, value in refs.items():
                        m[key].append((uid, value))
                    m['utt2spk'].append((uid, speaker))
                    m['utt2num_samples'].append((uid, str(length)))
            if (i + 1) % 100 == 0 or i + 1 == limit:
                print('Generated %d/%d tt entries at both rates' % (i + 1, limit), flush=True)
    for setname, files in manifests.items():
        folder = out / 'raw' / setname
        folder.mkdir(parents=True)
        for name, values in files.items():
            (folder / name).write_text(''.join(k + ' ' + v + '\n' for k, v in sorted(values)))
        speakers = defaultdict(list)
        for uid, spk in files['utt2spk']:
            speakers[spk].append(uid)
        (folder / 'spk2utt').write_text(''.join(k + ' ' + ' '.join(sorted(v)) + '\n'
                                             for k, v in sorted(speakers.items())))
        (folder / 'feats_type').write_text('raw\n')
    (out / 'generation_complete.json').write_text(json.dumps(dict(
        utterances_per_set=limit, sets=sorted(manifests), rir_counts=counts), indent=2) + '\n')
    print('Generation complete: ' + str(out), flush=True)


def validate(out):
    """Check every saved tuple, all SCP keys/paths, convolution and additive identity."""
    out = out.resolve()
    summary = json.loads((out / 'generation_complete.json').read_text())
    records = [json.loads(line) for line in (out / 'utterances.jsonl').read_text().splitlines()]
    inventory = json.loads((out / 'rir_inventory.json').read_text())
    for item in inventory:
        if digest(Path(item['original_path'])) != item['sha256']:
            raise ValueError('Original RIR hash mismatch')
    for item in json.loads((out / 'rir_metadata.json').read_text()):
        info = sf.info(item['path'])
        if info.channels != 1 or info.samplerate != item['sample_rate'] or info.frames != item['samples']:
            raise ValueError('Reference RIR header mismatch')
        if digest(Path(item['path'])) != item['sha256']:
            raise ValueError('Reference RIR hash mismatch')
    if len(records) != 2 * summary['utterances_per_set']:
        raise ValueError('Unexpected number of utterance records')
    worst = 0.0
    rate_pairs = defaultdict(list)
    for rec in records:
        sr, n = rec['sample_rate'], rec['samples']
        signals = {}
        for key, path in rec['paths'].items():
            info = sf.info(path)
            if info.samplerate != sr or info.channels != 1 or info.frames != n:
                raise ValueError('Invalid shape/rate: ' + path)
            signals[key] = read_audio(path, sr)
        h = read_audio(rec['rir_path'], sr)
        reconstructed = fftconvolve(signals['source'], h)[:n]
        direct = np.zeros(n)
        p = rec['direct_peak_sample']
        if p < n:
            direct[p:] = signals['source'][:n-p] * h[p]
        errors = [np.max(np.abs(reconstructed - signals['reverb'])),
                  np.max(np.abs(signals['mix'] - signals['reverb'] - signals['noise'])),
                  np.max(np.abs(direct - signals['direct']))]
        error = float(max(errors))
        if not np.isfinite(error) or error > 3e-6:
            raise ValueError('Signal identity failed: ' + rec['utterance_id'])
        worst = max(worst, error)
        rate_pairs[rec['utterance_id']].append((sr, rec['but_rir'], n))
    for pair in rate_pairs.values():
        pair.sort()
        if len(pair) != 2 or pair[0][0] != 8000 or pair[1][0] != 16000 or pair[0][1] != pair[1][1] or abs(2*pair[0][2]-pair[1][2]) > 1:
            raise ValueError('8/16 kHz pairing failed')
    by_rate_id = {(r['sample_rate'], r['utterance_id']): r for r in records}
    for name in summary['sets']:
        folder = out / 'raw' / name
        sr = 8000 if name.endswith('_8k') else 16000
        expected = {r['utterance_id'] for r in records if r['sample_rate'] == sr}
        for path in list(folder.glob('*.scp')) + [folder / 'utt2num_samples', folder / 'utt2spk']:
            pairs = [line.split(maxsplit=1) for line in path.read_text().splitlines()]
            if len(pairs) != len(expected) or {p[0] for p in pairs} != expected:
                raise ValueError('SCP key mismatch: ' + str(path))
            for uid, value in pairs:
                if path.suffix == '.scp' and not Path(value).is_file():
                    raise FileNotFoundError(value)
                if path.name in ('wav.scp', 'speech_mix.scp'):
                    key = 'mix' if '_noisy_' in name else 'reverb'
                    if value != by_rate_id[sr, uid]['paths'][key]:
                        raise ValueError('Incorrect input condition: ' + name)
    room_counts = defaultdict(int)
    for rec in records:
        if rec['sample_rate'] == 16000:
            room_counts[rec['room']] += 1
    config = json.loads((out / 'generation_config.json').read_text())
    if config['limit'] is None and max(room_counts.values()) - min(room_counts.values()) > 1:
        raise ValueError('Unbalanced room assignment')
    report = dict(room_counts=dict(sorted(room_counts.items())), rir_count=len(inventory),
                  sets=len(summary['sets']), utterances_per_set=summary['utterances_per_set'],
                  audio_tuples_checked=len(records), max_signal_identity_error=worst,
                  checks='mono/rate/length, convolution, direct-path, additive noise, paired rates, SCP keys/paths/input condition')
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECIPE / 'dump_but')
    parser.add_argument('--rir-root', type=Path, default=Path('/net/midgar/work2/nitsu/data/BUT_ReverbDB/rir_metadata'))
    parser.add_argument('--wsj-root', type=Path, default=WHAMR / 'enh_rir/data/wsj0/wsj0_wav')
    parser.add_argument('--noise-root', type=Path, default=Path('/net/midgar/work2/nitsu/data/wsj/wham_noise'))
    parser.add_argument('--filenames', type=Path, default=WHAMR / 'enh_rir/whamr_scripts/data/mix_2_spk_filenames_tt.csv')
    parser.add_argument('--scaling', type=Path, default=Path('/net/midgar/work2/nitsu/data/wsj/wham_noise/metadata/scaling_tt.npz'))
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--limit', type=int, help='Smoke test only; assignment uses the full tt list')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--extract-only', action='store_true', help='Extract RIR/metadata from downloaded archive, then exit')
    args = parser.parse_args()
    if args.extract_only:
        extract_rirs(args.rir_root)
        return
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if not args.validate_only:
        generate(args)
    validate(args.output)


if __name__ == '__main__':
    main()
