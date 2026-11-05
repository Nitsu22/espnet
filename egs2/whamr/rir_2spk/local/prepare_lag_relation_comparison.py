"""Prepare one NF8k dataset and one untrained speech initialization for all heads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scp(path):
    rows = [line.split(maxsplit=1) for line in Path(path).read_text().splitlines()]
    result = dict(rows)
    if len(result) != len(rows) or not result:
        raise ValueError(f"Empty or duplicate-key SCP: {path}")
    return result


def prepare(source, output, workers):
    source = source.resolve(strict=True)
    output = output.resolve()
    if not output.is_relative_to(Path('/gs/bs/tga-shinoda/nitsu')):
        raise ValueError("Output must be inside the user's TSUBAME storage")
    saved = json.loads((source / 'preparation.json').read_text())
    assert saved['spec']['noise'] is False and saved['spec']['num_spk'] == 2
    assert saved['spec']['sample_rate'] == '16k' and saved['spec']['length'] == 'min'
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True)
    names = ['speech_mix', 'speech_direct1', 'speech_direct2',
             'speech_reverb1', 'speech_reverb2', 'rir_ref1', 'rir_ref2',
             'rir_direct1', 'rir_direct2']
    hashes, counts, source_hashes, ranges = {}, {}, {}, {}
    worst_residual = 0.0
    for split in ('tr', 'cv', 'tt'):
        relative = Path('raw') / f'{split}_rir_2spk_nf_min_16k'
        folder = source / relative
        maps = {}
        for name in names:
            index = folder / f'{name}.scp'
            value = sha256(index)
            if value != saved['output_hashes'][str(relative / index.name)]:
                raise ValueError(f"Source index checksum mismatch: {index}")
            source_hashes[str(index)] = value
            maps[name] = read_scp(index)
        ids = sorted(maps['speech_mix'])
        assert all(set(mapping) == set(ids) for mapping in maps.values())
        assert len(ids) == saved['counts'][split]
        dest = output / 'raw' / f'{split}_rir_2spk_nf_min_8k'
        dest.mkdir(parents=True)

        def convert(uid):
            signals, files = {}, {}
            for name in names:
                wave, rate = sf.read(maps[name][uid], dtype='float32', always_2d=True)
                assert rate == 16000 and wave.shape[1] == 2
                assert np.isfinite(wave).all()
                wave = resample_poly(wave[:, 0], 1, 2)
                assert np.isfinite(wave).all()
                target = output / 'audio' / split / name / f'{uid}.wav'
                sf.write(target, wave, 8000, subtype='FLOAT')
                files[name] = str(target)
                if name.startswith('speech_'):
                    signals[name] = wave
            length = len(signals['speech_mix'])
            assert all(len(wave) == length for wave in signals.values())
            if length < 7552:
                raise ValueError(f"{uid}: only {length} samples; CTF60 needs 7552")
            residual = np.max(np.abs(signals['speech_mix'] -
                                      signals['speech_reverb1'] -
                                      signals['speech_reverb2']))
            # Independent PCM quantization in the source may leave rounding noise.
            if residual > 2e-4:
                raise ValueError(f"{uid}: not a noise-free mixture: {residual}")
            return uid, files, length, float(residual)

        for name in names:
            (output / 'audio' / split / name).mkdir(parents=True)
        lines = {name: [] for name in names}
        lengths = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, (uid, files, length, residual) in enumerate(pool.map(convert, ids), 1):
                for name in names:
                    lines[name].append(f'{uid} {files[name]}\n')
                lengths.append(length)
                worst_residual = max(worst_residual, residual)
                if idx % 2000 == 0:
                    print(f'{split}: {idx}/{len(ids)} converted', flush=True)
        for name, rows in lines.items():
            (dest / f'{name}.scp').write_text(''.join(rows))
        (dest / 'wav.scp').write_text(''.join(lines['speech_mix']))
        (dest / 'utt2num_samples').write_text(
            ''.join(f'{uid} {length}\n' for uid, length in zip(ids, lengths)))
        (dest / 'feats_type').write_text('raw\n')
        (dest / 'channel').write_text('0\n')
        counts[split] = len(ids)
        ranges[split] = [min(lengths), max(lengths)]
        for path in dest.iterdir():
            hashes[str(path.relative_to(output))] = sha256(path)
        print(f'{split}: complete, {len(ids)} utterances', flush=True)
    report = dict(spec=dict(source=str(source), sample_rate='8k', length='min',
                            num_spk=2, channel=0, noise=False),
                  counts=counts, output_hashes=hashes, source_hashes=source_hashes,
                  source_preparation_sha256=sha256(source / 'preparation.json'),
                  conversion='scipy.signal.resample_poly(1,2), left channel, FLOAT',
                  sample_ranges=ranges, max_noise_free_residual=worst_residual)
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.workers)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
