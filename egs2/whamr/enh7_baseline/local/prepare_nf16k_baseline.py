"""Index existing native 16 kHz NF-WHAMR audio and derive mono shapes.

Run in a CPU allocation. No audio is copied/resampled and no feature statistics
are needed by this waveform model (utterance variance normalization only).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def read_scp(path):
    values = {}
    for line in path.read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in values or value.endswith("|"):
            raise ValueError("Duplicate ID or unsupported pipe: " + str(path))
        values[uid] = value
    return values


def write_scp(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("{} {}\n".format(uid, values[uid]) for uid in sorted(values)))


def prepare(source, output, stats):
    ready = output / "preparation.json"
    if ready.exists():
        raise ValueError("Already prepared; use existing verified data or a new directory")
    report = dict(source=str(source.resolve()), sample_rate=16000, channel=0, splits={}, hashes={})
    for split, count in (("tr", 20000), ("cv", 5000), ("tt", 3000)):
        src = source / "raw" / (split + "_rir_2spk_nf_min_16k")
        names = ("wav.scp", "speech_direct1.scp", "speech_direct2.scp",
                 "speech_reverb1.scp", "speech_reverb2.scp")
        maps = [read_scp(src / name) for name in names]
        lengths = read_scp(src / "utt2num_samples")
        ids = sorted(maps[0])
        if len(ids) != count or any(set(m) != set(ids) for m in maps[1:] + [lengths]):
            raise ValueError("Incomplete or mismatched IDs: " + split)
        for uid in ids:
            expected = int(lengths[uid])
            for mapping in maps:
                info = sf.info(mapping[uid])
                if info.samplerate != 16000 or info.channels not in (1, 2) or info.frames != expected:
                    raise ValueError("Wrong audio header: " + mapping[uid])
            if expected <= 0:
                raise ValueError("Empty input: " + uid)
        sample_ids = ids[::max(1, len(ids) // 8)][:8]
        error = 0.0
        for uid in sample_ids:
            waves = [sf.read(m[uid], dtype="float64", always_2d=True)[0][:, 0] for m in maps]
            if any(not np.isfinite(w).all() or not np.any(w) for w in waves):
                raise ValueError("Invalid waveform: " + uid)
            error = max(error, float(np.max(np.abs(waves[0] - waves[3] - waves[4]))))
        if error > 1e-4:
            raise ValueError("Noise-free mixture is not the sum of two reverb sources")
        dst = output / "raw" / (split + "_mix_clean_reverb_min_16k")
        for name, mapping in zip(("wav.scp", "spk1.scp", "spk2.scp"), maps):
            # Names and absolute audio paths come directly from the verified dump.
            for value in mapping.values():
                if not Path(value).is_absolute():
                    raise ValueError("Source audio paths must be absolute")
            write_scp(dst / name, mapping)
            report["hashes"][str(dst / name)] = hashlib.sha256((dst / name).read_bytes()).hexdigest()
        write_scp(dst / "utt2num_samples", lengths)
        (dst / "feats_type").write_text("raw\n")
        (dst / "channel").write_text("0\n")
        if split != "tt":
            phase = "train" if split == "tr" else "valid"
            for field in ("speech_mix", "speech_ref1", "speech_ref2"):
                write_scp(stats / phase / (field + "_shape"), lengths)
            if max(int(v) for v in lengths.values()) >= 400000:
                raise ValueError("Fold length would reduce the requested batch size")
        report["splits"][split] = dict(count=count, max_samples=max(int(v) for v in lengths.values()),
                                        sampled_ids=sample_ids, mixture_sum_error=error)
        print("Verified {}: {} examples".format(split, count), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    ready.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dump_nf16k"))
    parser.add_argument("--stats", type=Path, default=Path("exp/enh_stats_16k"))
    args = parser.parse_args()
    prepare(args.source, args.output, args.stats)


if __name__ == "__main__":
    main()
