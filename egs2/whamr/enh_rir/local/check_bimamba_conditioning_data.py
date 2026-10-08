"""Validate the shared native-16k NF dump without copying audio or changing SCPs."""

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
            raise ValueError(f"Duplicate ID or unsupported pipe in {path}")
        values[uid] = value
    return values


def check_dump(dump, output):
    if not dump.is_symlink() or not dump.is_dir():
        raise ValueError("Use the symbolic link to rir_2spk's native 16-kHz dump")
    report = dict(source=str(dump.resolve()), sample_rate=16000, channel=0, splits={})
    for split, expected_count in (("tr", 20000), ("cv", 5000), ("tt", 3000)):
        folder = dump / "raw" / f"{split}_rir_2spk_nf_min_16k"
        names = ("wav.scp", "speech_direct1.scp", "speech_direct2.scp",
                 "speech_reverb1.scp", "speech_reverb2.scp")
        maps = [read_scp(folder / name) for name in names]
        lengths = read_scp(folder / "utt2num_samples")
        ids = sorted(maps[0])
        if len(ids) != expected_count or any(set(m) != set(ids) for m in maps[1:] + [lengths]):
            raise ValueError(f"Incomplete or mismatched {split} keys")
        if min(int(n) for n in lengths.values()) <= 256:
            raise ValueError("Observation too short for the pretrained BiMamba STFT")
        if max(int(n) for n in lengths.values()) >= 400000:
            raise ValueError("Fold length would change the requested physical batch")
        for uid in ids:
            for mapping in maps:
                path = Path(mapping[uid])
                if not path.is_absolute() or not path.is_file():
                    raise ValueError(f"Missing or non-absolute audio path: {path}")
        sampled = ids[::max(1, len(ids) // 8)][:8]
        worst_error = 0.0
        for uid in sampled:
            waves = []
            for mapping in maps:
                wave, sr = sf.read(mapping[uid], dtype="float32", always_2d=True)
                if sr != 16000 or wave.shape[1] != 2 or len(wave) != int(lengths[uid]):
                    raise ValueError(f"Wrong native-16k header or length: {uid}")
                if not np.isfinite(wave).all():
                    raise ValueError(f"Nonfinite waveform: {uid}")
                waves.append(wave[:, 0])
            worst_error = max(worst_error, float(np.max(np.abs(waves[0] - waves[3] - waves[4]))))
        if worst_error > 1e-4:
            raise ValueError(f"Mixture is not noise-free: {split}")
        report["splits"][split] = dict(
            count=len(ids), sampled_ids=sampled, mixture_sum_error=worst_error,
            hashes={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in names + ("utt2num_samples",)},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({split: value["count"] for split, value in report["splits"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_dump(args.dump, args.output)
