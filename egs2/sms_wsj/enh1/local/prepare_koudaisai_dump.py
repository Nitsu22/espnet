#!/usr/bin/env python3
"""Build an explicit rsync manifest or verify a private relocated dump.

No file under the source dump is ever written. Audio is copied byte-for-byte;
microphone 0 selection happens in the training preprocessor.
"""

import argparse
import json
from pathlib import Path

SPLITS = ("train_si284_4mic", "cv_dev93_4mic", "test_eval92_4mic")


def prepare(source, target, work):
    work.mkdir(parents=True, exist_ok=True)
    sizes, counts = {}, {}
    for split in SPLITS:
        out = work / "metadata" / "raw" / split
        out.mkdir(parents=True, exist_ok=True)
        expected = None
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            rows, keys = [], []
            for line in (source / "raw" / split / name).read_text().splitlines():
                key, filename = line.split(maxsplit=1)
                if filename.startswith("dump_4mic/"):
                    audio = source / filename.removeprefix("dump_4mic/")
                else:
                    audio = Path(filename)
                rel = audio.resolve().relative_to(source.resolve())
                sizes[str(rel)] = audio.stat().st_size
                rows.append(f"{key} {target / rel}\n")
                keys.append(key)
            assert len(keys) == len(set(keys)), (split, name, "duplicate IDs")
            if expected is None:
                expected = keys
            assert keys == expected, (split, name, "unaligned IDs")
            (out / name).write_text("".join(rows))
        counts[split] = len(expected)
        for name in ("utt2spk", "spk2utt", "utt2num_samples", "utt2dur", "feats_type"):
            src = source / "raw" / split / name
            if src.exists():
                (out / name).write_bytes(src.read_bytes())
        # Required by enh.sh; reference/mix are still multichannel on disk.
        (out / "feats_type").write_text("raw\n")
    info = {"source": str(source), "target": str(target), "counts": counts,
            "files": sizes, "total_bytes": sum(sizes.values())}
    (work / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    (work / "audio_files.txt").write_text("\n".join(sorted(sizes)) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2))
    print(f"Audio files: {len(sizes)}")


def verify(target):
    info = json.loads((target / "manifest.json").read_text())
    assert Path(info["target"]) == target
    for rel, size in info["files"].items():
        audio = target / rel
        assert audio.is_file() and not audio.is_symlink(), audio
        assert audio.stat().st_size == size, audio
    for split, count in info["counts"].items():
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            lines = (target / "raw" / split / name).read_text().splitlines()
            assert len(lines) == count
            for line in lines:
                filename = Path(line.split(maxsplit=1)[1])
                filename.relative_to(target)
                assert filename.is_file(), filename
    # Written only after the caller's rsync --checksum audit has succeeded.
    assert (target / "rsync_checksum_verified").is_file()
    result = {"counts": info["counts"], "audio_files": len(info["files"]),
              "total_bytes": info["total_bytes"], "target": str(target)}
    (target / "COPY_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        assert args.source and args.work and args.target.is_absolute()
        prepare(args.source, args.target, args.work)
    else:
        verify(args.target)
