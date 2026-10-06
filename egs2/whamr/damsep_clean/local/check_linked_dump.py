"""Validate existing NF-WHAMR and source-image dumps without rewriting SCPs."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def read_scp(path):
    entries = {}
    for line in path.read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in entries or value.endswith("|"):
            raise ValueError(f"Duplicate ID or unsupported audio pipe: {path}")
        # Resolve just as the ESPnet sound loader does, relative to recipe CWD.
        entries[uid] = Path(value)
    return entries


def check(baseline, reverb, workers=4, waveform_checks=8, expected_counts=None):
    report = dict(
        sample_rate=8000,
        channel=0,
        noise=False,
        baseline_dump=str(baseline.resolve()),
        reverb_dump=str(reverb.resolve()),
        splits={},
        source_hashes={},
    )
    for split in ("tr", "cv", "tt"):
        base = baseline / "raw" / f"{split}_mix_clean_reverb_min_8k"
        rev = reverb / "raw" / f"{split}_mix_both_reverb_min_8k"
        paths = {
            "wav": base / "wav.scp",
            "spk1": base / "spk1.scp",
            "spk2": base / "spk2.scp",
            "spk1_reverb": rev / "spk1_reverb.scp",
            "spk2_reverb": rev / "spk2_reverb.scp",
        }
        maps = {name: read_scp(path) for name, path in paths.items()}
        ids = sorted(maps["wav"])
        if not ids or any(set(m) != set(ids) for m in maps.values()):
            raise ValueError(f"Mixture/source ID mismatch: {split}")
        if expected_counts is not None and len(ids) != expected_counts[split]:
            raise ValueError(f"Unexpected {split} count: {len(ids)}")
        for path in paths.values():
            report["source_hashes"][str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

        def inspect(uid):
            headers = [sf.info(values[uid]) for values in maps.values()]
            if any(h.samplerate != 8000 or h.channels != 2 for h in headers):
                raise ValueError(f"{uid}: expected stereo 8 kHz baseline/source audio")
            lengths = {h.frames for h in headers}
            if len(lengths) != 1 or min(lengths) <= 256:
                raise ValueError(f"{uid}: unaligned or too short audio: {lengths}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(inspect, ids))
        checks = (
            ids[:: max(1, len(ids) // waveform_checks)][:waveform_checks]
            if waveform_checks
            else []
        )
        worst = 0.0
        for uid in checks:
            audio = {
                k: sf.read(values[uid], dtype="float32", always_2d=True)[0][:, 0]
                for k, values in maps.items()
            }
            if any(not np.isfinite(x).all() for x in audio.values()):
                raise ValueError(f"{uid}: nonfinite audio")
            error = float(
                np.max(
                    np.abs(audio["wav"] - audio["spk1_reverb"] - audio["spk2_reverb"])
                )
            )
            if error > 4e-5:
                raise ValueError(
                    f"{uid}: mixture is not the sum of reverb sources ({error})"
                )
            worst = max(worst, error)
        report["splits"][split] = dict(
            count=len(ids),
            all_headers_checked=True,
            waveform_ids=checks,
            max_sum_error=worst,
        )
        print(f"{split}: {len(ids)} aligned existing examples", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dump", type=Path, default=Path("dump_clean"))
    parser.add_argument("--reverb-dump", type=Path, default=Path("dump_reverb"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--waveform-checks", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.waveform_checks < 0:
        parser.error("workers must be positive and waveform-checks nonnegative")
    report = check(
        args.baseline_dump,
        args.reverb_dump,
        args.workers,
        args.waveform_checks,
        dict(tr=20000, cv=5000, tt=3000),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
