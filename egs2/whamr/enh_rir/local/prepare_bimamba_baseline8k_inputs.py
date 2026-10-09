"""Resolve enh7_baseline's original 8-kHz SCPs without copying/changing WAVs.

Original SCP audio paths are recipe-relative. A dump symlink alone cannot
change that base directory. Write only small absolute-path indexes under exp;
retain the Baseline's exact utterance IDs, audio and teacher files.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import soundfile as sf

from check_bimamba_conditioning_data import read_scp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dump, baseline_recipe, output, workers=4,
            expected_counts=(("tr", 20000), ("cv", 5000), ("tt", 3000))):
    if not dump.is_symlink() or not dump.is_dir():
        raise ValueError("Use a symlink to enh7_baseline's NF-WHAMR dump")
    spec = dict(source=str(dump.resolve()), base_directory=str(baseline_recipe.resolve()),
                sample_rate=8000, channel=0, source_hashes={})
    for split, _ in expected_counts:
        folder = dump / "raw" / f"{split}_mix_clean_reverb_min_8k"
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            spec["source_hashes"][f"{split}/{name}"] = sha(folder / name)
    marker = output / "preparation.json"
    if output.exists():
        if not marker.is_file():
            raise ValueError("Existing input directory is incomplete; use a new output")
        previous = json.loads(marker.read_text())
        if previous["spec"] != spec:
            raise ValueError("Baseline source changed; use a new experiment/input directory")
        for name, digest in previous["output_hashes"].items():
            if sha(output / name) != digest:
                raise ValueError(f"Prepared input index changed: {name}")
        print("Existing Baseline input indexes verified", flush=True)
        return previous

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".bimamba_nf8k_", dir=output.parent))
    try:
        counts = {}
        for split, expected_count in expected_counts:
            dataset = f"{split}_mix_clean_reverb_min_8k"
            src = dump / "raw" / dataset
            maps = [read_scp(src / name) for name in ("wav.scp", "spk1.scp", "spk2.scp")]
            ids = sorted(maps[0])
            if len(ids) != expected_count or any(set(m) != set(ids) for m in maps[1:]):
                raise ValueError(f"Incomplete or mismatched Baseline {split} keys")
            resolved = []
            for values in maps:
                resolved.append({uid: str((Path(path) if Path(path).is_absolute()
                                          else baseline_recipe / path).resolve(strict=True))
                                 for uid, path in values.items()})

            def inspect(uid):
                headers = [sf.info(values[uid]) for values in resolved]
                length = headers[0].frames
                if any(h.samplerate != 8000 or h.channels not in (1, 2)
                       or h.frames != length for h in headers):
                    raise ValueError(f"Wrong sample rate or unaligned Baseline teachers: {uid}")
                if length <= 128 or length >= 200000:
                    raise ValueError(f"Invalid length/fold threshold for Baseline: {uid}")
                return length

            with ThreadPoolExecutor(max_workers=workers) as executor:
                lengths = list(executor.map(inspect, ids))
            dest = temp / "raw" / dataset
            dest.mkdir(parents=True)
            for name, values in zip(("wav.scp", "spk1.scp", "spk2.scp"), resolved):
                (dest / name).write_text("".join(f"{uid} {values[uid]}\n" for uid in ids))
            (dest / "utt2num_samples").write_text("".join(
                f"{uid} {length}\n" for uid, length in zip(ids, lengths)))
            counts[split] = len(ids)
            print(f"Verified exact Baseline {split}: {len(ids)} examples", flush=True)
        hashes = {str(p.relative_to(temp)): sha(p) for p in temp.rglob("*") if p.is_file()}
        report = dict(spec=spec, counts=counts, output_hashes=hashes)
        (temp / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        temp.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temp)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--baseline-recipe", type=Path, default=Path("../enh7_baseline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.dump, args.baseline_recipe, args.output, args.workers)
