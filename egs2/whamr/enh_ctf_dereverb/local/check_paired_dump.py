#!/usr/bin/env python3
"""Validate an existing preparation manifest/indexes without rereading WAV audio."""

import argparse
import hashlib
import json
from pathlib import Path


def read_index(path):
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        if key in values:
            raise ValueError(f"Duplicate utterance in {path}: {key}")
        values[key] = value
    return values


def check(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["sample_rate"] != 8000 or manifest["microphone"] != "left":
        raise ValueError("Expected paired, left-channel 8-kHz WHAMR data")
    if manifest["input"] != "baseline mix_both_reverb":
        raise ValueError("Expected noisy mix_both_reverb; do not substitute NF-WHAMR")
    hashes = manifest["index_sha256"]
    for name, digest in hashes.items():
        path = root / name
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"Manifest index leaves the prepared dump: {name}") from error
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared index changed: {path}")
    for split, shape_split, expected in (("tr", "train", 20000),
                                         ("cv", "valid", 5000),
                                         ("tt", None, 3000)):
        folder = root / "raw" / f"{split}_mix_both_reverb_min_8k"
        indexes = []
        for name in ("wav.scp", "spk1.scp", "spk2.scp",
                     "spk1_reverb.scp", "spk2_reverb.scp"):
            relative = str((folder / name).relative_to(root))
            if relative not in hashes:
                raise ValueError(f"Index was not verified by preparation: {relative}")
            index = read_index(folder / name)
            if len(index) != expected:
                raise ValueError(f"Expected {expected} utterances: {folder / name}")
            for value in index.values():
                if not Path(value).is_absolute() or value.endswith("|"):
                    raise ValueError(f"Expected absolute WAV paths: {folder / name}")
            indexes.append(index)
        keys = set(indexes[0])
        if any(set(index) != keys for index in indexes):
            raise ValueError(f"Mismatched paired utterance IDs: {folder}")
        # An inexpensive host check; the original adapter verified every WAV.
        for index in indexes:
            first = next(iter(index.values()))
            if not Path(first).is_file():
                raise FileNotFoundError(f"Prepared for another host or missing WAV: {first}")
        if manifest["splits"][split]["count"] != expected:
            raise ValueError(f"Preparation count differs: {split}")
        if shape_split:
            lengths = []
            for name in ("speech_mix", "speech_ref1", "speech_ref2"):
                path = root / "stats" / shape_split / f"{name}_shape"
                shapes = read_index(path)
                if set(shapes) != keys:
                    raise ValueError(f"Shape utterance IDs differ: {path}")
                current = {key: int(value.split(",")[0])
                           for key, value in shapes.items()}
                if any(n <= 128 or n >= 200000 for n in current.values()):
                    raise ValueError(f"Invalid length/folding threshold: {path}")
                lengths.append(current)
            if any(length != lengths[0] for length in lengths[1:]):
                raise ValueError(f"Unaligned mixture/reference shapes: {shape_split}")
    print("Verified paired manifest, index hashes, split IDs and batching shapes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    check(parser.parse_args().dump)
