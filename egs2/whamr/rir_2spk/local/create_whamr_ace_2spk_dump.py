#!/usr/bin/env python3
"""Create 16-kHz two-speaker ACE evaluation mixtures from WHAMR tt metadata."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rir_1ch/local"))
from create_whamr_ace_dump import ACE_MD5, digest, load_rirs, quantize, read_audio, write_audio


def read_scp(path):
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs: {path}")
    return result


def generate(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    with args.filenames.open() as stream:
        rows = list(csv.DictReader(stream))
    by_name = {row["output_filename"]: row for row in rows}
    scaling = np.load(args.scaling, allow_pickle=True)
    filenames = [str(value) for value in scaling["utterance_id"]]
    if len(set(filenames)) != len(filenames) or set(filenames) != set(by_name):
        raise ValueError("tt CSV and scaling IDs differ")

    output.mkdir(parents=True)
    rirs = load_rirs(args.ace_archive, output)
    rooms = defaultdict(list)
    for rir in rirs:
        rooms[str(Path(rir[3]).parent.parent)].append(rir)
    room_pairs = []
    for room, entries in sorted(rooms.items()):
        if len(entries) != 2:
            raise ValueError(f"Expected two ACE positions in {room}, got {len(entries)}")
        room_pairs.append(tuple(sorted(entries)))
    if len(room_pairs) != 7:
        raise ValueError(f"Expected seven ACE room pairs, got {len(room_pairs)}")

    assignment = np.arange(len(filenames)) % len(room_pairs)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(assignment)
    swap = rng.integers(0, 2, size=len(filenames), dtype=np.int8)
    cached = {}
    rir_metadata = []
    for pair in room_pairs:
        for name, samples, sample_rate, member in pair:
            rir = resample_poly(samples, 1, 3)
            divisor = float(np.max(np.abs(rir)))
            if divisor <= 0:
                raise ValueError(f"Silent RIR: {name}")
            rir = (rir / divisor).astype(np.float32).astype(np.float64)
            rir_path = write_audio(output / "audio/16000/rir" / name, rir, 16000)
            cached[name] = (rir, rir_path)
            rir_metadata.append(dict(ace_member=member, room=str(Path(member).parent.parent),
                                     path=rir_path, samples=len(rir),
                                     normalization_divisor=divisor, sha256=digest(rir_path)))

    config = dict(generator_sha256=digest(Path(__file__)), seed=args.seed,
                  split="tt", length_mode="min", sample_rate=16000, limit=args.limit,
                  ace_archive=str(args.ace_archive.resolve()), ace_md5=ACE_MD5,
                  filenames=str(args.filenames.resolve()), scaling=str(args.scaling.resolve()),
                  wsj_root=str(args.wsj_root.resolve()), noise_root=str(args.noise_root.resolve()),
                  assignment="balanced seeded room assignment; two measured positions in the same ACE room; seeded source-position swap",
                  source_gain="per-source WHAMR wsjmix gain followed by int16 quantization and shared WHAM speech gain",
                  conditions=["clean", "noisy"], rir="full peak-normalized measured RIR; acquisition delay retained",
                  tail="convolution cropped to WHAMR min mixture length", output_subtype="FLOAT")
    (output / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "rir_metadata.json").write_text(json.dumps(rir_metadata, indent=2) + "\n")

    manifests = defaultdict(lambda: defaultdict(list))
    records = []
    limit = len(filenames) if args.limit is None else min(args.limit, len(filenames))
    for index, filename in enumerate(filenames[:limit]):
        row = by_name[filename]
        uid = Path(filename).stem
        sources = [read_audio(args.wsj_root / row[f"s{i}_path"], 16000) for i in (1, 2)]
        length = min(map(len, sources))
        gains = scaling["scaling_wsjmix_16k_min"][index]
        speech_gain = float(scaling["scaling_wham_speech_16k_min"][index])
        sources = [quantize(source[:length] * float(gain)) * speech_gain
                   for source, gain in zip(sources, gains)]
        pair = list(room_pairs[int(assignment[index])])
        if swap[index]:
            pair.reverse()
        rir_names = [entry[0] for entry in pair]
        reverbs = [fftconvolve(source, cached[name][0])[:length]
                   for source, name in zip(sources, rir_names)]
        clean = reverbs[0] + reverbs[1]
        noise_gain = float(scaling["scaling_wham_noise_16k_min"][index])
        noise = read_audio(args.noise_root / "tt" / filename, 16000) * noise_gain
        start = int(scaling["speech_start_sample_16k"][index])
        noise = noise[start:start + length]
        if len(noise) != length:
            raise ValueError(f"Noise too short: {filename}")
        noisy = clean + noise
        joint_gain = min(1.0, 0.99 / max(max(np.max(np.abs(x)) for x in
                                                (*sources, *reverbs, noise, noisy)), 1e-12))
        paths = {}
        for condition, mixture in (("clean", clean), ("noisy", noisy)):
            paths[condition] = write_audio(
                output / f"audio/16000/mix_{condition}" / filename,
                mixture * joint_gain, 16000)
            setname = f"tt_ace_2spk_{condition}_reverb_min_16k"
            manifest = manifests[setname]
            for key in ("wav.scp", "speech_mix.scp"):
                manifest[key].append((uid, paths[condition]))
            for source_index, name in enumerate(rir_names, 1):
                manifest[f"rir_ref{source_index}.scp"].append((uid, cached[name][1]))
            manifest["utt2num_samples"].append((uid, str(length)))
        records.append(dict(utterance_id=uid, room=str(Path(pair[0][3]).parent.parent),
                            ace_rir1=rir_names[0], ace_rir2=rir_names[1], samples=length,
                            source_gains=[float(x) for x in gains], speech_gain=speech_gain,
                            noise_gain=noise_gain, noise_start_sample=start,
                            joint_gain=joint_gain, paths=paths))
        if (index + 1) % 100 == 0 or index + 1 == limit:
            print(f"Generated {index + 1}/{limit}", flush=True)

    for setname, entries in manifests.items():
        folder = output / "raw" / setname
        folder.mkdir(parents=True)
        for name, values in entries.items():
            (folder / name).write_text("".join(f"{key} {value}\n" for key, value in sorted(values)))
        (folder / "feats_type").write_text("raw\n")
    with (output / "utterances.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    (output / "generation_complete.json").write_text(json.dumps(
        dict(utterances_per_set=limit, sets=sorted(manifests), room_pairs=len(room_pairs)), indent=2) + "\n")


def validate(output):
    complete = json.loads((output / "generation_complete.json").read_text())
    expected = int(complete["utterances_per_set"])
    for setname in complete["sets"]:
        folder = output / "raw" / setname
        maps = [read_scp(folder / name) for name in
                ("wav.scp", "speech_mix.scp", "rir_ref1.scp", "rir_ref2.scp", "utt2num_samples")]
        if any(len(values) != expected or set(values) != set(maps[0]) for values in maps):
            raise ValueError(f"Manifest mismatch: {setname}")
        for uid, wav in maps[0].items():
            info = sf.info(wav)
            if info.samplerate != 16000 or info.frames != int(maps[-1][uid]):
                raise ValueError(f"Audio mismatch: {uid}")
            if maps[2][uid] == maps[3][uid]:
                raise ValueError(f"Sources use the same RIR: {uid}")
    print(f"Validation complete: {output}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ace-archive", type=Path, required=True)
    parser.add_argument("--filenames", type=Path, required=True)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--wsj-root", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(args.output.resolve())
    else:
        generate(args)
        validate(args.output.resolve())


if __name__ == "__main__":
    main()
