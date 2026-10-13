#!/usr/bin/env python3
"""Make a self-contained WHAMR dump with a second independent speech view.

No room simulation is performed. Physical direct/reverberant RIRs are reused
unchanged. Only the second mixture is synthesized; original indexed WAVs are
copied byte-for-byte. ``--index-root`` allows construction on local scratch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

SPLITS = ("tr", "cv", "tt")
RATE = 16000
STREAMS = (
    "speech_mix", "speech_direct1", "speech_direct2", "speech_reverb1",
    "speech_reverb2", "rir_direct1", "rir_direct2", "rir_ref1", "rir_ref2",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_index(path):
    result = {}
    with Path(path).open() as f:
        for line in f:
            key, value = line.rstrip().split(maxsplit=1)
            if key in result:
                raise ValueError("Duplicate key in {}: {}".format(path, key))
            result[key] = value
    return result


def safe_uid(uid):
    if Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Unsafe utterance ID: " + uid)
    return uid


def speaker_id(path):
    # WSJ0 speaker directories and the first three basename characters agree.
    p = Path(path)
    if p.parent.name != p.stem[:3]:
        raise ValueError("Not a WSJ0 speaker path: " + str(path))
    return p.parent.name


def split_dir(split):
    return split + "_rir_2spk_nf_min_16k"


def load_sources(source_dump, metadata_root, limit=None):
    result, dry_pools = {}, {}
    for split in SPLITS:
        raw = Path(source_dump) / "raw" / split_dir(split)
        indexes = {name: read_index(raw / (name + ".scp")) for name in STREAMS}
        all_uids = sorted(indexes["speech_mix"])
        for name, entries in indexes.items():
            if set(entries) != set(all_uids):
                raise ValueError("Mismatched keys: {} {}".format(split, name))
        lengths = read_index(raw / "utt2num_samples")
        if set(lengths) != set(all_uids):
            raise ValueError("Mismatched sample counts in " + split)
        metadata = {}
        with (Path(metadata_root) / split / "utterances.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                uid = row["utterance_id"]
                if uid in metadata:
                    raise ValueError("Duplicate source metadata: " + uid)
                metadata[uid] = row
        if not set(all_uids).issubset(metadata):
            raise ValueError("Missing source metadata: " + split)
        pool = sorted({str(Path(metadata[u][s]).resolve()) for u in all_uids
                       for s in ("source1", "source2")})
        for p in pool:
            speaker_id(p)
        dry_pools[split] = pool
        ids = all_uids if limit is None else all_uids[:limit]
        result[split] = {
            "raw": raw, "indexes": indexes, "ids": ids, "metadata": metadata,
            "lengths": {u: int(lengths[u]) for u in ids},
        }
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if set(dry_pools[a]) & set(dry_pools[b]):
                raise ValueError("Dry utterance split leakage: {} {}".format(a, b))
    return result, dry_pools


def estimate(sources):
    samples = {s: sum(sources[s]["lengths"].values()) for s in SPLITS}
    # Existing source WAVs are stereo FLOAT; retain full variable-length RIRs.
    speech = sum(samples.values()) * 8 * 5
    view2 = (samples["tr"] + samples["cv"]) * 8
    max_rir_bytes = 0
    for split in SPLITS:
        for uid in sources[split]["ids"]:
            for name in ("rir_direct1", "rir_direct2", "rir_ref1", "rir_ref2"):
                max_rir_bytes += Path(sources[split]["indexes"][name][uid]).stat().st_size
    wav_headers = sum(len(sources[s]["ids"]) for s in SPLITS) * 5 * 4096
    wav_headers += sum(len(sources[s]["ids"]) for s in ("tr", "cv")) * 4096
    total = speech + view2 + max_rir_bytes + wav_headers
    return {"counts": {s: len(sources[s]["ids"]) for s in SPLITS},
            "original_speech_payload_bytes": speech, "new_view2_payload_bytes": view2,
            "original_rir_file_bytes": max_rir_bytes,
            "estimated_total_bytes": total, "estimated_total_gib": total / 2**30,
            "recommended_free_bytes": int(total * 1.10) + 2 * 2**30}


def copy_with_hash(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    with Path(source).open("rb") as src, tmp.open("xb") as dst:
        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            dst.write(chunk)
    os.replace(tmp, target)
    return digest.hexdigest()


def dry_info(path):
    info = sf.info(path)
    if info.samplerate != RATE or info.channels != 1 or info.frames <= 0:
        raise ValueError("Expected mono 16 kHz dry speech: " + path)
    return {"path": path, "speaker": speaker_id(path), "frames": info.frames}


def select_dry(pool, original_speakers, samples, rng):
    eligible = {}
    for item in pool:
        if item["frames"] >= samples and item["speaker"] not in original_speakers:
            eligible.setdefault(item["speaker"], []).append(item)
    speakers = sorted(eligible)
    if len(speakers) < 2:
        raise ValueError("Fewer than two new speakers with {} samples".format(samples))
    chosen = rng.choice(len(speakers), 2, replace=False)
    return [eligible[speakers[i]][rng.integers(len(eligible[speakers[i]]))]
            for i in chosen]


def preflight_eligibility(entry, pool):
    longest = {}
    for item in pool:
        longest[item["speaker"]] = max(longest.get(item["speaker"], 0), item["frames"])
    for uid in entry["ids"]:
        original = {speaker_id(entry["metadata"][uid][s]) for s in ("source1", "source2")}
        candidates = [s for s, length in longest.items()
                      if length >= entry["lengths"][uid] and s not in original]
        if len(candidates) < 2:
            raise ValueError("Cannot generate {} without padding: {} new speakers have >= {} samples".format(
                uid, len(candidates), entry["lengths"][uid]))


def convolve_crop(dry, rir, start, samples):
    """Preserve history: convolve the complete original before cropping."""
    full = fftconvolve(dry[:, None], rir, mode="full", axes=0)
    crop = full[start:start + samples]
    if crop.shape != (samples, rir.shape[1]):
        raise ValueError("Insufficient convolved samples")
    return crop


def synthesize_view(dry_signals, direct_rirs, reverb_rirs, original_direct,
                    original_mix, starts, target_sir_db=None):
    """Calibrate new direct energies, then common mix level across channels."""
    samples = len(original_mix)
    reverbs, gains = [], []
    target_energies = [float(np.mean(np.square(r[:, 0], dtype=np.float64)))
                       for r in original_direct]
    if target_sir_db is not None:
        ratio = 10 ** (target_sir_db / 10)
        total = sum(target_energies)
        target_energies = [total * ratio / (1 + ratio), total / (1 + ratio)]
    for dry, direct, reverb, target, start in zip(
            dry_signals, direct_rirs, reverb_rirs, target_energies, starts):
        direct_response = convolve_crop(dry, direct, start, samples)
        energy = np.mean(np.square(direct_response[:, 0], dtype=np.float64))
        if not np.isfinite(energy) or energy <= 1e-14 or target <= 1e-14:
            raise ValueError("Silent/nonfinite source after convolution")
        gain = float(np.sqrt(target / energy))
        reverbs.append(convolve_crop(dry, reverb, start, samples) * gain)
        gains.append(gain)
    mixture = reverbs[0] + reverbs[1]
    current = np.mean(np.square(mixture[:, 0], dtype=np.float64))
    target = np.mean(np.square(original_mix[:, 0], dtype=np.float64))
    if current <= 1e-14 or target <= 1e-14:
        raise ValueError("Silent mixture")
    rms_scale = float(np.sqrt(target / current))
    mixture *= rms_scale
    peak = float(np.max(np.abs(mixture)))
    peak_scale = min(1.0, 0.99 / max(peak, 1e-12))
    mixture *= peak_scale
    if not np.isfinite(mixture).all():
        raise ValueError("Nonfinite mixture")
    return mixture.astype(np.float32), {"direct_energy_gains": gains,
                                       "target_direct_sir_db": target_sir_db,
                                       "mixture_rms_scale": rms_scale,
                                       "mixture_peak_scale": peak_scale,
                                       "source_total_gains": [g * rms_scale * peak_scale
                                                              for g in gains]}


def read_audio(path, expected_samples=None):
    audio, rate = sf.read(path, always_2d=True, dtype="float64")
    if rate != RATE or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError("Invalid stereo FLOAT audio: " + str(path))
    if expected_samples is not None and len(audio) != expected_samples:
        raise ValueError("Sample length mismatch: " + str(path))
    return audio


def record_seed(seed, split, uid):
    text = "{}:{}:{}".format(seed, split, uid).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def process_record(task):
    split, uid, entry, pool, output, seed = task
    safe_uid(uid)
    samples = entry["lengths"][uid]
    metadata = entry["metadata"][uid]
    streams = {}
    for name in STREAMS:
        source = Path(entry["indexes"][name][uid]).resolve()
        rel = Path("audio") / split / name / (uid + ".wav")
        digest = copy_with_hash(source, output / rel)
        streams[name] = {"path": str(rel), "sha256": digest,
                         "source_path": str(source)}
    original_speakers = [speaker_id(metadata[s]) for s in ("source1", "source2")]
    record = {"utterance_id": uid, "split": split, "room_id": uid,
              "num_samples": samples, "sample_rate": RATE, "channels": 2,
              "original_speakers": original_speakers,
              "original_metadata": metadata, "streams": streams}
    if split == "tt":
        return record
    rng_seed = record_seed(seed, split, uid)
    rng = np.random.default_rng(rng_seed)
    selected = select_dry(pool, original_speakers, samples, rng)
    direct = [read_audio(output / streams[n]["path"])
              for n in ("rir_direct1", "rir_direct2")]
    reverb = [read_audio(output / streams[n]["path"])
              for n in ("rir_ref1", "rir_ref2")]
    original = [read_audio(output / streams[n]["path"], samples)
                for n in ("speech_direct1", "speech_direct2")]
    original_mix = read_audio(output / streams["speech_mix"]["path"], samples)
    signals, starts, provenance = [], [], []
    for item in selected:
        dry, rate = sf.read(item["path"], dtype="float64")
        if rate != RATE or dry.ndim != 1 or not np.isfinite(dry).all():
            raise ValueError("Invalid dry audio: " + item["path"])
        start = int(rng.integers(0, len(dry) - samples + 1))
        signals.append(dry)
        starts.append(start)
        provenance.append(dict(item, sha256=sha256(item["path"]), crop_start=start,
                               crop_samples=samples, padding_samples=0))
    target_sir_db = float(rng.uniform(-5.0, 5.0))
    mix, gains = synthesize_view(signals, direct, reverb, original, original_mix, starts,
                                 target_sir_db=target_sir_db)
    rel = Path("audio") / split / "speech_mix_view2" / (uid + ".wav")
    path = output / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    sf.write(tmp, mix, RATE, subtype="FLOAT", format="WAV")
    os.replace(tmp, path)
    streams["speech_mix_view2"] = {"path": str(rel), "sha256": sha256(path)}
    record["view2"] = {"seed": rng_seed, "sources": provenance, "gains": gains,
                       "noise": False, "peak": float(np.max(np.abs(mix))),
                       "position_order": ["rir_ref1", "rir_ref2"]}
    return record


def write_indexes(output, index_root, sources, manifests):
    hashes = {}
    for split in SPLITS:
        raw = output / "raw" / split_dir(split)
        raw.mkdir(parents=True)
        records = manifests[split]
        for name in STREAMS + (("speech_mix_view2",) if split != "tt" else ()):
            lines = [r["utterance_id"] + " " + str(index_root / r["streams"][name]["path"])
                     for r in records]
            (raw / (name + ".scp")).write_text("\n".join(lines) + "\n")
        shutil.copyfile(raw / "speech_mix.scp", raw / "wav.scp")
        (raw / "utt2num_samples").write_text("".join(
            "{} {}\n".format(r["utterance_id"], r["num_samples"]) for r in records))
        for name in ("utt2spk", "spk2utt"):
            # Enhancement recipes use utterance IDs as pseudo speaker IDs.
            (raw / name).write_text("".join("{} {}\n".format(r["utterance_id"], r["utterance_id"])
                                           for r in records))
        for name in ("channel", "feats_type"):
            src = sources[split]["raw"] / name
            if src.exists():
                shutil.copyfile(src, raw / name)
        for path in raw.iterdir():
            hashes[str(path.relative_to(output))] = sha256(path)
    return hashes


def verify_dump(output, check_hashes=True):
    output = Path(output).resolve()
    preparation = json.loads((output / "preparation.json").read_text())
    index_root = Path(preparation["index_root"])
    dry_pools = json.loads((output / "manifests/dry_pools.json").read_text())
    pool_sets = {s: set(p) for s, p in dry_pools.items()}
    counts, seen_dry = {}, {}
    for split in SPLITS:
        raw = output / "raw" / split_dir(split)
        indexes = {p.stem: read_index(p) for p in raw.glob("*.scp")}
        lengths = {u: int(n) for u, n in read_index(raw / "utt2num_samples").items()}
        ids = set(lengths)
        expected = set(STREAMS) | {"wav"}
        if split != "tt":
            expected.add("speech_mix_view2")
        if set(indexes) != expected or any(set(i) != ids for i in indexes.values()):
            raise ValueError("Index topology mismatch in " + split)
        with (output / "manifests" / (split + ".jsonl")).open() as manifest:
            records = [json.loads(l) for l in manifest]
        if len(records) != len(ids) or {r["utterance_id"] for r in records} != ids:
            raise ValueError("Manifest key mismatch in " + split)
        seen_dry[split] = set()
        for r in records:
            uid = r["utterance_id"]
            for name, item in r["streams"].items():
                rel = Path(item["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Nonlocal manifest path")
                absolute = Path(indexes[name][uid])
                if absolute != index_root / rel:
                    raise ValueError("Index is not internal: " + str(absolute))
                path = output / rel
                if path.is_symlink():
                    raise ValueError("Audio must be a real copied file: " + str(path))
                info = sf.info(path)
                if info.subtype != "FLOAT":
                    raise ValueError("Expected FLOAT WAV: " + str(path))
                audio = read_audio(path, lengths[uid] if name.startswith("speech_") else None)
                if name == "speech_mix_view2" and np.max(np.abs(audio)) > 0.990001:
                    raise ValueError("View2 exceeds its common peak limit")
                if check_hashes and sha256(path) != item["sha256"]:
                    raise ValueError("Audio checksum mismatch: " + str(path))
            if indexes["wav"][uid] != indexes["speech_mix"][uid]:
                raise ValueError("wav and mixture differ")
            if split != "tt":
                view = r["view2"]
                speakers = [s["speaker"] for s in view["sources"]]
                if len(set(speakers)) != 2 or set(speakers) & set(r["original_speakers"]):
                    raise ValueError("View2 speaker exclusion failed: " + uid)
                if view["position_order"] != ["rir_ref1", "rir_ref2"] or view["noise"]:
                    raise ValueError("Invalid view2 physical condition")
                for source in view["sources"]:
                    if source["path"] not in pool_sets[split]:
                        raise ValueError("Dry source outside its original split")
                    if speaker_id(source["path"]) != source["speaker"]:
                        raise ValueError("Dry speaker provenance mismatch")
                    seen_dry[split].add(source["path"])
                    if source["padding_samples"] != 0 or source["crop_samples"] != lengths[uid]:
                        raise ValueError("Unexpected crop/padding")
        counts[split] = len(records)
        print(json.dumps({"verified_split": split, "count": len(records)}), flush=True)
    if seen_dry["tr"] & seen_dry["cv"]:
        raise ValueError("Train/validation dry overlap")
    for rel, digest in preparation["output_hashes"].items():
        if sha256(output / rel) != digest:
            raise ValueError("Index checksum mismatch: " + rel)
    if counts != preparation["counts"]:
        raise ValueError("Count mismatch")
    return {"status": "complete", "counts": counts, "checked_audio_hashes": check_hashes,
            "self_contained": True, "sample_rate": RATE, "channels": 2,
            "verified_at_unix": time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dump", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.verify_existing:
        result = verify_dump(args.verify_existing)
        atomic_json(args.verify_existing / "verification.json", result)
        print(json.dumps(result), flush=True)
        return
    if args.source_dump is None or args.metadata_root is None:
        parser.error("--source-dump and --metadata-root are required")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    sources, dry_pools = load_sources(args.source_dump, args.metadata_root, args.limit)
    capacity = estimate(sources)
    print(json.dumps({"estimate": capacity}), flush=True)
    if args.estimate_only:
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.absolute()
    index_root = (args.index_root or output).absolute()
    if output.exists():
        raise FileExistsError("Will not overwrite output: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < capacity["recommended_free_bytes"]:
        raise OSError("Insufficient free space for a self-contained dump")
    output.mkdir()
    (output / "manifests").mkdir()
    pools = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split in ("tr", "cv"):
            pools[split] = list(executor.map(dry_info, dry_pools[split]))
            preflight_eligibility(sources[split], pools[split])
            print(json.dumps({"loaded_dry_pool": split, "count": len(pools[split])}), flush=True)
        manifests = {}
        for split in SPLITS:
            tasks = ((split, uid, sources[split], pools.get(split), output, args.seed)
                     for uid in sources[split]["ids"])
            records = []
            with (output / "manifests" / (split + ".jsonl")).open("x") as f:
                for record in executor.map(process_record, tasks):
                    records.append(record)
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                    if len(records) % 100 == 0:
                        f.flush()
                        print(json.dumps({"generated_split": split, "count": len(records)}), flush=True)
            manifests[split] = records
    hashes = write_indexes(output, index_root, sources, manifests)
    atomic_json(output / "manifests/dry_pools.json", dry_pools)
    hashes["manifests/dry_pools.json"] = sha256(output / "manifests/dry_pools.json")
    for split in SPLITS:
        hashes["manifests/" + split + ".jsonl"] = sha256(output / "manifests" / (split + ".jsonl"))
    preparation = {
        "format_version": 1, "spec": {"sample_rate": "16k", "num_spk": 2,
                                        "length": "min", "noise": False},
        "counts": capacity["counts"], "output_hashes": hashes,
        "source_dump": str(args.source_dump.resolve()),
        "metadata_root": str(args.metadata_root.resolve()), "index_root": str(index_root),
        "seed": args.seed, "workers": args.workers, "limit": args.limit,
        "generator_sha256": sha256(__file__), "capacity": capacity,
        "view2_splits": ["tr", "cv"], "dry_pool_counts": {s: len(v) for s, v in dry_pools.items()},
        "policy": "Same physical stereo RIRs; disjoint new speakers; split-local dry speech; "
                  "full convolution before crop; no padding; independently sampled direct SIR U[-5,5] dB; "
                  "common stereo gain; no noise; one fixed new view per training/validation item",
    }
    atomic_json(output / "preparation.json", preparation)
    (output / "README.txt").write_text(
        "Self-contained two-view WHAMR 16k/min/noise-free dump.\n"
        "raw/: final-path indexes. audio/: copied originals and only the new view2 mixtures.\n"
        "manifests/: source provenance, checksums, speaker identities, seeds, crops and gains.\n"
        "Training and validation each have view2. Test contains unchanged original audio only.\n"
        "Original/dry source paths in provenance are historical, never runtime dependencies.\n"
        "Teacher RIRs and source ordering are shared across views. Channel 0 is selected by training.\n"
        "room_id identifies a fixed RIR pair, not a position-independent physical room class.\n")
    result = verify_dump(output)
    atomic_json(output / "verification.json", result)
    atomic_json(output / "completion.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
