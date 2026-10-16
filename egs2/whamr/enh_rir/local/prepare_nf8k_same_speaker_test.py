#!/usr/bin/env python3
"""Build a test-only NF-WHAMR control with two utterances of one speaker.

Scene keys describe the ORIGINAL room/source geometry, not the new identities.
Read audit.tsv for actual identities. Both models must use this same new input;
the conditioning model must estimate its CTF from the NEW mixture, not reuse an
original/oracle CTF. No train/valid examples or estimated CTFs are produced here.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import __version__ as scipy_version
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf


DATASET = "tt_mix_clean_reverb_min_8k"
NAMES = ("wav.scp", "spk1.scp", "spk2.scp")
ALLOWED_ROOTS = (Path("/net/midgar/work2/nitsu"), Path("/gs/bs/tga-shinoda/nitsu"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scp(path):
    result = {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result or fields[1].endswith("|"):
            raise ValueError(f"Invalid, duplicate or pipeline SCP entry: {path}:{lineno}")
        result[fields[0]] = fields[1]
    if not result:
        raise ValueError(f"Empty SCP: {path}")
    return result


def canonical_filename(uid):
    fields = uid.split("_")
    if (len(fields) != 7 or fields[-1] != "reverb"
            or fields[0] != fields[2][:3] or fields[1] != fields[4][:3]
            or "/" in uid or "\\" in uid):
        raise ValueError(f"Invalid original scene UID: {uid}")
    return "_".join(fields[2:-1]) + ".wav"


def read_filelist(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        name = row["output_filename"]
        if Path(name).name != name or name in result:
            raise ValueError(f"Duplicate or unsafe mixture filename: {path}: {name}")
        for k in (1, 2):
            p = Path(row[f"s{k}_path"])
            if p.is_absolute() or ".." in p.parts or p.stem[:3] != p.parent.name:
                raise ValueError(f"Unsafe or inconsistent raw speech path: {p}")
        result[name] = row
    if not result:
        raise ValueError(f"Empty filelist: {path}")
    return result


def check_split_disjoint(test, other):
    def identities(rows):
        files = {Path(r[f"s{k}_path"]).stem for r in rows.values() for k in (1, 2)}
        speakers = {Path(r[f"s{k}_path"]).parent.name for r in rows.values() for k in (1, 2)}
        return files, speakers

    test_files, test_speakers = identities(test)
    other_files, other_speakers = identities(other)
    if test_files & other_files or test_speakers & other_speakers:
        raise ValueError("Test raw utterances/speakers overlap train or validation")


def balanced_anchors(feasible, rng):
    """Assign half the scenes to each anchor position, respecting feasibility."""
    feasible = np.asarray(feasible, dtype=bool)
    if feasible.ndim != 2 or feasible.shape[1] != 2 or np.any(~feasible.any(axis=1)):
        raise ValueError("A scene has no distinct, sufficiently long same-speaker donor")
    target = len(feasible) // 2
    forced1 = np.flatnonzero(feasible[:, 0] & ~feasible[:, 1])
    forced2 = np.flatnonzero(~feasible[:, 0] & feasible[:, 1])
    flexible = np.flatnonzero(feasible.all(axis=1))
    remaining1 = target - len(forced1)
    if remaining1 < 0 or remaining1 > len(flexible):
        raise ValueError("Feasibility does not permit balanced anchor positions")
    out = np.full(len(feasible), 2, dtype=int)
    out[forced1] = 1
    out[rng.permutation(flexible)[:remaining1]] = 1
    return out


def read_left(path, rate):
    audio, actual_rate = sf.read(path, dtype="float64", always_2d=True)
    if actual_rate != rate or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite audio or wrong channels/sample rate: {path}")
    return audio[:, 0]


def convolve_8k(raw16, h16, length):
    if len(raw16) < 2 * length - 1:
        raise ValueError("Raw speech shorter than original mixture; no padding/repetition allowed")
    return resample_poly(fftconvolve(raw16, h16), 1, 2)[:length]


def match_power(direct, reverb, target):
    power = float(np.mean(direct ** 2))
    if not np.isfinite(power) or power <= 0 or not np.isfinite(target) or target <= 0:
        raise ValueError("Cannot normalize zero/nonfinite source power")
    gain = float(np.sqrt(target / power))
    return direct * gain, reverb * gain, gain


def control_error(predicted, path):
    target = read_left(path, 8000)
    if len(predicted) != len(target):
        raise ValueError(f"Identity control length mismatch: {path}")
    if sf.info(path).subtype == "PCM_16":
        predicted = np.clip(predicted, -1.0, 1.0)
    return float(np.max(np.abs(predicted - target)))


def write_pcm(path, audio):
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(audio).all() or peak >= 1:
        raise ValueError(f"Refusing PCM16 clipping/nonfinite samples: {path}; peak={peak}")
    sf.write(path, audio, 8000, subtype="PCM_16")
    return peak


def apply_peak_policy(audios, policy):
    """Optionally attenuate the entire example, never individual sources."""
    if policy not in ("error", "common-attenuate"):
        raise ValueError(f"Unknown peak policy: {policy}")
    if any(not np.isfinite(audio).all() for audio in audios):
        raise ValueError("Nonfinite samples before peak policy")
    peak = max(float(np.max(np.abs(audio))) for audio in audios)
    if peak >= 1 and policy == "error":
        raise ValueError(f"Refusing PCM16 clipping: unattenuated peak={peak}")
    gain = .99 / peak if peak >= 1 else 1.0
    return [audio * gain for audio in audios], gain, peak


def prepare(baseline_test, baseline_recipe, raw_root, physical_root, csv_dir,
            output, seed=0, workers=4, expected_count=3000,
            control_tolerance=5e-5, power_tolerance_db=0.01, destination_root=None,
            peak_policy="error"):
    output = Path(output).resolve()
    if not any(output.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"Output must be inside an authorized user-owned root: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dump: {output}")
    destination_root = output if destination_root is None else Path(destination_root)
    if not destination_root.is_absolute():
        raise ValueError("Destination root embedded in SCPs must be absolute")
    if (workers < 1 or expected_count < 2 or seed < 0
            or not np.isfinite(control_tolerance) or control_tolerance <= 0
            or not np.isfinite(power_tolerance_db) or power_tolerance_db <= 0):
        raise ValueError("Invalid worker/count/tolerance setting")
    if peak_policy not in ("error", "common-attenuate"):
        raise ValueError(f"Unknown peak policy: {peak_policy}")
    baseline_test, baseline_recipe, raw_root, physical_root, csv_dir = [
        Path(p).resolve(strict=True)
        for p in (baseline_test, baseline_recipe, raw_root, physical_root, csv_dir)]
    filelists = {s: read_filelist(csv_dir / f"mix_2_spk_filenames_{s}.csv")
                 for s in ("tr", "cv", "tt")}
    for split in ("tr", "cv"):
        check_split_disjoint(filelists["tt"], filelists[split])
    indexes = {name: read_scp(baseline_test / name) for name in NAMES}
    uids = sorted(indexes["wav.scp"])
    if len(uids) != expected_count or any(set(m) != set(uids) for m in indexes.values()):
        raise ValueError("Baseline test key sets/count differ from requested complete test")
    if {canonical_filename(u) for u in uids} != set(filelists["tt"]):
        raise ValueError("Baseline keys and original test CSV do not match")
    baseline = {name: {uid: (Path(p) if Path(p).is_absolute()
                            else baseline_recipe / p).resolve(strict=True)
                       for uid, p in values.items()} for name, values in indexes.items()}
    metadata_path = physical_root / "wav8k/min/tt/utterances.jsonl"
    metadata = {}
    for line in metadata_path.read_text().splitlines():
        row = json.loads(line)
        if row["utterance_id"] in metadata:
            raise ValueError("Duplicate physical-generation metadata key")
        metadata[row["utterance_id"]] = row
    if set(metadata) != {Path(canonical_filename(u)).stem for u in uids}:
        raise ValueError("Physical-generation metadata and baseline keys differ")

    candidate_paths = sorted({r[f"s{k}_path"] for r in filelists["tt"].values() for k in (1, 2)})
    pool, pool_audit = {}, []
    for relative in candidate_paths:
        path = (raw_root / relative).resolve(strict=True)
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
            raise ValueError(f"Expected mono 16-kHz raw speech: {path}")
        speaker = Path(relative).parent.name
        item = dict(path=str(path), relative=relative, speaker=speaker,
                    frames_16k=info.frames, sha256=sha256(path))
        pool.setdefault(speaker, []).append(item)
        pool_audit.append(item)

    scenes, feasibility = [], []
    for uid in uids:
        filename = canonical_filename(uid)
        row = filelists["tt"][filename]
        meta = metadata[Path(filename).stem]
        headers = [sf.info(baseline[name][uid]) for name in NAMES]
        n = headers[0].frames
        if n <= 128 or any(h.samplerate != 8000 or h.frames != n
                          or h.channels not in (1, 2) or h.subtype != "PCM_16"
                          for h in headers):
            raise ValueError(f"Expected aligned Baseline 8-kHz PCM16 files: {uid}")
        old = [(raw_root / row[f"s{k}_path"]).resolve(strict=True) for k in (1, 2)]
        if old[0].parent.name == old[1].parent.name:
            raise ValueError(f"Original scene is already same-speaker: {uid}")
        for k in (1, 2):
            # Metadata can contain obsolete absolute prefixes; identity must match.
            if Path(meta[f"source{k}"]).name != old[k - 1].name:
                raise ValueError(f"Raw source and saved physical metadata differ: {uid}")
        choices = [[item for item in pool[p.parent.name]
                    if item["path"] != str(p) and item["frames_16k"] >= 2 * n - 1]
                   for p in old]
        feasibility.append([bool(c) for c in choices])
        scenes.append(dict(uid=uid, filename=filename, length=n, old=old,
                           choices=choices, meta=meta))
    rng = np.random.default_rng(seed)
    anchors = balanced_anchors(feasibility, rng)
    for scene, anchor in zip(scenes, anchors):
        scene["anchor"] = int(anchor)
        choices = scene["choices"][anchor - 1]
        scene["donor"] = choices[int(rng.integers(len(choices)))]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        folder = temporary / "raw" / DATASET
        for signal in ("wav", "spk1", "spk2"):
            (folder / "data" / signal).mkdir(parents=True)

        def process(scene):
            uid, n, filename = scene["uid"], scene["length"], scene["filename"]
            references = [read_left(baseline[f"spk{k}.scp"][uid], 8000) for k in (1, 2)]
            powers = [float(np.mean(x ** 2)) for x in references]
            raw = [read_left(p, 16000) for p in scene["old"]]
            physical, rir_hashes, rir_paths = [], {}, {}
            original = []
            for k in (1, 2):
                filtered = []
                for condition in ("anechoic", "reverb"):
                    key = f"rir{k}_{condition}"
                    path = physical_root / "wav16k/min/tt" / key / filename
                    h = read_left(path, 16000)
                    rir_paths[key], rir_hashes[key] = str(path), sha256(path)
                    filtered.append(convolve_8k(raw[k - 1], h, n))
                    physical.append(h)
                original.append(filtered)
            gains = [float(scene["meta"][f"s{k}_spatial_scaling"]) for k in (1, 2)]
            errors = [control_error(original[k][0] * gains[k], baseline[f"spk{k + 1}.scp"][uid])
                      for k in (0, 1)]
            errors.append(control_error(sum(original[k][1] * gains[k] for k in (0, 1)),
                                        baseline["wav.scp"][uid]))
            if max(errors) > control_tolerance:
                raise ValueError(f"Original identity control failed: {uid}, max_errors={errors}")
            new_paths = list(scene["old"])
            replaced = 2 - scene["anchor"]
            new_paths[replaced] = Path(scene["donor"]["path"])
            new_filtered = [list(pair) for pair in original]
            donor_raw = read_left(new_paths[replaced], 16000)
            new_filtered[replaced] = [convolve_8k(donor_raw, physical[2 * replaced + j], n)
                                     for j in (0, 1)]
            direct, reverb, applied = [], [], []
            for k in (0, 1):
                d, r, g = match_power(*new_filtered[k], powers[k])
                direct.append(d)
                reverb.append(r)
                applied.append(g)
            audios = [reverb[0] + reverb[1], *direct]
            try:
                audios, common_gain, unattenuated_peak = apply_peak_policy(audios, peak_policy)
            except ValueError as error:
                raise ValueError(f"{uid}: {error}") from error
            saved_powers = [power * common_gain ** 2 for power in powers]
            peaks, actual_powers = [], []
            for signal, audio in zip(("wav", "spk1", "spk2"), audios):
                path = folder / "data" / signal / f"{uid}.wav"
                peaks.append(write_pcm(path, audio))
                if signal != "wav":
                    actual_powers.append(float(np.mean(read_left(path, 8000) ** 2)))
            errors_db = [float(10 * np.log10(a / b)) for a, b in zip(actual_powers, saved_powers)]
            if max(abs(e) for e in errors_db) > power_tolerance_db:
                raise ValueError(f"PCM16 source power mismatch: {uid}: {errors_db} dB")
            record = dict(scene_uid=uid, original_filename=filename, frames_8k=n,
                          original_speaker1=scene["old"][0].parent.name,
                          original_speaker2=scene["old"][1].parent.name,
                          anchor_slot=scene["anchor"], actual_speaker=new_paths[0].parent.name,
                          source1=str(new_paths[0]), source2=str(new_paths[1]),
                          original_source1=str(scene["old"][0]), original_source2=str(scene["old"][1]),
                          eligible_anchor1=len(scene["choices"][0]), eligible_anchor2=len(scene["choices"][1]),
                          target_t60=float(scene["meta"]["target_t60"]),
                          source1_sha256=sha256(new_paths[0]), source2_sha256=sha256(new_paths[1]))
            record.update(common_peak_gain=common_gain, unattenuated_peak=unattenuated_peak)
            if new_paths[0].parent.name != new_paths[1].parent.name or new_paths[0] == new_paths[1]:
                raise ValueError(f"Same-speaker/distinct-utterance invariant failed: {uid}")
            for k in (0, 1):
                record.update({f"original_gain{k + 1}": gains[k],
                               f"power_matching_gain{k + 1}": applied[k],
                               f"applied_gain{k + 1}": applied[k] * common_gain,
                               f"target_power{k + 1}": powers[k], f"pcm16_power{k + 1}": actual_powers[k],
                               f"saved_target_power{k + 1}": saved_powers[k],
                               f"power_error_db{k + 1}": errors_db[k],
                               f"control_spk{k + 1}_max_error": errors[k]})
            record.update(control_mix_max_error=errors[2], peak_mix=peaks[0],
                          peak_spk1=peaks[1], peak_spk2=peaks[2])
            for signal, name in zip(("wav", "spk1", "spk2"), NAMES):
                record[f"new_{signal}_sha256"] = sha256(folder / "data" / signal / f"{uid}.wav")
                record[f"baseline_{signal}_sha256"] = sha256(baseline[name][uid])
            record.update(rir_paths)
            record.update({f"{key}_sha256": value for key, value in rir_hashes.items()})
            return record

        records = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for count, record in enumerate(executor.map(process, scenes), 1):
                records.append(record)
                if count % 100 == 0 or count == len(scenes):
                    print(f"Prepared and identity-checked {count}/{len(scenes)} scenes", flush=True)
        for name, signal in zip(NAMES, ("wav", "spk1", "spk2")):
            (folder / name).write_text("".join(
                f"{uid} {destination_root / 'raw' / DATASET / 'data' / signal / (uid + '.wav')}\n" for uid in uids))
        (folder / "utt2num_samples").write_text("".join(f"{r['scene_uid']} {r['frames_8k']}\n" for r in records))
        (folder / "feats_type").write_text("raw\n")
        (folder / "channel").write_text("0\n")
        (folder / "utt2spk").write_text("".join(f"{r['scene_uid']} {r['actual_speaker']}_{r['actual_speaker']}\n" for r in records))
        grouped = {}
        for r in records:
            speaker = f"{r['actual_speaker']}_{r['actual_speaker']}"
            grouped.setdefault(speaker, []).append(r["scene_uid"])
        (folder / "spk2utt").write_text("".join(f"{s} {' '.join(v)}\n" for s, v in sorted(grouped.items())))
        with (temporary / "audit.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(records[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        (temporary / "audit.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        (temporary / "candidate_pool.json").write_text(json.dumps(pool_audit, indent=2, sort_keys=True) + "\n")
        provenance = dict(
            definition="two distinct utterances of ONE speaker, original scene/RIR geometry retained",
            original_scene_ids_retained=True, estimated_ctf="MUST re-estimate from this new mixture; not generated here",
            sample_rate=8000, microphone="left/channel0", channels=1, subtype="PCM_16",
            source_normalization="match original prepared anechoic channel0 power per position, then optional common whole-example attenuation; saved target=original target*common_peak_gain**2",
            filtering="raw16 -> saved physical RIR16 convolution -> resample_poly(1,2) -> truncate original length",
            seed=seed, rng="numpy.default_rng/PCG64", numpy_version=np.__version__,
            scipy_version=scipy_version, soundfile_version=sf.__version__,
            libsndfile_version=sf.__libsndfile_version__,
            scene_count=len(records), raw_test_utterance_count=len(pool_audit),
            raw_test_speaker_count=len(pool), anchor_slot1_count=int(np.sum(anchors == 1)),
            anchor_slot2_count=int(np.sum(anchors == 2)),
            forced_anchor_count=int(np.sum(np.asarray(feasibility).sum(axis=1) == 1)),
            peak_policy=peak_policy,
            attenuated_scene_count=sum(r["common_peak_gain"] < 1 for r in records),
            max_attenuation_db=float(-20 * np.log10(min(r["common_peak_gain"] for r in records))),
            train_valid_created=False, test_train_valid_utterances_and_speakers_disjoint=True,
            baseline_test=str(baseline_test), baseline_recipe=str(baseline_recipe),
            destination_root=str(destination_root),
            raw_root=str(raw_root), physical_root=str(physical_root), csv_dir=str(csv_dir),
            control_tolerance=control_tolerance, power_tolerance_db=power_tolerance_db,
            worst_control_max_error=max(r[key] for r in records for key in
                                        ("control_mix_max_error", "control_spk1_max_error", "control_spk2_max_error")),
            worst_power_error_db=max(abs(r[f"power_error_db{k}"]) for r in records for k in (1, 2)),
            source_hashes={str(p): sha256(p) for p in [metadata_path, *[baseline_test / n for n in NAMES],
                *[csv_dir / f"mix_2_spk_filenames_{s}.csv" for s in ("tr", "cv", "tt")], Path(__file__)]},
            output_hashes={str(p.relative_to(temporary)): sha256(p)
                           for p in sorted(temporary.rglob("*")) if p.is_file()})
        (temporary / "preparation.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
        return provenance
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-test", "baseline-recipe", "raw-root", "physical-root", "csv-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=3000)
    parser.add_argument("--control-tolerance", type=float, default=5e-5)
    parser.add_argument("--power-tolerance-db", type=float, default=0.01)
    parser.add_argument("--destination-root", type=Path,
                        help="Absolute final SCP path prefix, e.g. the remote dump path")
    parser.add_argument("--peak-policy", choices=("error", "common-attenuate"), default="error",
                        help="Explicitly allow common attenuation of all three signals if PCM16 would clip")
    args = parser.parse_args()
    report = prepare(**vars(args))
    print(json.dumps({k: report[k] for k in ("scene_count", "anchor_slot1_count", "anchor_slot2_count",
                                            "worst_control_max_error", "worst_power_error_db")}, indent=2))


if __name__ == "__main__":
    main()
