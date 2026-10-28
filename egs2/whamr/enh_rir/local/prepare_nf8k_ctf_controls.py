#!/usr/bin/env python3
"""Generate CPU-only, test-only NF-WHAMR conditioning-audio controls.

The output mixtures are inputs for a FUTURE CTF estimation experiment, not
replacement separation/scoring inputs. No CTFs, model inference, or scores are
generated. All conditions retain the original scene UID and length. Read the
audit before interpreting the original UID as the new speech/RIR assignment.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import __version__ as scipy_version
import soundfile as sf

from prepare_nf8k_same_speaker_test import (
    ALLOWED_ROOTS,
    NAMES,
    apply_peak_policy,
    canonical_filename,
    check_split_disjoint,
    control_error,
    convolve_8k,
    match_power,
    read_filelist,
    read_left,
    read_scp,
    sha256,
    write_pcm,
)


CONDITIONS = {
    "same_rir_other_utt": "tt_same_rir_other_utt_8k",
    "same_utt_swapped_rir": "tt_same_utt_swapped_rir_8k",
    "same_utt_other_rir": "tt_same_utt_other_rir_8k",
}
DEFAULT_CONDITIONS = ("same_rir_other_utt", "same_utt_swapped_rir")
DEFINITIONS = {
    "same_rir_other_utt": (
        "s1_prime*h1 + s2_prime*h2; independently select two different-speaker "
        "raw utterances from the original test pool, and neither utterance "
        "occurs in the target scene; preserve the complete target length"
    ),
    "same_utt_swapped_rir": (
        "s1*h2 + s2*h1; the original two utterances are retained and both "
        "anechoic and reverberant RIR assignments are swapped; this retains "
        "the RIR set and is NOT a wrong-RIR-set control"
    ),
    "same_utt_other_rir": (
        "s1*h3 + s2*h4; retain the original dry utterances and use the complete "
        "anechoic/reverberant RIR pair from one different original test scene; "
        "fixed-seed Sattolo derangement over the complete test ensures distinct "
        "target/donor scene UIDs and bijective donor use; reject hash-identical "
        "RIR sets, including order-insensitive reverberant pairs"
    ),
}


def authorized_output(path):
    """Resolve existing parent symlinks before any directory is created."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Output and destination roots must be absolute")
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in ALLOWED_ROOTS):
        raise ValueError(f"Output must be inside an authorized user-owned root: {resolved}")
    return resolved


def read_metadata(path, uids):
    metadata = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        key = row["utterance_id"]
        if key in metadata:
            raise ValueError(f"Duplicate physical metadata key: {path}:{lineno}")
        metadata[key] = row
    if set(metadata) != {Path(canonical_filename(uid)).stem for uid in uids}:
        raise ValueError("Physical-generation metadata and baseline keys differ")
    return metadata


def choose_donors(scenes, seed):
    """Select two eligible raw test utterances; reuse across targets is allowed."""
    pool = {}
    for scene in scenes:
        for path, frames in zip(scene["original_sources"], scene["raw_frames_16k"]):
            item = pool.setdefault(str(path), dict(
                path=path, frames_16k=frames, speaker=path.parent.name, scene_uids=[]))
            item["scene_uids"].append(scene["uid"])
    pool = [pool[key] for key in sorted(pool)]
    rng = np.random.default_rng(seed)
    for target in scenes:
        original = {str(path) for path in target["original_sources"]}
        minimum_frames = 2 * target["frames_8k"] - 1
        eligible = [
            candidate for candidate in pool
            if str(candidate["path"]) not in original
            and candidate["frames_16k"] >= minimum_frames
        ]
        if len({candidate["speaker"] for candidate in eligible}) < 2:
            raise ValueError(
                f"No two different-speaker test donor utterances distinct from "
                f"both target utterances with sufficient raw length: {target['uid']}"
            )
        first = eligible[int(rng.integers(len(eligible)))]
        eligible_second = [candidate for candidate in eligible
                           if candidate["speaker"] != first["speaker"]]
        second = eligible_second[int(rng.integers(len(eligible_second)))]
        target["eligible_donor_source1_count"] = len(eligible)
        target["eligible_donor_source2_count"] = len(eligible_second)
        target["donor_sources"] = [first["path"], second["path"]]
        target["donor_origin_scene_uids"] = [first["scene_uids"], second["scene_uids"]]


def choose_rir_donors(scenes, seed):
    """Assign each complete physical RIR scene once using a Sattolo cycle."""
    ordered = sorted(scenes, key=lambda scene: scene["uid"])
    if len(ordered) < 2 or len({scene["uid"] for scene in ordered}) != len(ordered):
        raise ValueError("RIR derangement requires at least two unique scene UIDs")
    permutation = list(range(len(ordered)))
    rng = np.random.default_rng(seed)
    for index in range(len(permutation) - 1, 0, -1):
        other = int(rng.integers(index))  # Strictly below index: no fixed points.
        permutation[index], permutation[other] = permutation[other], permutation[index]
    hashes = {
        scene["uid"]: {key: sha256(path) for key, path in scene["rir_paths"].items()}
        for scene in ordered
    }
    for index, target in enumerate(ordered):
        donor = ordered[permutation[index]]
        target_hashes, donor_hashes = hashes[target["uid"]], hashes[donor["uid"]]
        target_reverb = sorted(target_hashes[f"rir{k}_reverb"] for k in (1, 2))
        donor_reverb = sorted(donor_hashes[f"rir{k}_reverb"] for k in (1, 2))
        if (sorted(target_hashes.values()) == sorted(donor_hashes.values())
                or target_reverb == donor_reverb):
            raise ValueError(
                "Donor RIR set equals target RIR set by hashes "
                f"(including order-insensitive reverb pairs): "
                f"target={target['uid']}, donor={donor['uid']}"
            )
        target["donor_rir_uid"] = donor["uid"]
        target["donor_rir_paths"] = donor["rir_paths"]
        target["donor_rir_t60"] = float(donor["metadata"]["target_t60"])


def write_audits(root, records):
    with (root / "audit.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (root / "audit.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def prepare(baseline_test, baseline_recipe, raw_root, physical_root, csv_dir,
            output, seed=0, workers=4, expected_count=3000, limit=None,
            control_tolerance=5e-5, power_tolerance_db=0.01,
            destination_root=None, peak_policy="common-attenuate", conditions=None):
    condition_names = DEFAULT_CONDITIONS if conditions is None else tuple(conditions)
    if (not condition_names or len(set(condition_names)) != len(condition_names)
            or any(name not in CONDITIONS for name in condition_names)):
        raise ValueError("Conditions must be a nonempty, unique subset of known controls")
    selected_conditions = {name: CONDITIONS[name] for name in condition_names}
    output_argument = Path(output)
    output = authorized_output(output_argument)
    if output.exists() or output_argument.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing dump: {output_argument}")
    destination_root = output if destination_root is None else authorized_output(destination_root)
    if (workers < 1 or expected_count < 2 or seed < 0
            or (limit is not None and not 1 <= limit <= expected_count)
            or not np.isfinite(control_tolerance) or control_tolerance <= 0
            or not np.isfinite(power_tolerance_db) or power_tolerance_db <= 0):
        raise ValueError("Invalid workers/count/limit/seed/tolerance setting")
    if peak_policy not in ("error", "common-attenuate"):
        raise ValueError(f"Unknown peak policy: {peak_policy}")
    baseline_test, baseline_recipe, raw_root, physical_root, csv_dir = [
        Path(path).resolve(strict=True)
        for path in (baseline_test, baseline_recipe, raw_root, physical_root, csv_dir)
    ]
    filelists = {
        split: read_filelist(csv_dir / f"mix_2_spk_filenames_{split}.csv")
        for split in ("tr", "cv", "tt")
    }
    for split in ("tr", "cv"):
        check_split_disjoint(filelists["tt"], filelists[split])
    indexes = {name: read_scp(baseline_test / name) for name in NAMES}
    all_uids = sorted(indexes["wav.scp"])
    if (len(all_uids) != expected_count
            or any(set(values) != set(all_uids) for values in indexes.values())):
        raise ValueError("Baseline test key sets/count differ from requested complete test")
    if {canonical_filename(uid) for uid in all_uids} != set(filelists["tt"]):
        raise ValueError("Baseline keys and original test CSV do not match")
    baseline = {
        name: {
            uid: (Path(path) if Path(path).is_absolute() else baseline_recipe / path).resolve(strict=True)
            for uid, path in values.items()
        }
        for name, values in indexes.items()
    }
    metadata_path = physical_root / "wav8k/min/tt/utterances.jsonl"
    metadata = read_metadata(metadata_path, all_uids)

    # The pool is strictly the original test CSV, never train/cv or a new dump.
    raw_info = {}
    for relative in sorted({row[f"s{k}_path"] for row in filelists["tt"].values() for k in (1, 2)}):
        path = (raw_root / relative).resolve(strict=True)
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
            raise ValueError(f"Expected mono 16-kHz raw speech: {path}")
        raw_info[str(path)] = dict(
            path=str(path), relative=relative, speaker=Path(relative).parent.name,
            frames_16k=info.frames, sha256=sha256(path),
        )
    scenes = []
    for uid in all_uids:
        filename = canonical_filename(uid)
        row = filelists["tt"][filename]
        meta = metadata[Path(filename).stem]
        headers = [sf.info(baseline[name][uid]) for name in NAMES]
        length = headers[0].frames
        if length <= 128 or any(
            info.samplerate != 8000 or info.frames != length
            or info.channels not in (1, 2) or info.subtype != "PCM_16"
            for info in headers
        ):
            raise ValueError(f"Expected aligned Baseline 8-kHz PCM16 files: {uid}")
        sources = [(raw_root / row[f"s{k}_path"]).resolve(strict=True) for k in (1, 2)]
        if sources[0] == sources[1]:
            raise ValueError(f"Original scene contains the identical raw utterance twice: {uid}")
        gains = [float(meta[f"s{k}_spatial_scaling"]) for k in (1, 2)]
        if any(not np.isfinite(gain) or gain <= 0 for gain in gains):
            raise ValueError(f"Invalid saved physical-generation gains: {uid}")
        for k in (1, 2):
            if Path(meta[f"source{k}"]).name != sources[k - 1].name:
                raise ValueError(f"Raw source and saved physical metadata differ: {uid}")
        frames = [raw_info[str(path)]["frames_16k"] for path in sources]
        if any(n < 2 * length - 1 for n in frames):
            raise ValueError(f"Original raw speech shorter than requested target: {uid}")
        rirs = {
            f"rir{k}_{kind}": (physical_root / "wav16k/min/tt" / f"rir{k}_{kind}" / filename).resolve(strict=True)
            for k in (1, 2) for kind in ("anechoic", "reverb")
        }
        scenes.append(dict(
            uid=uid, filename=filename, frames_8k=length,
            original_sources=sources, raw_frames_16k=frames,
            original_gains=gains, metadata=meta, rir_paths=rirs,
        ))
    if "same_rir_other_utt" in selected_conditions:
        choose_donors(scenes, seed)
    if "same_utt_other_rir" in selected_conditions:
        choose_rir_donors(scenes, seed)
    selected = scenes if limit is None else scenes[:limit]

    def identity_check(scene):
        uid, length = scene["uid"], scene["frames_8k"]
        raw = [read_left(path, 16000) for path in scene["original_sources"]]
        generated = []
        for k in (1, 2):
            generated.append({
                kind: convolve_8k(raw[k - 1], read_left(scene["rir_paths"][f"rir{k}_{kind}"], 16000), length)
                for kind in ("anechoic", "reverb")
            })
        gains = scene["original_gains"]
        errors = [
            control_error(generated[k]["anechoic"] * gains[k], baseline[f"spk{k + 1}.scp"][uid])
            for k in (0, 1)
        ]
        errors.append(control_error(
            generated[0]["reverb"] * gains[0] + generated[1]["reverb"] * gains[1],
            baseline["wav.scp"][uid],
        ))
        if max(errors) > control_tolerance:
            raise ValueError(f"Original identity control failed: {uid}, max_errors={errors}")
        return dict(
            scene_uid=uid, frames_8k=length,
            control_spk1_max_error=errors[0], control_spk2_max_error=errors[1],
            control_mix_max_error=errors[2],
        )

    # A pilot still verifies the complete original test before producing data.
    print(f"Checking original identity reconstruction for all {len(scenes)} test scenes", flush=True)
    controls = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for count, result in enumerate(executor.map(identity_check, scenes), 1):
            controls.append(result)
            if count % 100 == 0 or count == len(scenes):
                print(f"Identity-checked {count}/{len(scenes)} scenes", flush=True)
    control_map = {record["scene_uid"]: record for record in controls}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for dataset in selected_conditions.values():
            for signal in ("wav", "spk1", "spk2"):
                (temporary / "raw" / dataset / "data" / signal).mkdir(parents=True)

        def generate(scene):
            uid, length = scene["uid"], scene["frames_8k"]
            references = [read_left(baseline[f"spk{k}.scp"][uid], 8000) for k in (1, 2)]
            powers = [float(np.mean(audio ** 2)) for audio in references]
            original_rir_hashes = {key: sha256(path) for key, path in scene["rir_paths"].items()}
            records = []
            for condition, dataset in selected_conditions.items():
                sources = (scene["donor_sources"] if condition == "same_rir_other_utt"
                           else scene["original_sources"])
                assignments = (2, 1) if condition == "same_utt_swapped_rir" else (1, 2)
                rir_paths = (scene["donor_rir_paths"] if condition == "same_utt_other_rir"
                             else scene["rir_paths"])
                rirs = {key: read_left(path, 16000) for key, path in rir_paths.items()}
                rir_hashes = {key: sha256(path) for key, path in rir_paths.items()}
                direct, reverb, applied = [], [], []
                for k, (path, rir_slot) in enumerate(zip(sources, assignments)):
                    raw = read_left(path, 16000)
                    filtered = [convolve_8k(raw, rirs[f"rir{rir_slot}_{kind}"], length)
                                for kind in ("anechoic", "reverb")]
                    d, r, gain = match_power(*filtered, powers[k])
                    direct.append(d)
                    reverb.append(r)
                    applied.append(gain)
                try:
                    audios, common_gain, unattenuated_peak = apply_peak_policy(
                        [reverb[0] + reverb[1], *direct], peak_policy,
                    )
                except ValueError as error:
                    raise ValueError(f"{condition}/{uid}: {error}") from error
                folder = temporary / "raw" / dataset
                saved_powers = [power * common_gain ** 2 for power in powers]
                peaks, actual_powers, output_hashes = [], [], {}
                for signal, audio in zip(("wav", "spk1", "spk2"), audios):
                    path = folder / "data" / signal / f"{uid}.wav"
                    peaks.append(write_pcm(path, audio))
                    output_hashes[signal] = sha256(path)
                    if signal != "wav":
                        actual_powers.append(float(np.mean(read_left(path, 8000) ** 2)))
                errors_db = [float(10 * np.log10(a / b)) for a, b in zip(actual_powers, saved_powers)]
                ratio_error_db = float(10 * np.log10(
                    (actual_powers[0] / actual_powers[1]) / (powers[0] / powers[1])))
                if max([abs(error) for error in errors_db] + [abs(ratio_error_db)]) > power_tolerance_db:
                    raise ValueError(
                        f"PCM16 source power/ratio mismatch: {condition}/{uid}: "
                        f"errors={errors_db}, ratio={ratio_error_db} dB"
                    )
                donor_origins = (scene["donor_origin_scene_uids"]
                                 if condition == "same_rir_other_utt"
                                 else [[uid], [uid]])
                record = dict(
                    condition=condition, dataset=dataset, scene_uid=uid,
                    original_filename=scene["filename"], frames_8k=length,
                    seed=seed,
                    source1_origin_scene_uids=";".join(donor_origins[0]),
                    source2_origin_scene_uids=";".join(donor_origins[1]),
                    eligible_donor_source1_count=scene["eligible_donor_source1_count"] if condition == "same_rir_other_utt" else 0,
                    eligible_donor_source2_count=scene["eligible_donor_source2_count"] if condition == "same_rir_other_utt" else 0,
                    original_source1=str(scene["original_sources"][0]),
                    original_source2=str(scene["original_sources"][1]),
                    source1=str(sources[0]), source2=str(sources[1]),
                    source1_rir_slot=assignments[0], source2_rir_slot=assignments[1],
                    source1_speaker=sources[0].parent.name, source2_speaker=sources[1].parent.name,
                    target_t60=float(scene["metadata"]["target_t60"]),
                    source_t60=(scene["donor_rir_t60"] if condition == "same_utt_other_rir"
                                else float(scene["metadata"]["target_t60"])),
                    donor_rir_uid=scene["donor_rir_uid"] if condition == "same_utt_other_rir" else uid,
                    common_peak_gain=common_gain, unattenuated_peak=unattenuated_peak,
                    power_ratio_error_db=ratio_error_db,
                    control_spk1_max_error=control_map[uid]["control_spk1_max_error"],
                    control_spk2_max_error=control_map[uid]["control_spk2_max_error"],
                    control_mix_max_error=control_map[uid]["control_mix_max_error"],
                    peak_mix=peaks[0], peak_spk1=peaks[1], peak_spk2=peaks[2],
                )
                for k in (0, 1):
                    record.update({
                        f"original_source{k + 1}_sha256": raw_info[str(scene["original_sources"][k])]["sha256"],
                        f"source{k + 1}_sha256": raw_info[str(sources[k])]["sha256"],
                        f"original_gain{k + 1}": scene["original_gains"][k],
                        f"power_matching_gain{k + 1}": applied[k],
                        f"applied_gain{k + 1}": applied[k] * common_gain,
                        f"target_power{k + 1}": powers[k],
                        f"saved_target_power{k + 1}": saved_powers[k],
                        f"pcm16_power{k + 1}": actual_powers[k],
                        f"power_error_db{k + 1}": errors_db[k],
                    })
                for key, path in rir_paths.items():
                    record[key] = str(path)
                    record[f"{key}_sha256"] = rir_hashes[key]
                    record[f"original_{key}"] = str(scene["rir_paths"][key])
                    record[f"original_{key}_sha256"] = original_rir_hashes[key]
                for name, signal in zip(NAMES, ("wav", "spk1", "spk2")):
                    record[f"baseline_{signal}"] = str(baseline[name][uid])
                    record[f"baseline_{signal}_sha256"] = sha256(baseline[name][uid])
                    record[f"new_{signal}_sha256"] = output_hashes[signal]
                records.append(record)
            return records

        records = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for count, result in enumerate(executor.map(generate, selected), 1):
                records.extend(result)
                if count % 100 == 0 or count == len(selected):
                    print(f"Generated {len(selected_conditions)} condition(s) for {count}/{len(selected)} scenes", flush=True)
        for condition, dataset in selected_conditions.items():
            folder = temporary / "raw" / dataset
            subset = [record for record in records if record["condition"] == condition]
            for name, signal in zip(NAMES, ("wav", "spk1", "spk2")):
                (folder / name).write_text("".join(
                    f"{record['scene_uid']} {destination_root / 'raw' / dataset / 'data' / signal / (record['scene_uid'] + '.wav')}\n"
                    for record in subset
                ))
            (folder / "utt2num_samples").write_text("".join(
                f"{record['scene_uid']} {record['frames_8k']}\n" for record in subset))
            (folder / "feats_type").write_text("raw\n")
            (folder / "channel").write_text("0\n")
            (folder / "utt2spk").write_text("".join(
                f"{record['scene_uid']} {record['source1_speaker']}_{record['source2_speaker']}\n"
                for record in subset))
            grouped = {}
            for record in subset:
                key = f"{record['source1_speaker']}_{record['source2_speaker']}"
                grouped.setdefault(key, []).append(record["scene_uid"])
            (folder / "spk2utt").write_text("".join(
                f"{key} {' '.join(values)}\n" for key, values in sorted(grouped.items())))
        write_audits(temporary, records)
        (temporary / "identity_controls.json").write_text(json.dumps(controls, indent=2, sort_keys=True) + "\n")
        (temporary / "candidate_pool.json").write_text(json.dumps(
            list(raw_info.values()), indent=2, sort_keys=True) + "\n")
        source_paths = [
            metadata_path, Path(__file__).resolve(),
            Path(__file__).with_name("prepare_nf8k_same_speaker_test.py").resolve(),
            *[baseline_test / name for name in NAMES],
            *[csv_dir / f"mix_2_spk_filenames_{split}.csv" for split in ("tr", "cv", "tt")],
        ]
        provenance = dict(
            condition_definitions={name: DEFINITIONS[name] for name in selected_conditions},
            intended_use="CTF-estimator input audio only; keep original test mixture/references fixed for future separation/scoring",
            ctf_estimation_run=False, model_inference_run=False, scoring_run=False,
            train_valid_created=False, original_scene_ids_retained=True,
            expected_original_test_scene_count=expected_count,
            original_identity_verified_count=len(controls),
            scene_count=len(selected), condition_count=len(selected_conditions),
            generated_wav_count=3 * len(records), limit=limit,
            pilot_selection="lexicographically first N original scene UIDs; no metric/score selection",
            seed=seed, rng="numpy.default_rng/PCG64",
            donor_selection=("sequential uniform selection from eligible raw test utterances; second donor restricted to a different speaker from first; donor reuse across targets permitted; selected over complete original test before applying limit"
                             if "same_rir_other_utt" in selected_conditions else "not applicable; original raw utterances retained"),
            donor_eligibility=("neither donor utterance equals either target utterance; both mono16k raw lengths >= 2*target_frames_8k-1; donor speakers differ from one another but may match original speakers"
                               if "same_rir_other_utt" in selected_conditions else "not applicable; no speech donor selection"),
            padding_repetition_or_shortening_used=False,
            filtering="raw16 -> saved physical RIR16 convolution -> resample_poly(1,2) -> truncate original target length",
            source_normalization="match original prepared anechoic channel0 source power independently using the same gain for direct/reverb, then optional common whole-example attenuation",
            sample_rate=8000, channels=1, microphone="left/channel0", subtype="PCM_16",
            peak_policy=peak_policy, control_tolerance=control_tolerance,
            power_tolerance_db=power_tolerance_db,
            test_train_valid_utterances_and_speakers_disjoint=True,
            baseline_test=str(baseline_test), baseline_recipe=str(baseline_recipe),
            raw_root=str(raw_root), physical_root=str(physical_root), csv_dir=str(csv_dir),
            output_root=str(output), destination_root=str(destination_root),
            worst_control_max_error=max(record[key] for record in controls for key in
                                        ("control_mix_max_error", "control_spk1_max_error", "control_spk2_max_error")),
            worst_power_error_db=max(abs(record[f"power_error_db{k}"]) for record in records for k in (1, 2)),
            worst_power_ratio_error_db=max(abs(record["power_ratio_error_db"]) for record in records),
            attenuated_condition_count=sum(record["common_peak_gain"] < 1 for record in records),
            max_attenuation_db=float(-20 * np.log10(min(record["common_peak_gain"] for record in records))),
            numpy_version=np.__version__, scipy_version=scipy_version,
            soundfile_version=sf.__version__, libsndfile_version=sf.__libsndfile_version__,
            source_hashes={str(path): sha256(path) for path in source_paths},
            output_hashes={str(path.relative_to(temporary)): sha256(path)
                           for path in sorted(temporary.rglob("*")) if path.is_file()},
        )
        if "same_utt_other_rir" in selected_conditions:
            provenance.update(
                rir_donor_selection="Sattolo single-cycle derangement over lexicographically sorted complete original test UIDs before applying limit; one coherent physical scene pair per donor; no hash-identical target/donor RIR sets",
                rir_donor_scene_count=len(scenes),
                rir_donor_fixed_point_count=sum(scene["uid"] == scene["donor_rir_uid"] for scene in scenes),
                rir_donor_unique_count=len({scene["donor_rir_uid"] for scene in scenes}),
            )
        (temporary / "preparation.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite concurrently created dump: {output}")
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
    parser.add_argument("--limit", type=int, help="Generate only the first N sorted UIDs after validating the complete original test")
    parser.add_argument("--control-tolerance", type=float, default=5e-5)
    parser.add_argument("--power-tolerance-db", type=float, default=0.01)
    parser.add_argument("--destination-root", type=Path, help="Authorized absolute final path prefix embedded in SCPs")
    parser.add_argument("--peak-policy", choices=("error", "common-attenuate"), default="common-attenuate")
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS),
                        help="Generate only selected controls; default is the original two controls")
    args = parser.parse_args()
    report = prepare(**vars(args))
    print(json.dumps({key: report[key] for key in (
        "scene_count", "condition_count", "generated_wav_count",
        "worst_control_max_error", "worst_power_error_db", "worst_power_ratio_error_db",
    )}, indent=2))


if __name__ == "__main__":
    main()
