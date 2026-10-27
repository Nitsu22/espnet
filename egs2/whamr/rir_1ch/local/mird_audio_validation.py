"""Independently reconstruct MIRD evaluation audio from native recordings.

A smoke result checks real audio while preserving its explicit incomplete
status. Only a complete nine-condition, 3,000-mixture result can validate as
the production evaluation dataset.
"""

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

import create_whamr_but_dump as common
from create_whamr_but_2spk_dump import scp_references
from mird_rir import (
    NATIVE_ANGLES,
    NATIVE_FS,
    NATIVE_SAMPLES,
    archive_stem,
    inspect_response,
    member_name,
)


RATES = (8000, 16000)
CONDITIONS = ("clean", "noisy")
SIGNALS = {"source1", "source2", "direct1", "direct2", "reverb1", "reverb2",
           "noise", "mix_clean", "mix_both"}
TOLERANCE = 3e-6


def _read_json(path):
    return json.loads(Path(path).read_text())


def _same_number(actual, expected, label, *, tolerance=1e-12):
    if (isinstance(actual, bool) or not math.isfinite(float(actual))
            or not math.isclose(float(actual), float(expected),
                                rel_tol=tolerance, abs_tol=tolerance)):
        raise ValueError("Incorrect " + label)


def _output_path(path, out):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(out):
        raise ValueError("Audio path must point inside the dataset output")
    return path


def _float_audio(path, sr, frames):
    info = sf.info(path)
    if (info.samplerate, info.channels, info.frames, info.subtype) != (
            sr, 1, frames, "FLOAT"):
        raise ValueError("Audio header mismatch: " + str(path))
    wave, actual_rate = sf.read(path, dtype="float64")
    if actual_rate != sr or wave.ndim != 1 or not np.all(np.isfinite(wave)):
        raise ValueError("Invalid stored mono audio: " + str(path))
    return wave


def _checked_native_rirs(out, config, plan, selected, audit):
    """Rebuild each selected mono RIR directly from its checked native MAT."""
    rows = _read_json(out / "rir_metadata.json")
    expected = {(sr, rec["rt60_ms"], rec["spacing"], rec["microphone"], source)
                for rec in selected for source in rec["sources"] for sr in RATES}
    metadata = {}
    for item in rows:
        key = tuple(item[name] for name in (
            "sample_rate", "rt60_ms", "spacing", "microphone", "source_index"))
        if key in metadata:
            raise ValueError("Duplicate processed RIR metadata")
        metadata[key] = item
    if set(metadata) != expected:
        raise ValueError("Processed RIR inventory differs from selected receivers")
    audit_rows = audit["archives"]
    sessions = {(row["rt60_ms"], row["spacing"]): row for row in audit_rows}
    if len(sessions) != 9 or len(audit_rows) != 9:
        raise ValueError("Native audit must describe exactly nine conditions")
    valid = sum(row["valid"] is True for row in audit_rows)
    if (audit["valid_conditions"] != valid
            or audit["all_conditions_valid"] != (valid == 9)):
        raise ValueError("Inconsistent native archive audit status")
    archive_root = Path(config["archives_root"]).resolve()
    # Verify the original archives themselves, including the rejected archive
    # recorded in a smoke audit, rather than trusting saved local hashes.
    for (rt, spacing), row in sessions.items():
        path = archive_root / (archive_stem(rt, spacing) + ".zip")
        if str(path.resolve()) != row["path"]:
            raise ValueError("Native archive path differs from audited source")
        if common.digest(path) != row["sha256"]:
            raise ValueError("Native archive changed since its audit")
    grouped = defaultdict(lambda: defaultdict(list))
    for key, item in metadata.items():
        sr, rt, spacing, mic, index = key
        session = sessions[rt, spacing]
        if not session["valid"]:
            raise ValueError("Selected RIR belongs to a rejected native archive")
        position = plan["source_grid"][index]
        distance, angle = position["distance_m"], position["native_azimuth_deg"]
        archive = archive_root / (archive_stem(rt, spacing) + ".zip")
        member = member_name(rt, spacing, distance, angle)
        if (item["native_archive"] != str(archive.resolve())
                or item["archive_sha256"] != session["sha256"]
                or item["native_member"] != member
                or item["distance_m"] != distance
                or item["native_azimuth_deg"] != angle
                or item["native_sample_rate"] != NATIVE_FS
                or item["native_samples"] != NATIVE_SAMPLES
                or item["samples"] != 10 * sr):
            raise ValueError("RIR metadata differs from its native source")
        grouped[rt, spacing][distance, angle].append((key, item))
    cached = {}
    worst = 0.0
    for (rt, spacing), members in sorted(grouped.items()):
        archive_path = archive_root / (archive_stem(rt, spacing) + ".zip")
        with zipfile.ZipFile(archive_path) as archive:
            wanted = {member_name(rt, spacing, d, a)
                      for d in (1, 2) for a in NATIVE_ANGLES}
            names = [m.filename for m in archive.infolist() if not m.is_dir()]
            if len(names) != 26 or set(names) != wanted:
                raise ValueError("Selected native archive lacks its 26 MAT files")
            for (distance, angle), entries in sorted(members.items()):
                native, _ = inspect_response(
                    archive.read(member_name(rt, spacing, distance, angle)),
                    rt, spacing, distance, angle,
                )
                for key, item in entries:
                    sr, _, _, mic, _ = key
                    divisor = math.gcd(NATIVE_FS, sr)
                    response = resample_poly(
                        native[:, mic - 1].astype(np.float64),
                        sr // divisor, NATIVE_FS // divisor,
                    )
                    gain = float(np.max(np.abs(response)))
                    expected_wave = (response / gain).astype(np.float32)
                    peak = int(np.argmax(np.abs(expected_wave)))
                    path = _output_path(item["path"], out)
                    stored = _float_audio(path, sr, 10 * sr)
                    if common.digest(path) != item["sha256"]:
                        raise ValueError("Processed RIR hash mismatch")
                    error = float(np.max(np.abs(stored - expected_wave)))
                    worst = max(worst, error)
                    if error != 0.0 or item["peak_sample"] != peak:
                        raise ValueError(
                            "Saved RIR differs from selected native microphone"
                        )
                    _same_number(item["normalization_divisor"], gain,
                                 "RIR normalization divisor")
                    cached[key] = (expected_wave, peak, str(path.resolve()))
    return cached, worst


def _check_scps(out, by_key, selected, summary):
    expected_sets = {
        f"tt_mird_2spk_{condition}_reverb_min_{sr // 1000}k"
        for sr in RATES for condition in CONDITIONS
    }
    if (len(summary["sets"]) != 4 or set(summary["sets"]) != expected_sets
            or {p.name for p in (out / "raw").iterdir()} != expected_sets):
        raise ValueError("Expected exactly the four MIRD clean/noisy rate sets")
    extra = {"utt2spk", "spk2utt", "utt2spk1", "utt2spk2", "utt2num_samples"}
    uids = {record["utterance_id"] for record in selected}
    for name in sorted(expected_sets):
        folder = out / "raw" / name
        sr = 8000 if name.endswith("_8k") else 16000
        condition = "noisy" if "_noisy_" in name else "clean"
        subset = {uid: by_key[sr, uid] for uid in uids}
        wanted = set(scp_references(next(iter(subset.values())), condition))
        if {p.name for p in folder.glob("*.scp")} != wanted:
            raise ValueError("Incorrect SCP file schema: " + name)
        if (folder / "feats_type").read_text() != "raw\n":
            raise ValueError("Incorrect feature type")
        for filename in sorted(wanted | extra):
            lines = (folder / filename).read_text().splitlines()
            entries = [line.split(maxsplit=1) for line in lines]
            if (any(len(entry) != 2 for entry in entries)
                    or len(entries) != len(uids)
                    or {entry[0] for entry in entries} != uids
                    or [entry[0] for entry in entries] != sorted(uids)):
                raise ValueError("Incomplete, duplicate or unsorted SCP keys")
            for uid, value in entries:
                record = subset[uid]
                if filename in wanted:
                    expected = scp_references(record, condition)[filename]
                elif filename == "utt2num_samples":
                    expected = str(record["samples"])
                elif filename in ("utt2spk1", "utt2spk2"):
                    expected = record["speakers"][int(filename[-1]) - 1]
                else:
                    expected = uid
                if value != expected:
                    raise ValueError("Incorrect mapping: " + str(folder / filename))


def validate(out, workers=4):
    """Check original sources, native receiver selection and every audio sum."""
    # Import only when invoked so the generator can expose this validator
    # without a module initialization cycle.
    import create_whamr_mird_2spk_dump as gen

    out = Path(out).resolve()
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("Validation workers must be a positive integer")
    # An earlier success report must not survive a failed check after changes.
    (out / "validation.json").unlink(missing_ok=True)
    config = _read_json(out / "generation_config.json")
    mode = config["mode"]
    if mode not in ("smoke", "production") or config["target_count"] != 3000:
        raise ValueError("Unknown generation mode or production target")
    if (config["sample_rates"] != list(RATES)
            or config["conditions"] != list(CONDITIONS)):
        raise ValueError("Expected paired sample rates and clean/noisy conditions")
    production = mode == "production"
    marker = "generation_complete.json" if production else "smoke_complete.json"
    forbidden = "smoke_complete.json" if production else "generation_complete.json"
    if (out / forbidden).exists():
        raise ValueError("Production and smoke completion markers must be distinct")
    summary = _read_json(out / marker)
    if (summary["mode"] != mode
            or summary["target_utterances_per_set"] != 3000):
        raise ValueError("Completion marker mode/target mismatch")
    for key in ("filenames", "scaling"):
        if common.digest(Path(config[key])) != config[key + "_sha256"]:
            raise ValueError("Original WHAM source metadata changed: " + key)
    artifacts = {
        "assignment_plan.json": "plan_sha256",
        "native_rir_audit.json": "native_audit_sha256",
        "selected_records.json": "selected_records_sha256",
        "rir_metadata.json": "rir_metadata_sha256",
    }
    for filename, checksum in artifacts.items():
        if common.digest(out / filename) != config[checksum]:
            raise ValueError("Generation artifact hash mismatch: " + filename)
    args = argparse.Namespace(
        plan=out / "assignment_plan.json", filenames=Path(config["filenames"]),
        scaling=Path(config["scaling"]), wsj_root=Path(config["wsj_root"]),
        noise_root=Path(config["noise_root"]),
    )
    rows, scaling, ids, plan = gen.load_inputs(args)
    audit = _read_json(out / "native_rir_audit.json")
    smoke_count = config["smoke_per_condition"]
    if production and smoke_count is not None:
        raise ValueError("Production must use the complete unchanged assignment")
    selected = gen.expected_selection(plan, audit, smoke_count)
    saved_selected = _read_json(out / "selected_records.json")
    if selected != saved_selected:
        raise ValueError("Selected records differ from the original production plan")
    if (not selected or config["selected_count"] != len(selected)
            or summary["utterances_per_set"] != len(selected)
            or (production and len(selected) != 3000)):
        raise ValueError("Selected/completed mixture counts do not agree")
    selected_by_id = {record["utterance_id"]: record for record in selected}
    if len(selected_by_id) != len(selected):
        raise ValueError("Duplicate selected utterance IDs")
    cached, rir_error = _checked_native_rirs(
        out, config, plan, selected, audit,
    )
    missing = [dict(rt60_ms=row["rt60_ms"], spacing=row["spacing"])
               for row in audit["archives"] if not row["valid"]]
    if (summary["rir_count"] != len(cached) // len(RATES)
            or summary["native_conditions_available"] != audit["valid_conditions"]
            or summary["missing_conditions"] != missing):
        raise ValueError("Completion marker differs from native RIR availability")
    records = [json.loads(line)
               for line in (out / "utterances.jsonl").read_text().splitlines()]
    by_key = {(rec["sample_rate"], rec["utterance_id"]): rec for rec in records}
    expected_keys = {(sr, uid) for sr in RATES for uid in selected_by_id}
    if (len(records) != 2 * len(selected) or len(by_key) != len(records)
            or set(by_key) != expected_keys):
        raise ValueError("Expected one record per selected utterance and sample rate")

    def check_record(rec):
        sr, uid = rec["sample_rate"], rec["utterance_id"]
        selection = selected_by_id[uid]
        for key in ("filename", "plan_index", "rt60_ms", "spacing", "microphone",
                    "sources", "assignment_phase", "utterance_id"):
            if rec[key] != selection[key]:
                raise ValueError("Audio record differs from selected plan: " + key)
        index = selection["plan_index"]
        filename = ids[index]
        if filename != rec["filename"]:
            raise ValueError("Record index differs from original NPZ order")
        row = rows[filename]
        source_paths = [args.wsj_root / row[f"s{side}_path"] for side in (1, 2)]
        speakers = [path.parent.name for path in source_paths]
        noise_path = args.noise_root / "tt" / filename
        if (rec["original_sources"] != [str(path.resolve()) for path in source_paths]
                or rec["original_noise"] != str(noise_path.resolve())
                or rec["speakers"] != speakers or speakers[0] == speakers[1]):
            raise ValueError("Incorrect original source/noise paths or speakers")
        originals = [common.read_audio(path, sr) for path in source_paths]
        n = min(map(len, originals))
        if rec["samples"] != n:
            raise ValueError("Signal length differs from WHAMR min mode")
        suffix = f"{sr // 1000}k_min"
        wsj_gains = scaling["scaling_wsjmix_" + suffix][index]
        speech_gain = float(scaling["scaling_wham_speech_" + suffix][index])
        noise_gain = float(scaling["scaling_wham_noise_" + suffix][index])
        start = int(scaling["speech_start_sample_16k"][index]) // (16000 // sr)
        for actual, expected in zip(rec["wsj_gains"], wsj_gains):
            _same_number(actual, expected, "WSJ source gain")
        if len(rec["wsj_gains"]) != 2 or rec["noise_start_sample"] != start:
            raise ValueError("Incorrect WSJ gains or noise start")
        _same_number(rec["speech_gain"], speech_gain, "WHAM speech gain")
        _same_number(rec["noise_gain"], noise_gain, "WHAM noise gain")
        expected, peaks, rir_paths = {}, [], []
        for side, original, gain, source_index in zip(
                (1, 2), originals, wsj_gains, rec["sources"]):
            # This repeats the documented Matlab-compatible rounding order,
            # independently of the synthesis function under validation.
            scaled = np.round(32768 * (original[:n] * gain)).astype(np.int16)
            source = scaled.astype(np.float64) / 32768 * speech_gain
            h, peak, rir_path = cached[
                sr, rec["rt60_ms"], rec["spacing"], rec["microphone"], source_index
            ]
            h = h.astype(np.float64)
            direct = np.zeros(n)
            if peak < n:
                direct[peak:] = source[:n - peak] * h[peak]
            expected[f"source{side}"] = source
            expected[f"direct{side}"] = direct
            expected[f"reverb{side}"] = fftconvolve(source, h)[:n]
            peaks.append(peak)
            rir_paths.append(rir_path)
        if rec["direct_peak_samples"] != peaks or rec["rir_paths"] != rir_paths:
            raise ValueError("Incorrect signed-peak reference or receiver RIR path")
        noise = common.read_audio(noise_path, sr)[start:start + n]
        if len(noise) != n:
            raise ValueError("Original WHAM noise is too short")
        expected["noise"] = noise * noise_gain
        expected["mix_clean"] = expected["reverb1"] + expected["reverb2"]
        expected["mix_both"] = expected["mix_clean"] + expected["noise"]
        maximum = max(float(np.max(np.abs(wave))) for wave in expected.values())
        joint_gain = min(1.0, 0.99 / max(maximum, 1e-12))
        _same_number(rec["joint_gain"], joint_gain, "common attenuation")
        for label, left, right in (
                ("sir_reverberant_db", "reverb1", "reverb2"),
                ("snr_reverberant_db", "mix_clean", "noise")):
            ratio = common.power_ratio(expected[left], expected[right])
            if ratio is None:
                if rec[label] is not None:
                    raise ValueError("Incorrect silent-signal power ratio")
            else:
                _same_number(rec[label], ratio, label)
        if set(rec["paths"]) != SIGNALS:
            raise ValueError("Missing or extra output signal types")
        stored, errors = {}, []
        for label in sorted(SIGNALS):
            path = _output_path(rec["paths"][label], out)
            wanted_path = out / "audio" / str(sr) / label / filename
            if path.resolve() != wanted_path:
                raise ValueError("Signal path differs from its rate/type/filename")
            wave = _float_audio(path, sr, n)
            if np.max(np.abs(wave)) > 0.990001:
                raise ValueError("Signal exceeds the common peak limit")
            errors.append(float(np.max(np.abs(wave - expected[label] * joint_gain))))
            stored[label] = wave
        errors.extend((
            float(np.max(np.abs(stored["mix_clean"] - stored["reverb1"]
                                - stored["reverb2"]))),
            float(np.max(np.abs(stored["mix_both"] - stored["mix_clean"]
                                - stored["noise"]))),
        ))
        partner = by_key[16000 if sr == 8000 else 8000, uid]
        for key in ("filename", "plan_index", "rt60_ms", "spacing", "microphone",
                    "sources", "assignment_phase", "speakers", "original_sources",
                    "original_noise"):
            if rec[key] != partner[key]:
                raise ValueError(
                    "Source/receiver assignment differs across sample rates"
                )
        if abs(2 * by_key[8000, uid]["samples"] - by_key[16000, uid]["samples"]) > 1:
            raise ValueError("Paired sample-rate lengths differ")
        worst = max(errors)
        if not math.isfinite(worst) or worst > TOLERANCE:
            raise ValueError(f"Signal reconstruction failed for {uid}/{sr}: {worst}")
        return worst

    with ThreadPoolExecutor(max_workers=workers) as pool:
        worst = max(pool.map(check_record, records))
    _check_scps(out, by_key, selected, summary)
    counts = Counter(f"{record['rt60_ms']}:{record['spacing']}" for record in selected)
    report = dict(
        state="validated_production" if production else "validated_smoke",
        is_production_dataset=production, mode=mode,
        target_utterances_per_set=3000, utterances_per_set=len(selected), sets=4,
        audio_tuples_checked=len(records), rir_count=len(cached),
        recording_condition_counts=dict(sorted(counts.items())),
        max_native_rir_error=rir_error, max_signal_identity_error=worst,
        checks=("original CSV/NPZ and immutable plan; native archive hashes and "
                "MAT metadata; selected microphone and complete 10-second RIR; "
                "independent source quantization/convolution/direct/noise "
                "reconstruction; independently calculated joint attenuation; "
                "paired rates; all four sets and every SCP mapping"),
    )
    # Publish a success report only after every check passes.
    (out / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report
