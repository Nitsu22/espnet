#!/usr/bin/env python3
"""Generate paired 8/16-kHz WHAMR evaluation audio with measured MIRD RIRs.

Production requires all nine native recording conditions and the approved
2,925-case coverage plus 75 additions. Explicit smoke mode uses a small subset
of the unchanged plan and never writes a production completion marker.
"""

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

import create_whamr_but_dump as common
from create_whamr_but_2spk_dump import scp_references
from mird_audio_rirs import prepare_rirs
import mird_pair_assignment as assignment
from mird_rir import archive_stem, audit_dataset


RATES = (8000, 16000)
CONDITIONS = ("clean", "noisy")
SIGNALS = ("source1", "source2", "direct1", "direct2", "reverb1", "reverb2",
           "noise", "mix_clean", "mix_both")
RESERVE = 100 * 1024**3


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def set_name(rate, condition):
    return f"tt_mird_2spk_{condition}_reverb_min_{rate // 1000}k"


def process_rir(wave, fs, rate):
    """Keep the complete response, its delay and its signed peak."""
    wave = np.asarray(wave, dtype=np.float64)
    if (wave.ndim != 1 or not len(wave) or not np.all(np.isfinite(wave))
            or not np.any(wave)):
        raise ValueError("Expected a finite, nonzero mono RIR")
    if any(not isinstance(x, Integral) or isinstance(x, bool) or x <= 0
           for x in (fs, rate)):
        raise ValueError("Sample rates must be positive integers")
    if fs == rate:
        h = wave.copy()
    else:
        divisor = math.gcd(fs, rate)
        h = resample_poly(wave, rate // divisor, fs // divisor)
    gain = float(np.max(np.abs(h)))
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Resampled RIR is silent or nonfinite")
    h = (h / gain).astype(np.float32).astype(np.float64)
    return h, int(np.argmax(np.abs(h))), gain


def synthesize_signals(originals, noise, h_pair, wsj_gains, speech_gain,
                       noise_gain, noise_start):
    """Apply the original WHAM gains and one attenuation to all nine targets."""
    if len(originals) != 2 or len(h_pair) != 2 or len(wsj_gains) != 2:
        raise ValueError("Expected two speech sources, gains and RIRs")
    originals = [np.asarray(x, dtype=np.float64) for x in originals]
    noise = np.asarray(noise, dtype=np.float64)
    h_pair = [np.asarray(x, dtype=np.float64) for x in h_pair]
    if any(x.ndim != 1 or not len(x) or not np.all(np.isfinite(x))
           for x in originals + h_pair):
        raise ValueError("Nonfinite, empty or multichannel synthesis input")
    if any(not np.any(h) for h in h_pair):
        raise ValueError("Silent source RIR")
    gains = np.asarray([*wsj_gains, speech_gain, noise_gain], dtype=np.float64)
    if not np.all(np.isfinite(gains)) or np.any(gains < 0):
        raise ValueError("Expected finite nonnegative WHAM gains")
    length = min(map(len, originals))
    if (not isinstance(noise_start, Integral) or isinstance(noise_start, bool)
            or noise_start < 0 or noise.ndim != 1
            or not np.all(np.isfinite(noise))
            or noise_start + length > len(noise)):
        raise ValueError("Invalid WHAM noise window")
    signals, peaks = {}, []
    for side, (original, h, gain) in enumerate(zip(originals, h_pair, wsj_gains), 1):
        source = common.quantize(original[:length] * gain) * speech_gain
        peak = int(np.argmax(np.abs(h)))
        direct = np.zeros(length)
        if peak < length:
            direct[peak:] = source[:length - peak] * h[peak]
        signals[f"source{side}"] = source
        signals[f"direct{side}"] = direct
        signals[f"reverb{side}"] = fftconvolve(source, h)[:length]
        peaks.append(peak)
    signals["noise"] = noise[noise_start:noise_start + length] * noise_gain
    signals["mix_clean"] = signals["reverb1"] + signals["reverb2"]
    signals["mix_both"] = signals["mix_clean"] + signals["noise"]
    maximum = max(float(np.max(np.abs(x))) for x in signals.values())
    joint_gain = min(1.0, 0.99 / max(maximum, 1e-12))
    return {k: x * joint_gain for k, x in signals.items()}, joint_gain, peaks


def load_inputs(args):
    """Validate the complete immutable assignment and WHAM source indices."""
    plan = json.loads(Path(args.plan).read_text())
    if (plan.get("schema") != 2 or plan.get("count") != 3000
            or plan.get("artifact_type") != "mird_two_speaker_assignment_plan"):
        raise ValueError("Expected a schema-2 complete MIRD assignment plan")
    assignment.audit_assignment(plan["records"], require_production=True)
    expected_grid = [dict(distance_m=d, angle_deg=a, native_azimuth_deg=a % 360)
                     for d in (1, 2) for a in range(-90, 91, 15)]
    if plan["source_grid"] != expected_grid:
        raise ValueError(
            "Plan must explicitly bind signed angles to native MAT azimuths"
        )
    if plan["assignment_source_sha256"] != common.digest(Path(assignment.__file__)):
        raise ValueError("Assignment implementation changed; regenerate a new plan")
    for key in ("filenames", "scaling"):
        path = Path(getattr(args, key)).resolve()
        if plan["source_index_sha256"].get(str(path)) != common.digest(path):
            raise ValueError("Plan source-index hash mismatch: " + key)
    with Path(args.filenames).open() as stream:
        source_rows = list(csv.DictReader(stream))
    rows = {row["output_filename"]: row for row in source_rows}
    with np.load(args.scaling, allow_pickle=True) as data:
        scaling = {key: data[key] for key in data.files}
    ids = [str(uid) for uid in scaling["utterance_id"]]
    if (len(ids) != 3000 or len(set(ids)) != 3000 or len(rows) != len(source_rows)
            or set(ids) != set(rows)
            or ids != [row["filename"] for row in plan["records"]]):
        raise ValueError("Expected the same 3,000 ordered WHAM test source IDs")
    for filename, record in zip(ids, plan["records"]):
        row = rows[filename]
        sources = [row["s1_path"], row["s2_path"]]
        if (record["source_paths"] != sources
                or record["utterance_id"] != Path(filename).stem
                or Path(sources[0]).parent.name == Path(sources[1]).parent.name
                or Path(filename).name != filename):
            raise ValueError("Plan/CSV source or speaker mismatch: " + filename)
    for rate in RATES:
        suffix = f"{rate // 1000}k_min"
        for name, shape in (("scaling_wsjmix_", (3000, 2)),
                            ("scaling_wham_speech_", (3000,)),
                            ("scaling_wham_noise_", (3000,))):
            values = np.asarray(scaling[name + suffix])
            if (values.shape != shape or not np.all(np.isfinite(values))
                    or np.any(values < 0)):
                raise ValueError("Invalid WHAM scaling array: " + name + suffix)
    starts = np.asarray(scaling["speech_start_sample_16k"])
    if (starts.shape != (3000,) or not np.all(np.isfinite(starts))
            or np.any(starts < 0) or np.any(starts != starts.astype(np.int64))):
        raise ValueError("Invalid WHAM noise start indices")
    return rows, scaling, ids, plan


def expected_selection(plan, audit, smoke_per_condition):
    """Keep the production assignment unchanged when selecting smoke entries."""
    expected = {(rt, spacing) for rt in assignment.RT60_MS
                for spacing in assignment.SPACINGS}
    reports = audit["archives"]
    if (len(reports) != 9
            or {(x["rt60_ms"], x["spacing"]) for x in reports} != expected):
        raise ValueError("Native audit must account for all nine recording conditions")
    available = {(x["rt60_ms"], x["spacing"]) for x in reports if x["valid"]}
    if smoke_per_condition is None:
        if available != expected or not audit["all_conditions_valid"]:
            raise ValueError(
                "Production requires all nine valid native MIRD conditions"
            )
        return [dict(record, plan_index=i) for i, record in enumerate(plan["records"])]
    if (not isinstance(smoke_per_condition, Integral)
            or isinstance(smoke_per_condition, bool) or smoke_per_condition < 1):
        raise ValueError("smoke-per-condition must be a positive integer")
    if not available:
        raise ValueError(
            "No valid native MIRD conditions available for smoke generation"
        )
    counts, selected = Counter(), []
    for index, record in enumerate(plan["records"]):
        key = record["rt60_ms"], record["spacing"]
        if key in available and counts[key] < smoke_per_condition:
            selected.append(dict(record, plan_index=index))
            counts[key] += 1
    if any(counts[key] != smoke_per_condition for key in available):
        raise ValueError("Not enough planned entries per available smoke condition")
    return selected


def checked_audit(args):
    """Reuse full native inspection only if every archive's bytes still match."""
    if args.audit_report is None:
        return audit_dataset(args.archives)
    report = json.loads(args.audit_report.read_text())
    for item in report["archives"]:
        path = args.archives / (archive_stem(item["rt60_ms"], item["spacing"]) + ".zip")
        if (str(path.resolve()) != item["path"]
                or item.get("sha256") != common.digest(path)):
            raise ValueError("Native archive changed; run a new complete source audit")
    return report


def generate(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Refusing existing output: " + str(out))
    rows, scaling, ids, plan = load_inputs(args)
    audit = checked_audit(args)
    selected = expected_selection(plan, audit, args.smoke_per_condition)
    mode = "production" if args.smoke_per_condition is None else "smoke"
    samples, headers = 0, {}
    for record in selected:
        row = rows[record["filename"]]
        sources = [args.wsj_root / row[f"s{side}_path"] for side in (1, 2)]
        for path in sources + [args.noise_root / "tt" / record["filename"]]:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in sources:
            if path not in headers:
                headers[path] = sf.info(path)
        for rate in RATES:
            samples += min(math.ceil(headers[p].frames * rate / headers[p].samplerate)
                           for p in sources)
    # Include the maximum complete RIR cache and leave 100 GiB free.
    estimated = int((samples * 4 * len(SIGNALS) + 2 * 1024**3) * 1.1)
    if not out.parent.is_dir():
        raise FileNotFoundError(
            "Create the output parent directory first: " + str(out.parent)
        )
    if shutil.disk_usage(out.parent).free < RESERVE + estimated:
        raise OSError("Insufficient capacity plus 100 GiB disk reserve")
    out.mkdir()
    save_json(out / "capacity.json",
              dict(estimated_bytes=estimated, reserve_bytes=RESERVE))
    shutil.copyfile(args.plan, out / "assignment_plan.json")
    save_json(out / "selected_records.json", selected)
    save_json(out / "native_rir_audit.json", audit)
    cached, rir_meta = prepare_rirs(args.archives, selected, plan["source_grid"],
                                    out, RATES, process_rir)
    save_json(out / "rir_metadata.json", rir_meta)
    config = dict(
        schema=1, artifact_type="mird_two_speaker_audio_dump", mode=mode,
        target_count=3000, selected_count=len(selected),
        smoke_per_condition=args.smoke_per_condition, workers=args.workers,
        sample_rates=list(RATES), conditions=list(CONDITIONS),
        filenames=str(args.filenames.resolve()),
        filenames_sha256=common.digest(args.filenames),
        scaling=str(args.scaling.resolve()), scaling_sha256=common.digest(args.scaling),
        wsj_root=str(args.wsj_root.resolve()),
        noise_root=str(args.noise_root.resolve()),
        archives_root=str(args.archives.resolve()),
        plan_sha256=common.digest(out / "assignment_plan.json"),
        native_audit_sha256=common.digest(out / "native_rir_audit.json"),
        selected_records_sha256=common.digest(out / "selected_records.json"),
        rir_metadata_sha256=common.digest(out / "rir_metadata.json"),
        implementation_sha256={p.name: common.digest(p) for p in (
            Path(__file__), Path(common.__file__), Path(assignment.__file__),
            Path(__file__).with_name("mird_audio_rirs.py"),
            Path(__file__).with_name("mird_rir.py"),
            Path(__file__).with_name("mird_audio_validation.py"),
            Path(__file__).with_name("create_whamr_but_2spk_dump.py"),
        )},
        rir=("full native ten-second response; retain delay; "
             "resample then abs-peak normalize"),
        direct=("one signed absolute-peak tap; "
                "not a separately measured physical direct onset"),
        gains=("WHAM WSJ gains then int16 quantization then shared speech gain; "
               "original noise gain/crop"),
        joint_gain=("one attenuation for all nine signals per rate, "
                    "shared by clean/noisy"),
        tail="full RIR convolution; crop observed speech to original WHAMR min length",
    )
    save_json(out / "generation_config.json", config)

    def make_utterance(record):
        if shutil.disk_usage(out).free < RESERVE:
            raise OSError("Disk reserve reached; generation incomplete")
        index, filename = record["plan_index"], record["filename"]
        row = rows[filename]
        source_paths = [args.wsj_root / row[f"s{side}_path"] for side in (1, 2)]
        noise_path = args.noise_root / "tt" / filename
        results = []
        for rate in RATES:
            key = f"{rate // 1000}k_min"
            rirs = [cached[rate, record["rt60_ms"], record["spacing"],
                           record["microphone"], source]
                    for source in record["sources"]]
            originals = [common.read_audio(p, rate) for p in source_paths]
            gains = [float(x) for x in scaling["scaling_wsjmix_" + key][index]]
            speech_gain = float(scaling["scaling_wham_speech_" + key][index])
            noise_gain = float(scaling["scaling_wham_noise_" + key][index])
            start = int(scaling["speech_start_sample_16k"][index]) // (16000 // rate)
            signals, joint_gain, peaks = synthesize_signals(
                originals, common.read_audio(noise_path, rate), [x[0] for x in rirs],
                gains, speech_gain, noise_gain, start,
            )
            paths = {key: common.write_audio(
                out / "audio" / str(rate) / key / filename, wave, rate
            ) for key, wave in signals.items()}
            results.append(dict(
                utterance_id=record["utterance_id"], filename=filename,
                plan_index=index,
                sample_rate=rate, samples=len(signals["noise"]),
                rt60_ms=record["rt60_ms"], spacing=record["spacing"],
                microphone=record["microphone"], sources=record["sources"],
                assignment_phase=record["assignment_phase"],
                speakers=[p.parent.name for p in source_paths],
                original_sources=[str(p.resolve()) for p in source_paths],
                original_noise=str(noise_path.resolve()),
                rir_paths=[x[2] for x in rirs],
                direct_peak_samples=peaks, wsj_gains=gains, speech_gain=speech_gain,
                noise_gain=noise_gain, noise_start_sample=start, joint_gain=joint_gain,
                paths=paths,
                sir_reverberant_db=common.power_ratio(
                    signals["reverb1"], signals["reverb2"]),
                snr_reverberant_db=common.power_ratio(
                    signals["mix_clean"], signals["noise"]),
            ))
        return results

    manifests = defaultdict(lambda: defaultdict(list))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with (out / "utterances.jsonl").open("w") as stream:
            for count, records in enumerate(pool.map(make_utterance, selected), 1):
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    uid, rate = record["utterance_id"], record["sample_rate"]
                    for condition in CONDITIONS:
                        files = manifests[set_name(rate, condition)]
                        for key, path in scp_references(record, condition).items():
                            files[key].append((uid, path))
                        for key, value in (("utt2spk", uid), ("spk2utt", uid),
                                           ("utt2spk1", record["speakers"][0]),
                                           ("utt2spk2", record["speakers"][1]),
                                           ("utt2num_samples", str(record["samples"]))):
                            files[key].append((uid, value))
                if count % 100 == 0 or count == len(selected):
                    print(f"Generated {count}/{len(selected)} {mode} entries "
                          "at both rates", flush=True)
    for name, files in manifests.items():
        folder = out / "raw" / name
        folder.mkdir(parents=True)
        for key, entries in files.items():
            (folder / key).write_text("".join(f"{uid} {value}\n"
                                             for uid, value in sorted(entries)))
        (folder / "feats_type").write_text("raw\n")
    summary = dict(mode=mode, target_utterances_per_set=3000,
                   utterances_per_set=len(selected), sets=sorted(manifests),
                   rir_count=len(rir_meta) // len(RATES),
                   native_conditions_available=audit["valid_conditions"],
                   missing_conditions=[dict(rt60_ms=x["rt60_ms"], spacing=x["spacing"])
                                       for x in audit["archives"] if not x["valid"]])
    marker = ("generation_complete.json" if mode == "production"
              else "smoke_complete.json")
    save_json(out / marker, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--wsj-root", type=Path,
                        default=common.WHAMR / "enh_rir/data/wsj0/wsj0_wav")
    parser.add_argument("--noise-root", type=Path,
                        default=Path("/net/midgar/work2/nitsu/data/wsj/wham_noise"))
    parser.add_argument(
        "--filenames", type=Path, default=common.WHAMR
        / "enh_rir/whamr_scripts/data/mix_2_spk_filenames_tt.csv",
    )
    parser.add_argument(
        "--scaling", type=Path,
        default=Path("/net/midgar/work2/nitsu/data/wsj/wham_noise/metadata")
        / "scaling_tt.npz",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-per-condition", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.smoke_per_condition is not None
                            and args.smoke_per_condition < 1):
        parser.error("workers and smoke-per-condition must be positive")
    if not args.validate_only:
        if args.plan is None or args.archives is None:
            parser.error("generation requires --plan and --archives")
        generate(args)
    from mird_audio_validation import validate

    validate(args.output, args.workers)


if __name__ == "__main__":
    main()
