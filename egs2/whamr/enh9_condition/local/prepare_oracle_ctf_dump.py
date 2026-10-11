"""Verify baseline/RIR correspondence, then publish a dedicated Oracle CTF dump.

Audio is referenced by new absolute-path SCPs, never copied or regenerated.
Limited runs are marked as smoke dumps and are rejected for full training.
Relocation retains original RIR provenance and verifies exact baseline WAV bytes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from espnet2.rir.oracle_ctf import OracleCTFConfig, OracleCTFFitter, pack_ctf

COUNTS = {"tr": 20000, "cv": 5000, "tt": 3000}
AUDIO_NAMES = ("wav.scp", "spk1.scp", "spk2.scp")
SCHEMA_VERSION = 1
PCM16_TOLERANCE = 1 / 32768 + 1.0e-10
FLOAT_TOLERANCE = 1.0e-7


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_scp(path):
    result = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result or fields[1].endswith("|"):
            raise ValueError(f"Malformed/duplicate/non-file SCP entry: {path}:{number}")
        result[fields[0]] = fields[1]
    return result


def resolve(value, recipe):
    path = Path(value)
    return (path if path.is_absolute() else recipe / path).resolve(strict=True)


def load_wave(path):
    raw = path.read_bytes()
    with sf.SoundFile(io.BytesIO(raw)) as stream:
        metadata = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                        frames=len(stream), sample_rate=stream.samplerate,
                        channels=stream.channels, subtype=stream.subtype)
        wave = stream.read(dtype="float64", always_2d=True)
    if (metadata["sample_rate"] != 8000 or metadata["channels"] not in (1, 2)
            or not len(wave) or not np.isfinite(wave).all()):
        raise ValueError(f"Expected finite 8-kHz mono/stereo WAV: {path}")
    return wave[:, 0], metadata


def compare_audio(baseline, candidate, subtype):
    """Explain PCM16 clipping/quantization without fitting gain or delay."""
    if baseline.shape != candidate.shape:
        raise ValueError(f"Length mismatch: {len(baseline)} != {len(candidate)}")
    if subtype == "PCM_16":
        expected = np.clip(np.floor(candidate * 32768), -32768, 32767) / 32768
        tolerance = PCM16_TOLERANCE
        kind = "pcm16_floor_clip_with_one_lsb_float_rounding_tolerance"
    elif subtype == "FLOAT":
        expected = candidate.astype(np.float32).astype(np.float64)
        tolerance = FLOAT_TOLERANCE
        kind = "float32"
    elif subtype == "DOUBLE":
        expected, tolerance, kind = candidate, 1.0e-10, "float64"
    else:
        raise ValueError(f"Unsupported baseline encoding for correspondence: {subtype}")
    residual = np.abs(baseline - expected)
    report = dict(encoding=kind, tolerance=tolerance,
                  maximum_residual=float(residual.max()),
                  rms_residual=float(np.sqrt(np.mean(residual ** 2))),
                  different_samples=int(np.count_nonzero(residual > 1.0e-12)),
                  clipped_samples=int(np.count_nonzero(
                      (candidate < -1) | (candidate > 32767 / 32768))),
                  maximum_float_difference=float(np.max(np.abs(baseline - candidate))))
    if report["maximum_residual"] > tolerance:
        raise ValueError("Audio is not explained by encoding quantization/clipping: "
                         + json.dumps(report, sort_keys=True))
    return report


def catalogue(baseline_dump, baseline_recipe, baseline_reverb_data,
              rir_dump, rir_recipe, limit, expected_counts=COUNTS):
    records, source_indices = [], {}
    for split, expected in expected_counts.items():
        dataset = f"{split}_mix_clean_reverb_min_8k"
        folder = baseline_dump / "raw" / dataset
        rir_folder = rir_dump / "raw" / f"{split}_rir_2spk_nf_min_8k"
        maps = {}
        for name in AUDIO_NAMES:
            path = folder / name
            maps[name] = read_scp(path)
            source_indices[str(path.resolve())] = digest(path)
        for speaker in (1, 2):
            name = f"spk{speaker}_reverb.scp"
            path = baseline_reverb_data / dataset / name
            maps[name] = read_scp(path)
            source_indices[str(path.resolve())] = digest(path)
        ids = sorted(maps["wav.scp"])
        if len(ids) != expected or any(set(values) != set(ids) for values in maps.values()):
            raise ValueError(f"Incomplete or mismatched baseline {split} indexes")
        rir_maps = {}
        for name in ("wav.scp", "speech_direct1.scp", "speech_direct2.scp",
                     "speech_reverb1.scp", "speech_reverb2.scp", "rir_direct1.scp",
                     "rir_direct2.scp", "rir_ref1.scp", "rir_ref2.scp"):
            path = rir_folder / name
            rir_maps[name] = read_scp(path)
            source_indices[str(path.resolve())] = digest(path)
        if any(set(values) != set(rir_maps["wav.scp"]) for values in rir_maps.values()):
            raise ValueError(f"Mismatched RIR indexes: {split}")
        for uid in ids[:limit] if limit is not None else ids:
            rir_id = Path(maps["spk1_reverb.scp"][uid]).stem
            if Path(maps["spk2_reverb.scp"][uid]).stem != rir_id or rir_id not in rir_maps["wav.scp"]:
                raise ValueError(f"Baseline/RIR utterance identity mismatch: {split}/{uid}")
            records.append(dict(split=split, uid=uid, rir_id=rir_id,
                                baseline={name: str(resolve(value[uid], baseline_recipe))
                                          for name, value in maps.items()},
                                rir={name: str(resolve(value[rir_id], rir_recipe))
                                     for name, value in rir_maps.items()}))
    return records, source_indices


def verify_record(record):
    reports, baseline_audio, source_audio = {}, {}, {}
    candidate_names = ("wav.scp", "speech_direct1.scp", "speech_direct2.scp",
                       "speech_reverb1.scp", "speech_reverb2.scp")
    length = None
    for name, candidate_name in zip((*AUDIO_NAMES, "spk1_reverb.scp", "spk2_reverb.scp"),
                                    candidate_names):
        baseline, header = load_wave(Path(record["baseline"][name]))
        candidate, candidate_header = load_wave(Path(record["rir"][candidate_name]))
        if length is None:
            length = len(baseline)
        if len(baseline) != length or length <= 128:
            raise ValueError(f"Unaligned or too-short baseline teacher: {record['uid']}/{name}")
        try:
            reports[name] = compare_audio(baseline, candidate, header["subtype"])
        except ValueError as error:
            raise ValueError(f"{record['split']}/{record['uid']}/{name}: {error}") from error
        if name in AUDIO_NAMES:
            baseline_audio[name] = header
        source_audio[header["path"]] = header["sha256"]
        source_audio[candidate_header["path"]] = candidate_header["sha256"]
    return reports, baseline_audio, source_audio


def write_indexes(folder, records, output):
    folder.mkdir(parents=True)
    audio_root = Path(os.path.commonpath([
        str(Path(record["baseline"][name]).parent)
        for record in records for name in AUDIO_NAMES]))
    if audio_root == Path(audio_root.anchor):
        raise ValueError("Baseline audio lacks a shared data directory")
    (folder / "audio").symlink_to(audio_root, target_is_directory=True)
    final_audio = output / "raw" / folder.name / "audio"
    for name in AUDIO_NAMES:
        (folder / name).write_text("".join(
            f"{record['uid']} {final_audio / Path(record['baseline'][name]).relative_to(audio_root)}\n"
            for record in records))
    (folder / "rir_ctf.scp").write_text("".join(
        f"{r['uid']} {output / 'features' / r['split'] / (r['uid'] + '.npy')}\n"
        for r in records))
    (folder / "rir_ctf_shape").write_text("".join(f"{r['uid']} 60\n" for r in records))
    (folder / "utt2num_samples").write_text("".join(
        f"{r['uid']} {r['audio']['wav.scp']['frames']}\n" for r in records))
    (folder / "utt2spk").write_text("".join(f"{r['uid']} {r['uid']}\n" for r in records))
    (folder / "spk2utt").write_text("".join(f"{r['uid']} {r['uid']}\n" for r in records))
    (folder / "feats_type").write_text("raw\n")
    (folder / "channel").write_text("0\n")
    return {str(Path("raw") / folder.name / "audio"): str(audio_root)}


def _output_hashes(output):
    return {str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()
            and path.name != "preparation.json"}


def _validate_manifest(report, require_full):
    if (report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "complete"
            or report["spec"].get("method") != "direct_reverb_sweep_ridge_ls"
            or report["spec"].get("tap_order") != "causal"):
        raise ValueError("Unrecognized or incomplete Oracle CTF dump")
    native = OracleCTFConfig(regularization=report["spec"]["config"]["regularization"]).as_dict()
    if (report["spec"]["config"] != native or report["spec"].get("shape") != [60, 2, 129, 2]
            or report["spec"].get("dtype") != "float32"):
        raise ValueError("Oracle CTF grid differs from native NF8K estimator")
    if require_full and (report["spec"].get("limit") is not None or report["counts"] != COUNTS):
        raise ValueError("A limited/smoke Oracle CTF dump cannot be used for full training")
    if report["counts"] != {split: len(examples) for split, examples in report["baseline_audio"].items()}:
        raise ValueError("Oracle manifest example counts disagree")
    expected_links = {str(Path("raw") / f"{split}_mix_clean_reverb_min_8k" / "audio")
                      for split in report["counts"]}
    if set(report.get("audio_links", {})) != expected_links:
        raise ValueError("Oracle audio-link manifest is incomplete")


def validate_prepared_dump(output, require_full=True):
    output = Path(output)
    marker = output / "preparation.json"
    if not marker.is_file():
        raise ValueError("Oracle CTF dump has no complete preparation manifest")
    report = json.loads(marker.read_text())
    _validate_manifest(report, require_full)
    for name, target in report["audio_links"].items():
        link = output / name
        if not link.is_symlink() or os.readlink(link) != target or not link.is_dir():
            raise ValueError(f"Baseline audio symlink changed: {name}")
    actual_files = {str(path.relative_to(output)) for path in output.rglob("*")
                    if path.is_file() and path.name != "preparation.json"}
    if actual_files != set(report["output_hashes"]):
        raise ValueError("Oracle dump file inventory changed")
    for name, expected in report["output_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Oracle dump checksum changed: {name}")
    for split, examples in report["baseline_audio"].items():
        folder = output / "raw" / f"{split}_mix_clean_reverb_min_8k"
        maps = {name: read_scp(folder / name) for name in AUDIO_NAMES}
        ctf_map = read_scp(folder / "rir_ctf.scp")
        shape_map = read_scp(folder / "rir_ctf_shape")
        if (any(set(values) != set(examples) for values in maps.values())
                or set(ctf_map) != set(examples) or set(shape_map) != set(examples)):
            raise ValueError(f"Baseline index identity changed: {split}")
        for uid, files in examples.items():
            expected_feature = output / "features" / split / (uid + ".npy")
            if ctf_map[uid] != str(expected_feature) or shape_map[uid] != "60":
                raise ValueError(f"Oracle feature reference changed: {split}/{uid}")
            with expected_feature.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported Oracle array version: {version}")
            if shape != (60, 2, 129, 2) or dtype != np.dtype("float32"):
                raise ValueError(f"Oracle feature shape/dtype changed: {split}/{uid}")
            for name in AUDIO_NAMES:
                path = Path(maps[name][uid])
                if not path.is_absolute() or digest(path) != files[name]["sha256"]:
                    raise ValueError(f"Exact baseline audio changed: {split}/{uid}/{name}")
    return report


def prepare(baseline_dump, baseline_recipe, baseline_reverb_data, rir_dump,
            rir_recipe, output, mode="verify", limit=None, workers=4,
            config=OracleCTFConfig(), expected_counts=COUNTS):
    if limit is not None and limit <= 0 or workers <= 0:
        raise ValueError("limit/workers must be positive")
    paths = [Path(path).resolve() for path in (baseline_dump, baseline_recipe,
             baseline_reverb_data, rir_dump, rir_recipe, output)]
    baseline_dump, baseline_recipe, baseline_reverb_data, rir_dump, rir_recipe, output = paths
    records, source_indices = catalogue(baseline_dump, baseline_recipe,
        baseline_reverb_data, rir_dump, rir_recipe, limit, expected_counts)
    spec = dict(method="direct_reverb_sweep_ridge_ls", config=config.as_dict(),
                tap_order="causal", shape=[config.taps, 2, config.n_fft // 2 + 1, 2],
                dtype="float32", limit=limit, baseline_dump=str(baseline_dump),
                baseline_recipe=str(baseline_recipe), baseline_reverb_data=str(baseline_reverb_data),
                rir_dump=str(rir_dump), rir_recipe=str(rir_recipe),
                correspondence=dict(pcm16_tolerance=PCM16_TOLERANCE,
                                    float_tolerance=FLOAT_TOLERANCE))
    if output.exists():
        previous = validate_prepared_dump(output, require_full=limit is None)
        if previous["spec"] != spec or previous["source_index_hashes"] != source_indices:
            raise ValueError("Oracle computation/source configuration changed; use a new dump")
        for path, expected in previous["source_audio_hashes"].items():
            if digest(path) != expected:
                raise ValueError(f"Oracle source audio changed: {path}")
        print("Existing immutable Oracle CTF dump verified", flush=True)
        return previous
    baseline_audio, source_audio, correspondence = {}, {}, {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record, (metrics, audio, hashes) in zip(records, executor.map(verify_record, records)):
            split, uid = record["split"], record["uid"]
            baseline_audio.setdefault(split, {})[uid] = audio
            correspondence.setdefault(split, {})[uid] = metrics
            source_audio.update(hashes)
            record["audio"] = audio
            if len(baseline_audio[split]) % 1000 == 0:
                print(f"Verified correspondence {split}: {len(baseline_audio[split])}", flush=True)
    counts = {split: len(examples) for split, examples in baseline_audio.items()}
    report = dict(schema_version=SCHEMA_VERSION, status="verified", spec=spec,
                  counts=counts, source_index_hashes=source_indices,
                  source_audio_hashes=source_audio, baseline_audio=baseline_audio,
                  correspondence=correspondence)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == "verify":
        verification = output.with_name(output.name + ".verification.json")
        if verification.exists():
            if json.loads(verification.read_text()) != report:
                raise ValueError("Existing correspondence verification changed; use a new output")
        else:
            verification.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Correspondence verified: {counts}; report {verification}", flush=True)
        return report
    temp = Path(tempfile.mkdtemp(prefix=".oracle_ctf_", dir=output.parent))
    try:
        # Fitting is intentionally after ALL selected correspondence checks.
        fitter = OracleCTFFitter(config)
        reconstruction = {}
        def fit_record(record):
            coefficients, metrics = [], []
            hashes = {}
            for speaker in (1, 2):
                direct, direct_header = load_wave(Path(record["rir"][f"rir_direct{speaker}.scp"]))
                reverb, reverb_header = load_wave(Path(record["rir"][f"rir_ref{speaker}.scp"]))
                hashes[direct_header["path"]] = direct_header["sha256"]
                hashes[reverb_header["path"]] = reverb_header["sha256"]
                ctf, metric = fitter.fit(direct, reverb)
                coefficients.append(ctf)
                metrics.append(metric)
            path = temp / "features" / record["split"] / (record["uid"] + ".npy")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, pack_ctf(coefficients), allow_pickle=False)
            return metrics, hashes
        # The fitter holds only immutable CPU sweep/window tensors. Each worker
        # owns its response/LS arrays and feature file; reports update on main.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (record, (metrics, hashes)) in enumerate(
                    zip(records, executor.map(fit_record, records)), 1):
                source_audio.update(hashes)
                reconstruction.setdefault(record["split"], {})[record["uid"]] = metrics
                if index % 100 == 0:
                    print(f"Computed Oracle CTF {index}/{len(records)}", flush=True)
        audio_links = {}
        for split in counts:
            selected = [r for r in records if r["split"] == split]
            audio_links.update(write_indexes(
                temp / "raw" / f"{split}_mix_clean_reverb_min_8k", selected, output))
        report.update(status="complete", reconstruction=reconstruction,
                      audio_links=audio_links, output_hashes=_output_hashes(temp))
        (temp / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        temp.rename(output)
        print(f"Published Oracle CTF dump: {output}, {counts}", flush=True)
        return report
    except BaseException:
        shutil.rmtree(temp)
        raise


def relocate(source_dump, baseline_dump, baseline_recipe, output, workers=4):
    """Publish imported features with exact local baseline references and provenance."""
    source_dump, baseline_dump, baseline_recipe, output = [Path(p).resolve()
        for p in (source_dump, baseline_dump, baseline_recipe, output)]
    source_marker = source_dump / "preparation.json"
    source = json.loads(source_marker.read_text())
    _validate_manifest(source, require_full=source["spec"].get("limit") is None)
    for name, expected in source["output_hashes"].items():
        if digest(source_dump / name) != expected:
            raise ValueError(f"Imported Oracle dump checksum changed: {name}")
    if output.exists():
        previous = validate_prepared_dump(output, require_full=source["spec"]["limit"] is None)
        if previous.get("imported_manifest_sha256") != digest(source_marker):
            raise ValueError("Imported source changed; use a new output")
        return previous
    records = []
    for split, examples in source["baseline_audio"].items():
        folder = baseline_dump / "raw" / f"{split}_mix_clean_reverb_min_8k"
        maps = {name: read_scp(folder / name) for name in AUDIO_NAMES}
        for uid, files in examples.items():
            if any(uid not in values for values in maps.values()):
                raise ValueError(f"Remote baseline missing imported ID: {split}/{uid}")
            records.append(dict(split=split, uid=uid, audio=files,
                                baseline={name: str(resolve(values[uid], baseline_recipe))
                                          for name, values in maps.items()}))
    def verify(record):
        for name, metadata in record["audio"].items():
            if digest(record["baseline"][name]) != metadata["sha256"]:
                raise ValueError(f"Relocated baseline differs: {record['split']}/{record['uid']}/{name}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(verify, records))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".oracle_ctf_relocate_", dir=output.parent))
    try:
        shutil.copytree(source_dump / "features", temp / "features", symlinks=False)
        audio_links = {}
        for split in source["counts"]:
            audio_links.update(write_indexes(temp / "raw" / f"{split}_mix_clean_reverb_min_8k",
                          [r for r in records if r["split"] == split], output))
        provenance = temp / "provenance"
        provenance.mkdir()
        shutil.copy2(source_marker, provenance / "original_preparation.json")
        source["imported_manifest_sha256"] = digest(source_marker)
        source["spec"].update(baseline_dump=str(baseline_dump), baseline_recipe=str(baseline_recipe))
        source["audio_links"] = audio_links
        source["output_hashes"] = _output_hashes(temp)
        (temp / "preparation.json").write_text(json.dumps(source, indent=2) + "\n")
        temp.rename(output)
        print(f"Published portable Oracle CTF dump: {output}", flush=True)
        return source
    except BaseException:
        shutil.rmtree(temp)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dump", type=Path, default=Path("../enh7_baseline/dump_clean"))
    parser.add_argument("--baseline-recipe", type=Path, default=Path("../enh7_baseline"))
    parser.add_argument("--baseline-reverb-data", type=Path, default=Path("../enh4_clean/data"))
    parser.add_argument("--rir-dump", type=Path, default=Path("../rir_2spk/dump_nf_2spk_8k_min"))
    parser.add_argument("--rir-recipe", type=Path, default=Path("../rir_2spk"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify", "prepare", "relocate"), default="verify")
    parser.add_argument("--source-dump", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.mode == "relocate":
        if args.source_dump is None:
            parser.error("--source-dump is required for relocation")
        relocate(args.source_dump, args.baseline_dump, args.baseline_recipe, args.output, args.workers)
    else:
        prepare(args.baseline_dump, args.baseline_recipe, args.baseline_reverb_data,
                args.rir_dump, args.rir_recipe, args.output, args.mode, args.limit,
                args.workers, OracleCTFConfig(regularization=args.regularization))


if __name__ == "__main__":
    main()
