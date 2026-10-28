#!/usr/bin/env python3
"""Test-only CTF-input controls. Original separation mixture/references are fixed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

from check_same_speaker_evaluation import digest, read_index

# The historical Baseline did not save ESTOI. Compare only shared metrics;
# standard enh_scoring still saves ESTOI for the new conditions separately.
METRICS = ("SI_SNR", "SDR", "SAR", "SIR", "STOI")

CONDITIONS = {"same_rir_other_utt": "tt_same_rir_other_utt_8k",
              "same_utt_swapped_rir": "tt_same_utt_swapped_rir_8k"}
SUPPORTED_CONDITIONS = {**CONDITIONS, "same_utt_other_rir": "tt_same_utt_other_rir_8k"}
NAMES = ("wav.scp", "spk1.scp", "spk2.scp")
PIN = "728afa1b8d27d526ba87cb73ca48193ee08f91ee48023cc1f26d9f44bd8a990e"
ALLOWED = (Path("/net/midgar/work2/nitsu"), Path("/gs/bs/tga-shinoda/nitsu"))


def authorized(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Use an absolute output path")
    result = path.resolve()
    if not any(result.is_relative_to(p) and result != p for p in ALLOWED):
        raise ValueError("Output outside authorized personal storage")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_index(path, values):
    path.write_text("".join(f"{key} {value}\n" for key, value in sorted(values.items())))


def audio(path, expected=None, mono=False):
    info = sf.info(path)
    if (info.samplerate != 8000 or info.frames <= 128
            or info.channels not in ((1,) if mono else (1, 2))
            or (expected is not None and info.frames != expected)):
        raise ValueError(f"Invalid 8-kHz audio/length: {path}")
    return info.frames


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 8000 or not np.isfinite(wave).all():
        raise ValueError(f"Invalid waveform: {path}")
    return np.ascontiguousarray(wave[:, 0][None])


def verify_hash(item):
    path, checksum = item
    if digest(path) != checksum:
        raise ValueError(f"Pinned/transferred data changed: {path}")


def metric_values(directory, keys):
    result = {}
    for metric in METRICS:
        for speaker in (1, 2):
            name = f"{metric}_spk{speaker}"
            values = read_index(directory / name)
            if set(values) != set(keys):
                raise ValueError(f"Score key mismatch: {directory / name}")
            values = {key: float(value) for key, value in values.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Nonfinite scores: {directory / name}")
            result[name] = values
    return result


def selected_conditions(source, generation=False):
    """Old evaluation manifests retain their original two-condition meaning."""
    names = (source.get("condition_definitions") if generation
             else source.get("conditions", CONDITIONS))
    if not isinstance(names, dict) or not names or set(names) - set(SUPPORTED_CONDITIONS):
        raise ValueError("Unknown/empty conditioning intervention selection")
    result = {name: SUPPORTED_CONDITIONS[name] for name in names}
    if not generation and dict(names) != result:
        raise ValueError("Unexpected conditioning dataset names")
    return result


def validate_other_rir_audit(audits, keys):
    """Verify a paired-RIR derangement, including physical hash inequality."""
    donors = []
    for uid in keys:
        row = audits["same_utt_other_rir", uid]
        donor = row["donor_rir_uid"]
        if donor == uid or donor not in keys:
            raise ValueError("Wrong-RIR donor must be another original test scene")
        donors.append(donor)
        donor_row = audits["same_utt_other_rir", donor]
        source_t60, donor_t60 = float(row["source_t60"]), float(donor_row["target_t60"])
        if not math.isfinite(source_t60) or not math.isclose(source_t60, donor_t60):
            raise ValueError("Wrong-RIR RT60 metadata differs from the donor scene")
        if (row["source1"] != row["original_source1"]
                or row["source2"] != row["original_source2"]
                or (row["source1_rir_slot"], row["source2_rir_slot"]) != ("1", "2")):
            raise ValueError("Wrong-RIR intervention must retain original utterances")
        for k in (1, 2):
            for kind in ("anechoic", "reverb"):
                name = f"rir{k}_{kind}"
                if (row[name] != donor_row[f"original_{name}"]
                        or row[f"{name}_sha256"] != donor_row[f"original_{name}_sha256"]):
                    raise ValueError("Wrong-RIRs must be the coherent pair from one donor")
        actual = sorted(row[f"rir{k}_reverb_sha256"] for k in (1, 2))
        original = sorted(row[f"original_rir{k}_reverb_sha256"] for k in (1, 2))
        if actual == original:
            raise ValueError("Wrong-RIR donor has the original physical RIR set")
    if len(donors) != len(set(donors)) or set(donors) != set(keys):
        raise ValueError("Wrong-RIR donors must be a bijection within test")


def prepare(args):
    if args.expected_count < 4:
        raise ValueError("At least four scenes are required for four scoring shards")
    root = authorized(args.output)
    if root.exists() or args.output.is_symlink():
        raise FileExistsError("Use a new, separate evaluation output")
    dump, original, original_recipe, model = [
        p.resolve(strict=True) for p in
        (args.dump_root, args.original_test, args.original_recipe, args.model_exp)]
    for source in (dump, original, model):
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("Output must not overlap data/model directories")
    marker = dump / "preparation.json"
    generation = json.loads(marker.read_text())
    conditions = selected_conditions(generation, generation=True)
    if (generation["scene_count"] != args.expected_count
            or generation["condition_count"] != len(conditions)
            or generation["original_identity_verified_count"] != args.expected_count
            or generation["sample_rate"] != 8000
            or any(generation[k] for k in
                   ("ctf_estimation_run", "model_inference_run", "scoring_run"))):
        raise ValueError("Unexpected conditioning-audio dump provenance")
    tasks = []
    for relative, checksum in generation["output_hashes"].items():
        path = (dump / relative).resolve(strict=True)
        if not path.is_relative_to(dump):
            raise ValueError("Unsafe generation manifest path")
        tasks.append((path, checksum))
    print(f"Verify transferred dump: {len(tasks)} files", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify_hash, tasks))
    with (dump / "audit.tsv").open() as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    audits = {(r["condition"], r["scene_uid"]): r for r in records}
    if (len(records) != args.expected_count * len(conditions) or len(audits) != len(records)
            or set(audits) != {(c, uid) for c in conditions
                              for uid in read_index(original / "wav.scp")}):
        raise ValueError("Unexpected/duplicate audit keys")
    indexes = {}
    for name in NAMES:
        indexes[name] = {
            uid: str((Path(p) if Path(p).is_absolute() else original_recipe / p).resolve(strict=True))
            for uid, p in read_index(original / name).items()}
    keys = sorted(indexes["wav.scp"])
    if len(keys) != args.expected_count or any(set(v) != set(keys) for v in indexes.values()):
        raise ValueError("Original test key mismatch")
    lengths, hashes = {}, {}
    if "same_utt_other_rir" in conditions:
        validate_other_rir_audit(audits, set(keys))
    for uid in keys:
        record = audits[next(iter(conditions)), uid]
        for name, signal in zip(NAMES, ("wav", "spk1", "spk2")):
            path = indexes[name][uid]
            lengths[uid] = audio(path, lengths.get(uid))
            hashes[path] = record[f"baseline_{signal}_sha256"]
    print("Verify original test equals the generation identity-control target", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify_hash, hashes.items()))
    controls = {}
    for condition, dataset in conditions.items():
        values = read_index(dump / "raw" / dataset / "wav.scp")
        if set(values) != set(keys):
            raise ValueError("Conditioning keys differ from original")
        controls[condition] = {}
        for uid, value in values.items():
            relative = Path(value).relative_to(generation["destination_root"])
            path = (dump / relative).resolve(strict=True)
            if not path.is_relative_to(dump):
                raise ValueError("Conditioning path escapes dump")
            audio(path, lengths[uid], mono=True)
            row = audits[condition, uid]
            slots = (row["source1_rir_slot"], row["source2_rir_slot"])
            if condition == "same_rir_other_utt":
                if ({row["source1"], row["source2"]} & {row["original_source1"], row["original_source2"]}
                        or row["source1_speaker"] == row["source2_speaker"] or slots != ("1", "2")):
                    raise ValueError("Invalid independent-utterance intervention")
            elif condition == "same_utt_swapped_rir" and (row["source1"] != row["original_source1"]
                  or row["source2"] != row["original_source2"] or slots != ("2", "1")):
                raise ValueError("Invalid swapped-RIR intervention")
            if generation["output_hashes"][str(relative)] != row["new_wav_sha256"]:
                raise ValueError("Audit and generation hashes differ")
            controls[condition][uid] = str(path)
            hashes[str(path)] = row["new_wav_sha256"]
    config, checkpoint = [(model / n).resolve(strict=True) for n in ("config.yaml", "129epoch.pth")]
    score_dirs = {"baseline": args.baseline_score.resolve(strict=True),
                  "matched": args.matched_score.resolve(strict=True)}
    source_hashes = {str(marker): digest(marker), str(config): digest(config),
                     str(checkpoint): digest(checkpoint),
                     **{str(original / n): digest(original / n) for n in NAMES}}
    for directory in score_dirs.values():
        metric_values(directory, keys)
        for metric in METRICS:
            for speaker in (1, 2):
                path = directory / f"{metric}_spk{speaker}"
                source_hashes[str(path)] = digest(path)
    matched_outputs = {}
    for speaker in (1, 2):
        path = score_dirs["matched"].parent / f"spk{speaker}.scp"
        values = read_index(path)
        if set(values) != set(keys):
            raise ValueError("Missing saved matched outputs for parity control")
        matched_outputs[str(speaker)] = values
        source_hashes[str(path)] = digest(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        (temporary / "inputs/original").mkdir(parents=True)
        for name, values in indexes.items():
            save_index(temporary / "inputs/original" / name, values)
        for condition, values in controls.items():
            save_index(temporary / "inputs" / f"{condition}.scp", values)
        write_json(temporary / "provenance.json", dict(
            schema_version=1, count=len(keys), keys=keys, lengths=lengths,
            conditions=conditions, condition_definitions=generation["condition_definitions"],
            original_test=str(original), dump_root=str(dump), model_exp=str(model),
            config=str(config), checkpoint=str(checkpoint), frozen_bimamba_sha256=PIN,
            source_hashes=source_hashes, audio_hashes=hashes, matched_outputs=matched_outputs,
            score_dirs={k: str(v) for k, v in score_dirs.items()},
            settings=dict(fs=8000, batch_size=1, normalize_output_wav=True,
                          dtype="float32", segmented=False, seed=0),
            intervention="Only frozen BiMamba waveform changes; original separation mixture/references fixed",
            swapped_rir_caveat="Same RIR set, not a wrong-RIR-set negative control",
            input_hashes={str(p.relative_to(temporary)): digest(p)
                          for p in sorted((temporary / "inputs").rglob("*.scp"))}))
        temporary.rename(root)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"Prepared {len(keys)} paired scenes for {len(conditions)} condition(s)", flush=True)


def load_spec(root):
    root = authorized(root)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Symlinked evaluation write destination")
    spec = json.loads((root / "provenance.json").read_text())
    for item in spec["source_hashes"].items():
        verify_hash(item)
    for relative, checksum in spec["input_hashes"].items():
        verify_hash((root / relative, checksum))
    return spec


def infer(root):
    import torch
    from espnet2.bin.enh_inference import SeparateSpeech
    from espnet2.fileio.sound_scp import SoundScpWriter
    from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
    from ctf_audio_intervention import conditioning_override
    spec = load_spec(root)
    conditions = selected_conditions(spec)
    if (root / "inference_complete.json").exists() or any((root / c).exists() for c in conditions):
        raise FileExistsError("Refuse to overwrite previous inference; use a new output")
    if not torch.cuda.is_available():
        raise RuntimeError("Real BiMamba inference requires a GPU allocation")
    set_all_random_seed(0)
    separate = SeparateSpeech(train_config=spec["config"], model_file=spec["checkpoint"],
                              device="cuda", dtype="float32", normalize_output_wav=True)
    model = separate.enh_model
    if (separate.enh_train_args.separator != "tflocoformer_bimamba_ctf_concat"
            or not model.frozen_ctf.source_loaded.item()
            or bytes(model.frozen_ctf.source_sha256.tolist()).hex() != PIN
            or model.bimamba_sample_rate != 8000 or model.ref_channel != 0):
        raise ValueError("Unexpected model/predictor/source/channel")
    original = read_index(root / "inputs/original/wav.scp")
    controls = {c: read_index(root / "inputs" / f"{c}.scp") for c in conditions}
    by_length = sorted(spec["keys"], key=lambda uid: (spec["lengths"][uid], uid))
    smoke_keys = sorted({by_length[0], by_length[len(by_length)//2], by_length[-1]})
    smoke = []
    started = time.monotonic()
    for uid in smoke_keys:
        mixture = read_wave(original[uid])
        standard = separate(mixture, fs=8000)
        identity, checksum = conditioning_override(separate, mixture, mixture)
        if any(len(waves) != 2 or any(w.shape != mixture.shape or not np.isfinite(w).all()
                                     for w in waves) for waves in (standard, identity)):
            raise ValueError("Nonfinite/invalid identity-control output")
        error = max(float(np.max(np.abs(a-b))) for a, b in zip(standard, identity))
        saved_error = max(float(np.max(np.abs(w - read_wave(spec["matched_outputs"][str(k)][uid]))))
                          for k, w in enumerate(standard, 1))
        if error > 1e-5 or saved_error > 5e-5:
            raise ValueError(f"Identity/saved matched parity failed: {uid}: {error}, {saved_error}")
        entry = dict(uid=uid, identity_max_error=error, saved_matched_max_error=saved_error,
                     identity_ctf_sha256=checksum, conditions={})
        for condition in conditions:
            waves, checksum = conditioning_override(separate, mixture, read_wave(controls[condition][uid]))
            if len(waves) != 2 or any(w.shape != mixture.shape or not np.isfinite(w).all() for w in waves):
                raise ValueError("Invalid smoke inference")
            entry["conditions"][condition] = checksum
        smoke.append(entry)
    write_json(root / "smoke.json", dict(passed=True, scenes=smoke, seconds=time.monotonic()-started))
    print("PASS: identity, saved matched outputs and selected interventions; start full inference", flush=True)
    output_hashes = {}
    for condition in conditions:
        directory = root / condition
        directory.mkdir()
        writers = [SoundScpWriter(str(directory / "wavs" / str(k)), str(directory / f"spk{k}.scp"))
                   for k in (1, 2)]
        try:
            with (directory / "ctf_inputs.jsonl").open("w") as audit:
                for n, uid in enumerate(spec["keys"], 1):
                    paths = (original[uid], controls[condition][uid])
                    for path in paths:
                        verify_hash((path, spec["audio_hashes"][path]))
                    mixture = read_wave(paths[0])
                    waves, checksum = conditioning_override(separate, mixture, read_wave(paths[1]))
                    for writer, wave in zip(writers, waves):
                        if wave.shape != mixture.shape or not np.isfinite(wave).all():
                            raise ValueError(f"Invalid separated output: {uid}")
                        writer[uid] = 8000, wave[0]
                    audit.write(json.dumps(dict(uid=uid, mixture_path=paths[0],
                                               conditioning_path=paths[1], ctf_sha256=checksum)) + "\n")
                    if n % 100 == 0:
                        print(f"{condition}: {n}/{spec['count']}", flush=True)
        finally:
            for writer in writers:
                writer.close()
        for k in (1, 2):
            index = read_index(directory / f"spk{k}.scp")
            if set(index) != set(spec["keys"]):
                raise ValueError("Incomplete inference")
            for uid, path in index.items():
                audio(path, spec["lengths"][uid], mono=True)
                output_hashes[path] = digest(path)
    write_json(root / "inference_complete.json", dict(
        count_per_condition=spec["count"], seconds=time.monotonic()-started,
        output_hashes=output_hashes, input_hashes=spec["input_hashes"]))


def score(root):
    spec = load_spec(root)
    conditions = selected_conditions(spec)
    complete = json.loads((root / "inference_complete.json").read_text())
    if complete["count_per_condition"] != spec["count"] or complete["input_hashes"] != spec["input_hashes"]:
        raise ValueError("Incomplete inference provenance")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify_hash, complete["output_hashes"].items()))
        # Recheck original references as well as estimator inputs before scoring.
        list(executor.map(verify_hash, spec["audio_hashes"].items()))
    for condition in conditions:
        directory = root / condition
        for k in (1, 2):
            index = read_index(directory / f"spk{k}.scp")
            if set(index) != set(spec["keys"]):
                raise ValueError("Incomplete inference keys")
            for uid, path in index.items():
                audio(path, spec["lengths"][uid], mono=True)
        if (directory / "score").exists() or (directory / "score_parts").exists():
            raise FileExistsError("Refuse to overwrite previous scoring")
        original = read_index(root / "inputs/original/wav.scp")

        def worker(part):
            folder = directory / "score_parts" / str(part)
            folder.mkdir(parents=True)
            selected = set(spec["keys"][part::4])
            save_index(folder / "keys.scp", {k: v for k, v in original.items() if k in selected})
            command = [sys.executable, "-B", "-m", "espnet2.bin.enh_scoring",
                       "--ref_scp", str(root / "inputs/original/spk1.scp"),
                       "--ref_scp", str(root / "inputs/original/spk2.scp"),
                       "--inf_scp", str(directory / "spk1.scp"),
                       "--inf_scp", str(directory / "spk2.scp"),
                       "--key_file", str(folder / "keys.scp"), "--ref_channel", "0",
                       "--output_dir", str(folder / "score")]
            with (folder / "scoring.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            return folder / "score"

        with ThreadPoolExecutor(max_workers=4) as executor:
            parts = list(executor.map(worker, range(4)))
        destination = directory / "score"
        destination.mkdir()
        for name in sorted(p.name for p in parts[0].iterdir() if p.is_file()):
            values = {}
            for part in parts:
                incoming = read_index(part / name)
                if set(values) & set(incoming):
                    raise ValueError("Overlapping score shards")
                values.update(incoming)
            if set(values) != set(spec["keys"]):
                raise ValueError("Incomplete merged scores")
            save_index(destination / name, values)
        metric_values(destination, spec["keys"])
    summarize(root, spec)
    write_json(root / "score_complete.json", dict(count_per_condition=spec["count"],
               conditions=list(conditions), summary_sha256=digest(root / "summary.csv")))


def summarize(root, spec):
    conditions = selected_conditions(spec)
    dirs = {**{k: Path(v) for k, v in spec["score_dirs"].items()},
            **{c: root / c / "score" for c in conditions}}
    values = {label: metric_values(directory, spec["keys"]) for label, directory in dirs.items()}
    rows = []
    for metric in METRICS:
        row = {"metric": metric}
        for label in dirs:
            row[label] = math.fsum(values[label][f"{metric}_spk{k}"][uid]
                                  for k in (1, 2) for uid in spec["keys"]) / (2 * spec["count"])
        for condition in conditions:
            row[f"{condition}_delta_matched"] = row[condition] - row["matched"]
            row[f"{condition}_delta_baseline"] = row[condition] - row["baseline"]
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    paired = []
    for uid in spec["keys"]:
        row = {"uid": uid}
        for label in dirs:
            row[label] = sum(values[label][f"SI_SNR_spk{k}"][uid] for k in (1, 2)) / 2
        paired.append(row)
    with (root / "paired_si_snr.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for name in ("dump-root", "original-test", "original-recipe", "model-exp", "output",
                 "baseline-score", "matched-score"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--expected-count", type=int, default=3000)
    for name in ("infer", "score"):
        sub.add_parser(name).add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "infer":
        infer(authorized(args.output))
    else:
        score(authorized(args.output))


if __name__ == "__main__":
    main()
