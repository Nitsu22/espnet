#!/usr/bin/env python3
"""Compare TF-DAMSEP and noisy BiMamba on identical inputs and immutable LS RIRs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_index(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in result or Path(uid).name != uid or uid in (".", ".."):
            raise ValueError(f"Duplicate or unsafe ID in {path}: {uid}")
        result[uid] = value
    if not result:
        raise ValueError(f"Empty index: {path}")
    return result


def canonical_uid(uid):
    fields = uid.split("_", 2)
    if len(fields) != 3 or not uid.endswith("_reverb"):
        raise ValueError(f"Unexpected DAMSEP ID: {uid}")
    key = fields[2][: -len("_reverb")]
    parts = key.split("_")
    if len(parts) != 4 or fields[:2] != [parts[0][:3], parts[2][:3]]:
        raise ValueError(f"Invalid source prefixes: {uid}")
    return key


def paired_bootstrap(a, b, repetitions=10000, seed=20261028):
    """Mean B-A; one resampling unit is a mixture (both sources together)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if (a.shape != b.shape or a.ndim != 1 or not a.size
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or repetitions < 1):
        raise ValueError("Expected finite paired mixture means")
    delta = b - a
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 256):
        index = rng.integers(0, len(delta), size=(min(256, repetitions-start), len(delta)))
        draws.append(delta[index].mean(axis=1))
    low, high = np.quantile(np.concatenate(draws), [0.025, 0.975])
    return dict(difference_b_minus_a=float(delta.mean()),
                ci95=[float(low), float(high)], mixtures=len(a),
                repetitions=repetitions, seed=seed)


def validate(args, output):
    workspace = args.workspace.resolve()
    recipe = workspace / "egs2/whamr/rir_2spk"
    noisy = recipe / "dump_noisy_2spk_8k_min/raw/tt_rir_2spk_noisy_min_8k"
    reference = recipe / "dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
    source = args.source.resolve()
    manifest = json.loads((source / "source_manifest.json").read_text())
    info = manifest["inference"]
    if (info["count"] != 3000 or info["sample_rate"] != 8000
            or info["ctf_shape"] != [2, 257, 60] or not info["ctf_enabled"]):
        raise ValueError("Expected full three-loss TF-DAMSEP CTF export")
    if not manifest["source_experiment"].endswith("/tflocoformer_damsep_whamr_8k_three_loss"):
        raise ValueError("This comparison requires the TF-DAMSEP noisy WHAMR experiment")
    # Establish that the existing NF LS reference has the SAME physical RIR
    # pairs as the noisy test; adding noise does not change the target filter.
    ref_info = json.loads((reference / "preparation.json").read_text())
    expected = dict(sample_rate=8000, rir_length=16000, filter_length=16000,
                    sweep_duration=8.192, regularization=1e-6,
                    solver_rtol=1e-8, max_iterations=500)
    if ref_info["config"] != expected or ref_info["count"] != 3000:
        raise ValueError("Unexpected LS reference settings")
    for name in ("rir_ref1.scp", "rir_ref2.scp", "rir_direct1.scp", "rir_direct2.scp"):
        if digest(noisy / name) != ref_info["source_index_sha256"][name]:
            raise ValueError(f"Noisy input RIRs differ from LS reference: {name}")
    canonical = read_index(noisy / "wav.scp")
    mix_recipe = workspace / "egs2/whamr/enh1"
    mix_scp = mix_recipe / "dump/raw/tt_mix_both_reverb_min_8k/wav.scp"
    mixtures = read_index(mix_scp)
    if digest(mix_scp) != manifest["wav_scp_sha256"]:
        raise ValueError("Local/TSUBAME formatted input indexes differ")
    if set(mixtures) != set(manifest["entries"]) or len(mixtures) != 3000:
        raise ValueError("DAMSEP input/export ID mismatch")
    mapping = {canonical_uid(uid): uid for uid in mixtures}
    if len(mapping) != 3000 or set(mapping) != set(canonical):
        raise ValueError("Canonical noisy / DAMSEP IDs differ")
    reference_ids = read_index(reference / "reference.scp")
    if set(reference_ids) != set(canonical):
        raise ValueError("LS reference IDs differ")
    rows = []
    max_error = 0.0
    saturated_utterances = saturated_samples = 0
    for number, key in enumerate(sorted(canonical), 1):
        uid = mapping[key]
        path = (mix_recipe / mixtures[uid]).resolve()
        archive = source / "damsep/ctf" / (uid + ".npz")
        recorded = manifest["entries"][uid]
        if digest(path) != recorded["input_sha256"]:
            raise ValueError(f"Not the exact DAMSEP inference input: {uid}")
        if digest(archive) != recorded["ctf_sha256"]:
            raise ValueError(f"Saved CTF checksum mismatch: {uid}")
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        y, fs = sf.read(canonical[key], dtype="float32", always_2d=True)
        if sr != 8000 or fs != 8000 or x.shape != y.shape or not np.isfinite(x).all():
            raise ValueError(f"Input rate/shape/values differ: {uid}")
        clipped = np.clip(y, -1, 1-1/32768)
        error = float(np.max(np.abs(x-clipped)))
        if not np.isfinite(y).all() or error > 1/32768 + 1e-7:
            raise ValueError(f"Not the canonical noisy mixture, even after PCM16: {uid}")
        clipped_count = int(np.count_nonzero(y != clipped))
        saturated_utterances += int(clipped_count > 0)
        saturated_samples += clipped_count
        max_error = max(max_error, error)
        rows.append(dict(uid=key, damsep_uid=uid, input=str(path), ctf=str(archive),
                         input_sha256=recorded["input_sha256"],
                         ctf_sha256=recorded["ctf_sha256"]))
        if number % 500 == 0:
            print(f"Validated exact input/CTF correspondence {number}/3000", flush=True)
    # Independently check additive noise is present in a fixed sample.
    indices = {name: read_index(noisy / (name + ".scp")) for name in
               ("speech_reverb1", "speech_reverb2", "noise_ref")}
    for row in (rows[0], rows[len(rows)//2], rows[-1]):
        key = row["uid"]
        waves = [sf.read(indices[n][key], dtype="float32", always_2d=True)[0]
                 for n in indices]
        mixture = sf.read(canonical[key], dtype="float32", always_2d=True)[0]
        if not np.any(waves[2][:, 0] != 0):
            raise ValueError("Silent noise reference")
        np.testing.assert_allclose(mixture, sum(waves), atol=2e-6, rtol=0)
    audit = dict(count=3000, exact_input_sha256_match=True,
                 exact_ctf_sha256_match=True, same_physical_rir_indexes=True,
                 max_pcm16_quantization_error=max_error,
                 saturated_utterances=saturated_utterances,
                 saturated_samples_both_channels=saturated_samples,
                 source_manifest_sha256=digest(source / "source_manifest.json"),
                 ls_preparation_sha256=digest(reference / "preparation.json"),
                 source_experiment=manifest["source_experiment"],
                 source_checkpoint_sha256=manifest["checkpoint_sha256"], entries=rows)
    write_json(output / "input_audit.json", audit)
    return rows, reference, manifest


def diagnostics(output, reference):
    from score_rir import fix_length, peak_align, scale_signals
    from score_rir_pit import read_3col

    refs = read_3col(reference / "reference.scp")
    by_model = {}
    summary = {}
    for model in ("tfdamsep", "bimamba"):
        with (output / model / "score/per_source.csv").open() as stream:
            scored = list(csv.DictReader(stream))
        predictions = read_3col(output / model / "inference/rir.scp")
        rows = []
        for row in scored:
            uid = row["utt_id"]
            pred = sf.read(predictions[uid][int(row["pred_index"])], dtype="float64")[0]
            ref = sf.read(refs[uid][int(row["ref_index"])], dtype="float64")[0]
            pred, ref = scale_signals(peak_align(pred, ref), ref, "peak")
            pred, ref = fix_length(pred, 16000), fix_length(ref, 16000)
            onset = int(np.argmax(np.abs(ref))) + 400
            late_error = float(np.sum((pred[onset:]-ref[onset:])**2))
            late_energy = float(np.sum(ref[onset:]**2))
            if late_energy <= np.finfo(float).tiny:
                raise ValueError(f"No late LS reference energy: {uid}")
            row = dict(row, late_nmse_db=10*np.log10(max(late_error, 1e-30)/late_energy))
            rows.append(row)
        with (output / model / "diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows(rows)
        metrics = ("rmse", "corr", "rmse_direct_50ms", "late_nmse_db")
        means = {}
        for row in rows:
            values = [float(row[k]) for k in metrics]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite primary/late metric")
            means.setdefault(row["utt_id"], []).append(values)
        if any(len(v) != 2 for v in means.values()):
            raise ValueError("Both speakers must stay in each bootstrap unit")
        keys = sorted(means)
        by_model[model] = (keys, np.asarray([means[k] for k in keys]).mean(axis=1))
        summary[model] = {k: float(by_model[model][1][:, i].mean())
                          for i, k in enumerate(metrics)}
    if by_model["tfdamsep"][0] != by_model["bimamba"][0]:
        raise ValueError("Paired bootstrap model IDs differ")
    differences = {k: paired_bootstrap(by_model["tfdamsep"][1][:, i],
                                      by_model["bimamba"][1][:, i])
                   for i, k in enumerate(metrics)}
    result = dict(models=summary, a="tfdamsep", b="bimamba", differences=differences,
                  primary_metrics=["rmse", "corr"],
                  secondary_metrics=["rmse_direct_50ms", "late_nmse_db"],
                  late_definition="50 ms after LS reference absolute peak through 2 s",
                  bootstrap_unit="mixture, both sources together", multiple_seed_claim=False,
                  limitation="LS relative-transfer shape; absolute delay/gain not scored")
    write_json(output / "paired_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Explicit smoke-test subset")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    rows, reference, manifest = validate(args, output)
    if args.limit:
        if not 0 < args.limit <= 3000:
            raise ValueError("Invalid subset limit")
        rows = rows[:args.limit]
    import torch
    from espnet2.rir.damsep_pim import DAMSEPOrdinaryPIM
    from prepare_ls_rir_reference import validate_dump

    torch.set_num_threads(1)
    validate_dump(reference)
    baseline = args.workspace / "egs2/whamr/rir_2spk/exp/rir_train_pooled_bimamba_2spk_noisy_8k_sweep_v2"
    import yaml
    config = yaml.safe_load((baseline / "config.yaml").read_text())
    mix_inputs = [p for p, name, _ in config["train_data_path_and_name_and_type"]
                  if name == "speech_mix"]
    if (config["sample_rate"] != 8000
            or config["rir_model_type"] != "pooled_bimamba_sweep_v2_pit"
            or len(mix_inputs) != 1 or "noisy" not in mix_inputs[0]
            or config["model_conf"]["n_fft"] != 256
            or config["model_conf"]["hop_len"] != 128
            or config["model_conf"]["ctf_taps"] != 60):
        raise ValueError("Expected the native 8 kHz noisy BiMamba checkpoint")
    model_dirs = {name: output / name for name in ("tfdamsep", "bimamba")}
    for folder in model_dirs.values():
        folder.mkdir()
    folder = model_dirs["bimamba"]
    for name in ("config.yaml", "valid.loss.best.pth"):
        shutil.copyfile(baseline / name, folder / name)
    (folder / "wav.scp").write_text("".join(f"{r['uid']} {r['input']}\n" for r in rows))
    command = [sys.executable, "-B", "-m", "espnet2.bin.rec_rir_pit_inference",
               "--train_config", str(folder / "config.yaml"),
               "--model_file", str(folder / "valid.loss.best.pth"),
               "--wav_scp", str(folder / "wav.scp"),
               "--output_dir", str(folder / "inference"), "--sample_rate", "8000",
               "--rir_length", "16000", "--output_subtype", "FLOAT", "--device", args.device]
    print("Starting noisy BiMamba inference", flush=True)
    subprocess.run(command, check=True)
    write_json(folder / "metadata.json", dict(checkpoint=str(baseline / "valid.loss.best.pth"),
               checkpoint_sha256=digest(folder / "valid.loss.best.pth"),
               config_sha256=digest(folder / "config.yaml"), input="exact TF-DAMSEP PCM16 noisy mixture"))
    decoder = DAMSEPOrdinaryPIM()
    prediction = model_dirs["tfdamsep"] / "inference"
    prediction.mkdir()
    lines = []
    for number, row in enumerate(rows, 1):
        with np.load(row["ctf"], allow_pickle=False) as archive:
            ctf = archive["ctf"]
        if ctf.shape != (2, 257, 60):
            raise ValueError("Unexpected TF-DAMSEP CTF shape")
        paths = []
        for speaker in range(2):
            dest = prediction / f"{row['uid']}_s{speaker+1}.wav"
            sf.write(dest, decoder(ctf[speaker]), 8000, subtype="FLOAT")
            paths.append(str(dest))
        lines.append(f"{row['uid']} {' '.join(paths)}\n")
        if number % 100 == 0:
            print(f"Converted TF-DAMSEP CTF {number}/{len(rows)}", flush=True)
    (prediction / "rir.scp").write_text("".join(lines))
    write_json(model_dirs["tfdamsep"] / "metadata.json", dict(source=manifest,
               transform=dict(n_fft=512, win_length=256, hop_length=128, window="sqrthann"),
               measurement="ordinary sweep, causal full linear frame convolution; no oracle excitation"))
    # Restrict the scorer's reference IDs for an explicit smoke test only.
    reference_scp = reference / "reference.scp"
    if args.limit:
        chosen = {r["uid"] for r in rows}
        reference_scp = output / "subset_reference.scp"
        reference_scp.write_text("".join(l+"\n" for l in (reference / "reference.scp").read_text().splitlines()
                                        if l.split()[0] in chosen))
    scorer = Path(__file__).with_name("score_rir_pit.py")
    for model, folder in model_dirs.items():
        command = [sys.executable, "-B", str(scorer), "--ref_rir_scp", str(reference_scp),
                   "--pred_rir_scp", str(folder / "inference/rir.scp"), "--out_dir", str(folder / "score"),
                   "--sample_rate", "8000", "--rir_length", "16000", "--align", "peak",
                   "--scale_mode", "peak", "--pit_metric", "rmse"]
        print(f"Scoring {model}", flush=True)
        subprocess.run(command, check=True)
        summary = json.loads((folder / "score/summary.json").read_text())
        if summary["num_utts"] != len(rows) or summary["num_source_pairs"] != 2*len(rows):
            raise ValueError("Incomplete LS scoring")
    result = diagnostics(output, reference)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                     cwd=Path(__file__).resolve().parents[4]).strip()
    write_json(output / "complete.json", dict(count=len(rows), source_pairs=2*len(rows),
               full_dataset=not args.limit, elapsed_seconds=time.monotonic()-start, git_commit=commit,
               paired_comparison_sha256=digest(output / "paired_comparison.json")))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
