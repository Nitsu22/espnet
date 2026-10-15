#!/usr/bin/env python3
"""Compare saved DAMSEP CTFs and native 8 kHz BiMamba on immutable LS RIRs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from espnet2.rir.damsep_pim import DAMSEPOrdinaryPIM


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_index(path):
    entries = {}
    for line in Path(path).read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in entries or Path(uid).name != uid or uid in (".", ".."):
            raise ValueError(f"Duplicate or unsafe ID: {uid}")
        entries[uid] = value
    if not entries:
        raise ValueError(f"Empty index: {path}")
    return entries


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    recipe = workspace / "egs2/whamr/rir_2spk"
    source = output / "input/damsep"
    data = recipe / "dump_nf_2spk_8k_min/raw/tt_rir_2spk_nf_min_8k"
    reference = recipe / "dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
    experiment = recipe / "exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2"
    damsep_recipe = workspace / "egs2/whamr/damsep_clean"
    mixture_index = damsep_recipe / "dump_clean/raw/tt_mix_clean_reverb_min_8k/wav.scp"
    canonical = read_index(data / "wav.scp")
    mixtures = read_index(mixture_index)
    ctfs = read_index(source / "ctf.scp")
    provenance = json.loads((source / "inference.json").read_text())
    if (
        len(canonical) != 3000
        or set(ctfs) != set(mixtures)
        or provenance["count"] != 3000
        or provenance["sample_rate"] != 8000
        or provenance["ctf_shape"] != [2, 257, 60]
        or provenance.get("clean_only", False)
    ):
        raise ValueError("Expected complete three-loss DAMSEP test export")
    mapping = {}
    for uid in ctfs:
        # ESPnet adds the two speaker IDs and _reverb; retain the original
        # WHAMR mixture filename, including the signed source gains.
        parts = uid.split("_", 2)
        if len(parts) != 3 or not uid.endswith("_reverb"):
            raise ValueError(f"Unexpected DAMSEP ID: {uid}")
        key = parts[2][: -len("_reverb")]
        fields = key.split("_")
        if parts[:2] != [fields[0][:3], fields[2][:3]] or key in mapping:
            raise ValueError(f"Invalid speaker prefix or duplicate mapping: {uid}")
        mapping[key] = uid
    if set(mapping) != set(canonical):
        raise ValueError("DAMSEP/BiMamba/LS IDs differ")
    # The model directories must be new. Do not overwrite partial predictions.
    for name in ("damsep", "bimamba"):
        (output / name).mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    validated = []
    maximum_difference = 0.0
    for number, key in enumerate(canonical, 1):
        uid = mapping[key]
        path = Path(mixtures[uid])
        if not path.is_absolute():
            path = damsep_recipe / path
        path = path.resolve()
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        y, target_sr = sf.read(canonical[key], dtype="float32", always_2d=True)
        if sr != 8000 or target_sr != 8000 or x.shape != y.shape:
            raise ValueError(f"Input rate/shape differs: {uid}")
        error = float(np.max(np.abs(x - y)))
        # DAMSEP's formatted PCM16 input and the FLOAT NF dump differ by
        # quantization. BiMamba receives the exact SAME PCM16 input here.
        if not np.isfinite(x).all() or not error <= 1 / 32768 + 1e-7:
            raise ValueError(f"Not the same noise-free mixture: {uid}: {error}")
        maximum_difference = max(maximum_difference, error)
        validated.append(
            dict(
                uid=key,
                damsep_uid=uid,
                input=str(path),
                input_sha256=digest(path),
                canonical_input=canonical[key],
                maximum_sample_difference=error,
            )
        )
        if number % 500 == 0:
            print(f"Validated input correspondence {number}/3000", flush=True)
    (output / "input_correspondence.json").write_text(
        json.dumps(
            dict(
                count=len(validated),
                maximum_sample_difference=maximum_difference,
                baseline_input="same formatted PCM16 noise-free mixture as DAMSEP",
                ctfs_index_sha256=digest(source / "ctf.scp"),
                entries=validated,
            ),
            indent=2,
        )
        + "\n"
    )
    baseline = output / "bimamba"
    (baseline / "wav.scp").write_text(
        "".join(f"{item['uid']} {item['input']}\n" for item in validated)
    )
    for filename in ("config.yaml", "valid.loss.best.pth"):
        shutil.copy2(experiment / filename, baseline / filename)
    (baseline / "metadata.json").write_text(
        json.dumps(
            dict(
                inference="ordinary sweep PIM",
                sample_rate=8000,
                rir_length=16000,
                count=3000,
                input="full utterance, channel 0, same PCM16 as DAMSEP",
                experiment=str(experiment),
                checkpoint="valid.loss.best.pth",
                checkpoint_sha256=digest(baseline / "valid.loss.best.pth"),
                config_sha256=digest(baseline / "config.yaml"),
            ),
            indent=2,
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "espnet2.bin.rec_rir_pit_inference",
            "--train_config",
            str(baseline / "config.yaml"),
            "--model_file",
            str(baseline / "valid.loss.best.pth"),
            "--wav_scp",
            str(baseline / "wav.scp"),
            "--output_dir",
            str(baseline / "inference"),
            "--sample_rate",
            "8000",
            "--rir_length",
            "16000",
            "--output_subtype",
            "FLOAT",
            "--device",
            args.device,
        ],
        check=True,
    )
    (baseline / "complete.json").write_text(json.dumps(dict(count=3000)) + "\n")
    decoder = DAMSEPOrdinaryPIM()
    prediction = output / "damsep/inference"
    prediction.mkdir()
    rows = []
    checksums = {}
    for number, key in enumerate(canonical, 1):
        uid = mapping[key]
        archive = source / "ctf" / Path(ctfs[uid]).name
        if archive.name != uid + ".npz":
            raise ValueError(f"CTF filename/ID mismatch: {uid}")
        with np.load(archive, allow_pickle=False) as loaded:
            coefficients = loaded["ctf"]
        if coefficients.shape != (2, 257, 60):
            raise ValueError(f"Invalid CTF shape: {uid}")
        paths = []
        for speaker in range(2):
            path = prediction / f"{key}_s{speaker + 1}.wav"
            sf.write(path, decoder(coefficients[speaker]), 8000, subtype="FLOAT")
            paths.append(str(path))
        rows.append(f"{key} {' '.join(paths)}\n")
        checksums[uid] = digest(archive)
        if number % 100 == 0:
            print(f"Converted DAMSEP CTF {number}/3000", flush=True)
    (prediction / "rir.scp").write_text("".join(rows))
    (output / "damsep/metadata.json").write_text(
        json.dumps(
            dict(
                inference="ordinary sweep PIM, causal DAMSEP tap order",
                sample_rate=8000,
                rir_length=16000,
                count=3000,
                transform=dict(
                    n_fft=512, win_length=256, hop_length=128, window="sqrthann"
                ),
                transform_basis="DAMSEP CTF reconstruction loss, direct to reverb",
                source_inference=provenance,
                ctf_sha256=checksums,
                source_index_sha256=digest(source / "ctf.scp"),
                no_oracle_excitation=True,
            ),
            indent=2,
        )
        + "\n"
    )
    (output / "damsep/complete.json").write_text(json.dumps(dict(count=3000)) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_ls_direct_reverb.py")),
            "--reference_dir",
            str(reference),
            "--prediction",
            "DAMSEP_3loss_best29",
            str(prediction / "rir.scp"),
            "--prediction",
            "BiMamba_8k_best",
            str(baseline / "inference/rir.scp"),
            "--out_dir",
            str(output / "comparison"),
        ],
        check=True,
    )
    (output / "complete.json").write_text(
        json.dumps(
            dict(
                count=3000,
                source_pairs=6000,
                elapsed_seconds=time.monotonic() - started,
                git_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                    cwd=Path(__file__).resolve().parents[4],
                ).strip(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
