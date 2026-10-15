#!/usr/bin/env python3
"""Evaluate native 8 kHz Rec-RIR best97 with the saved DAMSEP/BiMamba inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    workspace, previous, output = (
        p.resolve() for p in (args.workspace, args.previous, args.output)
    )
    experiment = workspace / "egs2/whamr/rir/exp/rir_train_rec_rir_2spk_pit_clean_8192"
    checkpoint = experiment / "valid.loss.best.pth"
    reference = (
        workspace
        / "egs2/whamr/rir_2spk/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
    )
    old_report = json.loads((previous / "comparison/comparison.json").read_text())
    if (
        (previous / "exit_status").read_text().strip() != "0"
        or not old_report["full_dataset"]
        or old_report["count"] != 3000
        or old_report["reference_dir"] != str(reference)
        or digest(reference / "preparation.json")
        != old_report["reference_preparation_sha256"]
        or checkpoint.resolve().name != "97epoch.pth"
    ):
        raise ValueError("Expected completed, unchanged 8 kHz comparison and best97")
    config = yaml.safe_load((experiment / "config.yaml").read_text())
    if config["sample_rate"] != 8000 or config["rir_model_type"] != "rec_rir_pit":
        raise ValueError("Expected native 8 kHz Rec-RIR PIT model")
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("config.yaml", "valid.loss.best.pth"):
        shutil.copy2(experiment / filename, output / filename)
    shutil.copy2(previous / "bimamba/wav.scp", output / "wav.scp")
    rows = [
        line.split(maxsplit=1) for line in (output / "wav.scp").read_text().splitlines()
    ]
    if len(rows) != 3000 or len({uid for uid, _ in rows}) != 3000:
        raise ValueError("Expected the same complete 3000-mixture input index")
    (output / "metadata.json").write_text(
        json.dumps(
            dict(
                inference="ordinary sweep PIM",
                sample_rate=8000,
                rir_length=16000,
                count=3000,
                input="full utterance, channel 0, same PCM16 as DAMSEP/BiMamba",
                experiment=str(experiment),
                checkpoint="97epoch.pth",
                checkpoint_sha256=digest(output / "valid.loss.best.pth"),
                config_sha256=digest(output / "config.yaml"),
                input_index_sha256=digest(output / "wav.scp"),
                input_correspondence_sha256=digest(
                    previous / "input_correspondence.json"
                ),
                prior_comparison_sha256=digest(previous / "comparison/comparison.json"),
                loss_weights={
                    key: config["model_conf"][key]
                    for key in ("loss_w_cln", "loss_w_rvb", "loss_w_rec")
                },
            ),
            indent=2,
        )
        + "\n"
    )
    started = time.monotonic()
    # Check dependency/checkpoint compatibility on three real inputs before
    # running all 3000. The full inference has its own fresh output directory.
    (output / "smoke.scp").write_text(
        "".join(f"{uid} {path}\n" for uid, path in rows[:3])
    )
    command = [
        sys.executable,
        "-m",
        "espnet2.bin.rec_rir_pit_inference",
        "--train_config",
        str(output / "config.yaml"),
        "--model_file",
        str(output / "valid.loss.best.pth"),
        "--sample_rate",
        "8000",
        "--rir_length",
        "16000",
        "--output_subtype",
        "FLOAT",
        "--device",
        args.device,
    ]
    subprocess.run(
        command
        + [
            "--wav_scp",
            str(output / "smoke.scp"),
            "--output_dir",
            str(output / "smoke"),
        ],
        check=True,
    )
    print("Rec-RIR best97 smoke passed; starting all 3000 mixtures", flush=True)
    subprocess.run(
        command
        + [
            "--wav_scp",
            str(output / "wav.scp"),
            "--output_dir",
            str(output / "inference"),
        ],
        check=True,
    )
    (output / "inference_complete.json").write_text(
        json.dumps(dict(count=3000, elapsed_seconds=time.monotonic() - started)) + "\n"
    )
    predictions = []
    for name, model in old_report["models"].items():
        if digest(model["prediction_scp"]) != model["prediction_index_sha256"]:
            raise ValueError(f"Saved prediction index changed: {name}")
        predictions += ["--prediction", name, model["prediction_scp"]]
    predictions += [
        "--prediction",
        "RecRIR_8k_best97",
        str(output / "inference/rir.scp"),
    ]
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_ls_direct_reverb.py")),
            "--reference_dir",
            str(reference),
            *predictions,
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
                models=3,
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
