#!/usr/bin/env python3
"""Compare Rec-RIR loss ablations using average-five weights and native 8k LS."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    recipe = workspace / "egs2/whamr/rir_2spk"
    prior = recipe / "exp/rec_rir_ls8k_20261015"
    old_report = json.loads((prior / "comparison/comparison.json").read_text())
    reference = recipe / "dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
    old_model = old_report["models"]["RecRIR_8k_best97"]
    if (
        output.exists()
        or (recipe / "exp/rec_rir_ls8k_20261015.exit_status").read_text().strip() != "0"
        or not old_report["full_dataset"]
        or old_report["count"] != 3000
        or old_report["reference_dir"] != str(reference)
        or digest(reference / "preparation.json") != old_report["reference_preparation_sha256"]
        or digest(prior / "wav.scp") != json.loads((prior / "metadata.json").read_text())["input_index_sha256"]
        or digest(old_model["prediction_scp"]) != old_model["prediction_index_sha256"]
    ):
        raise ValueError("Expected fresh output and unchanged completed native-8k evaluation")
    rows = (prior / "wav.scp").read_text().splitlines()
    if len(rows) != 3000 or len({line.split()[0] for line in rows}) != 3000:
        raise ValueError("Expected all 3000 distinct mixture IDs")
    experiments = [
        ("RecRIR_3loss_ave5", "rir_train_rec_rir_2spk_pit_clean_8192"),
        ("RecRIR_rec_only_ave5", "rir_train_rec_rir_2spk_pit_clean_8192_loss_rec_only"),
    ]
    configurations = []
    for label, name in experiments:
        experiment = workspace / "egs2/whamr/rir/exp" / name
        conf = yaml.safe_load((experiment / "config.yaml").read_text())
        weights = [conf["model_conf"][key] for key in ("loss_w_cln", "loss_w_rvb", "loss_w_rec")]
        expected = [1, 1, 1] if label == "RecRIR_3loss_ave5" else [0, 0, 1]
        if conf["sample_rate"] != 8000 or conf["rir_model_type"] != "rec_rir_pit" or weights != expected:
            raise ValueError(f"Unexpected model configuration: {label}")
        # Require the loss ablation to preserve every other saved training setting.
        comparable = dict(conf)
        comparable.pop("config")
        comparable.pop("output_dir")
        comparable["model_conf"] = dict(conf["model_conf"])
        for key in ("loss_w_cln", "loss_w_rvb", "loss_w_rec"):
            comparable["model_conf"].pop(key)
        configurations.append(comparable)
        if not (experiment / "valid.loss.ave_5best.pth").is_file():
            raise ValueError(f"Missing average-five checkpoint: {label}")
    if configurations[0] != configurations[1]:
        raise ValueError("Training settings differ beyond the loss weights")

    output.mkdir(parents=True)
    shutil.copy2(prior / "wav.scp", output / "wav.scp")
    (output / "smoke.scp").write_text("\n".join(rows[:3]) + "\n")
    code_root = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    started = time.monotonic()
    provenance = dict(
        host=socket.gethostname(), git_commit=commit, input_index_sha256=digest(output / "wav.scp"),
        reference_preparation_sha256=digest(reference / "preparation.json"),
        training_configurations_match_except_loss_weights=True, models={},
    )
    predictions = []
    for label, name in experiments:
        experiment = workspace / "egs2/whamr/rir/exp" / name
        folder = output / label
        folder.mkdir()
        for source, target in (("config.yaml", "config.yaml"), ("valid.loss.ave_5best.pth", "model.pth")):
            shutil.copy2(experiment / source, folder / target)
            if digest(experiment / source) != digest(folder / target):
                raise ValueError("Snapshot hash mismatch")
        conf = yaml.safe_load((folder / "config.yaml").read_text())
        metadata = dict(
            inference="ordinary sweep PIM", sample_rate=8000, rir_length=16000,
            count=3000, input="full utterance, channel 0, same PCM16 as prior 8k LS evaluation",
            experiment=str(experiment), checkpoint="valid.loss.ave_5best.pth",
            checkpoint_sha256=digest(folder / "model.pth"), config_sha256=digest(folder / "config.yaml"),
            input_index_sha256=digest(output / "wav.scp"),
            loss_weights={key: conf["model_conf"][key] for key in ("loss_w_cln", "loss_w_rvb", "loss_w_rec")},
        )
        write_json(folder / "metadata.json", metadata)
        provenance["models"][label] = metadata
        write_json(output / "metadata.json", provenance)
        command = [sys.executable, "-m", "espnet2.bin.rec_rir_pit_inference",
                   "--train_config", str(folder / "config.yaml"), "--model_file", str(folder / "model.pth"),
                   "--sample_rate", "8000", "--rir_length", "16000", "--output_subtype", "FLOAT",
                   "--device", args.device]
        for stage, index in (("smoke", "smoke.scp"), ("inference", "wav.scp")):
            print(f"{label}: {stage}", flush=True)
            subprocess.run(command + ["--wav_scp", str(output / index), "--output_dir", str(folder / stage)], check=True)
        predictions += ["--prediction", label, str(folder / "inference/rir.scp")]
    # Re-score best97 as a control; this is separate from the ave5 loss comparison.
    control_label = "RecRIR_3loss_best97_control"
    predictions += ["--prediction", control_label, old_model["prediction_scp"]]
    subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_ls_direct_reverb.py")),
                    "--reference_dir", str(reference), *predictions, "--out_dir", str(output / "comparison")], check=True)
    report = json.loads((output / "comparison/comparison.json").read_text())
    if report["models"][control_label]["metrics"] != old_model["metrics"]:
        raise ValueError("best97 control differs from the prior LS evaluation")
    expected_ids = {line.split()[0] for line in rows}
    for label, model in report["models"].items():
        summary_path = Path(model["summary"])
        with summary_path.with_name("per_source.csv").open() as stream:
            scored = list(csv.DictReader(stream))
        if len(scored) != 6000 or {row["utt_id"] for row in scored} != expected_ids:
            raise ValueError(f"Incomplete scoring: {label}")
        if any(not math.isfinite(float(row[key])) for row in scored
               for key in ("rmse", "corr", "rt60_err", "drr_err", "c50_err")):
            raise ValueError(f"Nonfinite metrics: {label}")
    write_json(output / "final_checks.json", dict(
        count=3000, source_pairs_per_model=6000, all_primary_and_acoustic_metrics_finite=True,
        best97_control_exactly_reproduces_prior_summary=True,
        comparison_sha256=digest(output / "comparison/comparison.json"),
    ))
    write_json(output / "complete.json", dict(
        count=3000, models=3, git_commit=commit, elapsed_seconds=time.monotonic() - started,
    ))


if __name__ == "__main__":
    main()
