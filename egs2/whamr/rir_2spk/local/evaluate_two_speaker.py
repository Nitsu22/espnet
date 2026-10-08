#!/usr/bin/env python3
"""Evaluate one two-speaker RIRTask checkpoint with shared ordinary-PIM scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read_scp(path):
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only; zero means all")
    parser.add_argument("--stage", choices=["all", "inference", "score"], default="all")
    parser.add_argument("--also_polarity_aligned", action="store_true",
                        help="Additionally save polarity-aligned scores for BUT comparisons")
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parents[1]
    repo = recipe.parents[2]
    experiment, data, output = (p.resolve() for p in (args.experiment, args.data, args.output))
    wavs = read_scp(data / "wav.scp")
    refs = [read_scp(data / f"rir_ref{i}.scp") for i in (1, 2)]
    if any(set(ref) != set(wavs) for ref in refs):
        raise ValueError("Mixture and reference IDs differ")
    ids = list(wavs)[:args.limit or None]
    if not ids:
        raise ValueError("Empty test set")
    if args.stage != "score":
        output.mkdir(parents=True, exist_ok=False)
        for name in ("config.yaml", "valid.loss.best.pth"):
            shutil.copy2(experiment / name, output / name)
        (output / "wav.scp").write_text("".join(f"{uid} {wavs[uid]}\n" for uid in ids))
        (output / "reference.scp").write_text("".join(
            f"{uid} {refs[0][uid]} {refs[1][uid]}\n" for uid in ids))
    metadata = dict(experiment=str(experiment), data=str(data), count=len(ids),
                    checkpoint="valid.loss.best.pth", sample_rate=16000, rir_length=32000,
                    input="full utterance, channel 0", inference="ordinary sweep PIM",
                    output_subtype="FLOAT", align="peak", scale_mode="peak",
                    pit_metric="rmse", reference="physical clean-to-reverb RIR",
                    checkpoint_sha256=hashlib.sha256((output / "valid.loss.best.pth").read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    if args.stage != "score":
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        saved = json.loads((output / "metadata.json").read_text())
        for key in ("experiment", "data", "count", "checkpoint_sha256"):
            if saved[key] != metadata[key]:
                raise ValueError(f"Scoring metadata differs: {key}")
        marker = json.loads((output / "inference_complete.json").read_text())
        if marker["count"] != len(ids):
            raise ValueError("Incomplete inference marker")
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    start = time.monotonic()
    if args.stage != "score":
        subprocess.run([sys.executable, "-m", "espnet2.bin.rec_rir_pit_inference",
                    "--train_config", str(output / "config.yaml"),
                    "--model_file", str(output / "valid.loss.best.pth"),
                    "--wav_scp", str(output / "wav.scp"), "--output_dir", str(output / "inference"),
                    "--sample_rate", "16000", "--rir_length", "32000",
                        "--output_subtype", "FLOAT", "--device", args.device], cwd=repo, check=True)
        inference_seconds = time.monotonic() - start
    else:
        inference_seconds = marker["inference_seconds"]
    predictions = (output / "inference/rir.scp").read_text().splitlines()
    if {row.split()[0] for row in predictions} != set(ids) or len(predictions) != len(ids):
        raise ValueError("Incomplete predictions")
    if args.stage != "score":
        (output / "inference_complete.json").write_text(json.dumps(dict(
            count=len(ids), inference_seconds=inference_seconds), indent=2) + "\n")
    if args.stage == "inference":
        return
    scoring_start = time.monotonic()
    score_command = [sys.executable, str(recipe / "local/score_rir_pit.py"),
                    "--ref_rir_scp", str(output / "reference.scp"),
                    "--pred_rir_scp", str(output / "inference/rir.scp"),
                    "--out_dir", str(output / "score"), "--sample_rate", "16000",
                    "--rir_length", "32000", "--align", "peak", "--scale_mode", "peak",
                     "--pit_metric", "rmse"]
    subprocess.run(score_command, check=True)
    if args.also_polarity_aligned:
        aligned = score_command.copy()
        aligned[aligned.index("--out_dir") + 1] = str(output / "score_polarity_aligned")
        subprocess.run(aligned + ["--polarity_align"], check=True)
    scoring_seconds = time.monotonic() - scoring_start
    for score_name in (["score", "score_polarity_aligned"]
                       if args.also_polarity_aligned else ["score"]):
        summary = json.loads((output / score_name / "summary.json").read_text())
        if summary["num_utts"] != len(ids) or summary["num_source_pairs"] != 2 * len(ids):
            raise ValueError(f"Incomplete scoring: {score_name}")
    (output / "complete.json").write_text(json.dumps(dict(
        count=len(ids), inference_seconds=inference_seconds,
        scoring_seconds=scoring_seconds,
        total_seconds=(time.monotonic() - start if args.stage == "all"
                       else inference_seconds + scoring_seconds)), indent=2) + "\n")


if __name__ == "__main__":
    main()
