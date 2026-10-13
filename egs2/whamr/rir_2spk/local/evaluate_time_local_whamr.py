#!/usr/bin/env python3
"""Evaluate the trained local-attention checkpoint with physical and LS RIRs."""

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import soundfile as sf

from evaluate_two_speaker import read_scp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def validate_score(folder, ids):
    summary = json.loads((folder / "summary.json").read_text())
    protocol = dict(num_utts=len(ids), num_source_pairs=2 * len(ids),
                    sample_rate=16000, rir_length=32000, align="peak",
                    scale_mode="peak", pit_metric="rmse", polarity_align=False,
                    direct_index_source="reference_peak")
    for key, value in protocol.items():
        if summary[key] != value:
            raise ValueError(f"Scoring protocol/count mismatch: {key}")
    with (folder / "per_source.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row["utt_id"] for row in rows)
    if set(counts) != ids or set(counts.values()) != {2}:
        raise ValueError("Expected two scored sources for every test utterance")
    for uid in ids:
        pairs = [row for row in rows if row["utt_id"] == uid]
        if ({r["pred_index"] for r in pairs} != {"0", "1"}
                or {r["ref_index"] for r in pairs} != {"0", "1"}):
            raise ValueError("Invalid PIT assignment")
    fields = dict(rmse="rmse_mean", rmse_50ms="rmse_50ms_mean",
                  rmse_direct_50ms="rmse_direct_50ms_mean", corr="corr_mean",
                  corr_direct_50ms="corr_direct_50ms_mean", rt60_err="rt60_mae",
                  drr_err="drr_mae", c50_err="c50_mae")
    valid_counts = {}
    for field, aggregate in fields.items():
        values = [float(row[field]) for row in rows]
        finite = [v for v in values if math.isfinite(v)]
        valid_counts[field] = len(finite)
        if not finite or (field.startswith("rmse") and len(finite) != len(rows)):
            raise ValueError(f"Invalid waveform metric: {field}")
        if field.endswith("_err"):
            finite = [abs(v) for v in finite]
        if not math.isclose(math.fsum(finite) / len(finite), summary[aggregate],
                            rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Metric aggregate mismatch: {field}")
    write_json(folder / "valid_metric_counts.json", valid_counts)
    return dict(summary=summary, valid_counts=valid_counts)


def require_free_gpu(index):
    rows = csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits"], text=True)))
    selected = next([v.strip() for v in r] for r in rows if int(r[0]) == index)
    apps = csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True)))
    if int(selected[2]) >= 200 or any(r and r[0].strip() == selected[1] for r in apps):
        raise RuntimeError("Selected GPU is occupied; refusing to start inference")


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("experiment", "training_dir", "data", "ls_reference_dir",
                 "ls_helper_dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu_index", type=int, required=True)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parents[1]
    repo = recipe.parents[2]
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh evaluation output directory")
    trained = json.loads((args.training_dir / "training_complete.json").read_text())
    if (args.training_dir / "exit_status").read_text().strip() != "0":
        raise ValueError("Training has not finished successfully")
    checkpoint = args.experiment / "valid.loss.best.pth"
    if (checkpoint.resolve() != Path(trained["checkpoint"]).resolve()
            or digest(checkpoint) != trained["checkpoint_sha256"]
            or digest(args.experiment / "config.yaml") != trained["config_sha256"]):
        raise ValueError("Selected checkpoint/config differs from completed training")
    ids = set(read_scp(args.data / "wav.scp"))
    if len(ids) != 3000:
        raise ValueError("Expected the complete 3000-mixture WHAMR test set")
    for speaker in (1, 2):
        if set(read_scp(args.data / f"rir_ref{speaker}.scp")) != ids:
            raise ValueError("Physical reference IDs differ")
    ls = json.loads((args.ls_reference_dir / "preparation.json").read_text())
    if (ls["count"] != 3000 or not ls["full_dataset"]
            or ls["config"]["sample_rate"] != 16000
            or ls["config"]["rir_length"] != 32000):
        raise ValueError("Incompatible LS reference")
    for name, checksum in ls["source_index_sha256"].items():
        if digest(args.data / name) != checksum:
            raise ValueError("LS reference belongs to different physical inputs")
    ls_rows = [line.split() for line in
               (args.ls_reference_dir / "reference.scp").read_text().splitlines()]
    if len(ls_rows) != 3000 or {r[0] for r in ls_rows} != ids:
        raise ValueError("LS reference IDs differ")
    output.mkdir(parents=True)
    phase = "ls_reference_validation"
    try:
        ls_env = dict(os.environ)
        ls_env["PYTHONPATH"] = str(args.ls_helper_dir.resolve().parents[3])
        ls_env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, str(args.ls_helper_dir / "prepare_ls_rir_reference.py"),
                        "--mode", "validate", "--output", str(args.ls_reference_dir)],
                       env=ls_env, check=True)
        require_free_gpu(args.gpu_index)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_index),
                   PYTHONPATH=str(repo), TZ="Asia/Tokyo")
        started = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
                       host=socket.gethostname(), gpu_index=args.gpu_index,
                       checkpoint=str(checkpoint.resolve()),
                       checkpoint_sha256=trained["checkpoint_sha256"],
                       config_sha256=trained["config_sha256"],
                       evaluation_commit=subprocess.check_output(
                           ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                       ls_reference_preparation_sha256=digest(
                           args.ls_reference_dir / "preparation.json"))
        write_json(output / "started.json", started)
        evaluator = recipe / "local/evaluate_two_speaker.py"
        def evaluate(stage, destination, limit=0):
            subprocess.run([sys.executable, str(evaluator), "--stage", stage,
                            "--experiment", str(args.experiment), "--data", str(args.data),
                            "--output", str(destination), "--limit", str(limit)],
                           env=env, check=True)
        phase = "smoke"
        evaluate("all", output / "smoke_whamr", 2)
        require_free_gpu(args.gpu_index)
        phase = "inference"
        evaluate("inference", output / "whamr")
        env["CUDA_VISIBLE_DEVICES"] = ""
        phase = "physical_scoring"
        evaluate("score", output / "whamr")
        prediction_scp = output / "whamr/inference/rir.scp"
        predictions = [line.split() for line in prediction_scp.read_text().splitlines()]
        if len(predictions) != 3000 or {r[0] for r in predictions} != ids:
            raise ValueError("Incomplete prediction index")
        for row in predictions:
            if len(row) != 3:
                raise ValueError("Expected two predicted RIRs")
            for path in row[1:]:
                info = sf.info(path)
                if (info.samplerate != 16000 or info.frames != 32000
                        or info.channels != 1 or info.subtype != "FLOAT"):
                    raise ValueError("Predicted RIR header differs")
        metadata = json.loads((output / "whamr/metadata.json").read_text())
        if (metadata["checkpoint_sha256"] != trained["checkpoint_sha256"]
                or metadata["inference"] != "ordinary sweep PIM"
                or digest(output / "whamr/config.yaml") != trained["config_sha256"]):
            raise ValueError("Inference checkpoint/config/protocol differs")
        phase = "ls_scoring"
        ls_score = output / "whamr/score_ls_direct_reverb"
        subprocess.run([sys.executable, str(recipe / "local/score_rir_pit.py"),
                        "--ref_rir_scp", str(args.ls_reference_dir / "reference.scp"),
                        "--pred_rir_scp", str(prediction_scp), "--out_dir", str(ls_score),
                        "--sample_rate", "16000", "--rir_length", "32000",
                        "--align", "peak", "--scale_mode", "peak", "--pit_metric", "rmse"],
                       env=env, check=True)
        phase = "finalization"
        report = dict(started, finished_at=datetime.datetime.now().astimezone().isoformat(),
                      count=3000, source_pairs=6000,
                      physical=validate_score(output / "whamr/score", ids),
                      ls_direct_reverb=validate_score(ls_score, ids))
        write_json(ls_score / "ls_reference_metadata.json", dict(
            reference_type="ls_direct_reverb",
            reference_preparation_sha256=started["ls_reference_preparation_sha256"],
            acoustic_metrics="relative-transfer diagnostics, not physical room parameters"))
        write_json(output / "evaluation_complete.json", report)
        write_json(output / "exit_status.json", dict(exit_status=0, phase="complete"))
    except Exception:
        write_json(output / "exit_status.json", dict(exit_status=1, failed_phase=phase))
        raise


if __name__ == "__main__":
    main()
