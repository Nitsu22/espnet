#!/usr/bin/env python3
"""Run a trained frequency variant with ordinary PIM and shared LS scoring."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import soundfile as sf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", required=True,
                        choices=["freq_post_bimamba", "freq_both_bimamba"])
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--result_recipe", required=True, type=Path)
    parser.add_argument("--inference_repo", required=True, type=Path)
    args = parser.parse_args()
    recipe = args.result_recipe.resolve()
    inference_repo = args.inference_repo.resolve()
    local = Path(__file__).resolve().parent
    run = recipe / "exp/frequency_ls_evaluation_20261014" / args.variant
    if run.exists():
        raise ValueError(f"Use a fresh result directory: {run}")
    model = recipe / f"exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_{args.variant}"
    data = recipe / "dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k"
    reference = recipe / "dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_16k"
    trained = json.loads((recipe / "exp/frequency_module_training_20261012" /
                          args.variant / "training_complete.json").read_text())
    if digest(model / "valid.loss.best.pth") != trained["checkpoint_sha256"]:
        raise ValueError("Best checkpoint differs from the training completion record")
    if digest(model / "config.yaml") != trained["config_sha256"]:
        raise ValueError("Training configuration differs")
    gpu_rows = list(csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used",
        "--format=csv,noheader,nounits"], text=True))))
    selected = next([v.strip() for v in row] for row in gpu_rows
                    if int(row[0]) == args.gpu)
    apps = list(csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True))))
    if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1]
                                    for row in apps):
        raise RuntimeError("Selected GPU is occupied; refusing to start")
    run.mkdir(parents=True)
    temporary = run / ".tmp"
    temporary.mkdir()
    status = 1
    phase = "preflight"
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TZ="Asia/Tokyo",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TMPDIR=str(temporary))
    for key in ("NUMBA_CACHE_DIR", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR", "TORCH_HOME"):
        environment[key] = str(temporary / key.lower())
    environment["PYTHONPATH"] = str(inference_repo)
    def execute(command, code_root=None):
        selected_environment = environment.copy()
        if code_root is not None:
            selected_environment["PYTHONPATH"] = str(code_root)
        subprocess.run([sys.executable, "-B"] + command,
                       env=selected_environment, cwd=inference_repo, check=True)
    try:
        metadata = dict(host=socket.gethostname(), gpu_index=args.gpu,
                        gpu_uuid=selected[1], gpu=selected[2], variant=args.variant,
                        started_at=datetime.datetime.now().astimezone().isoformat(),
                        inference_commit=subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=inference_repo,
                            text=True).strip(),
                        evaluation_commit=subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=local,
                            text=True).strip(), checkpoint_sha256=trained["checkpoint_sha256"])
        (run / "started.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Include the longest test input before committing to a full evaluation.
        maps = {name: dict(line.split(maxsplit=1) for line in
                           (data / name).read_text().splitlines())
                for name in ("wav.scp", "rir_ref1.scp", "rir_ref2.scp")}
        if any(set(values) != set(maps["wav.scp"]) for values in maps.values()):
            raise ValueError("Mixture/reference IDs differ")
        ordered = sorted(maps["wav.scp"], key=lambda uid:
                         sf.info(maps["wav.scp"][uid]).frames)
        if len(ordered) != 3000:
            raise ValueError("Expected the full 3,000-utterance test set")
        smoke = run / "smoke_data"
        smoke.mkdir()
        ids = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        for name, entries in maps.items():
            (smoke / name).write_text("".join(f"{uid} {entries[uid]}\n" for uid in ids))
        evaluator = inference_repo / "egs2/whamr/rir_2spk/local/evaluate_two_speaker.py"
        common = [str(evaluator), "--experiment", str(model), "--device", "cuda:0"]
        phase = "smoke"
        execute(common + ["--data", str(smoke), "--output", str(run / "smoke_whamr")])
        phase = "full_inference_and_physical_scoring"
        execute(common + ["--data", str(data), "--output", str(run / "whamr")])
        phase = "ls_scoring"
        execute([str(local / "evaluate_ls_direct_reverb.py"),
                 "--reference_dir", str(reference), "--prediction", args.variant,
                 str(run / "whamr/inference/rir.scp"), "--out_dir", str(run / "ls"),
                 "--no_comparison_csv"], code_root=local.parents[3])
        result = json.loads((run / "ls/comparison.json").read_text())
        scored = result["models"][args.variant]
        if (result["count"] != 3000 or result["source_pairs"] != 6000 or
                any(n != 6000 for n in scored["valid_counts"].values())):
            raise ValueError("Incomplete or nonfinite LS metrics")
        if (run / "ls/comparison.csv").exists():
            raise RuntimeError("Unexpected comparison CSV")
        (run / "evaluation_complete.json").write_text(json.dumps(dict(
            finished_at=datetime.datetime.now().astimezone().isoformat(),
            count=3000, source_pairs=6000, reference_preparation_sha256=
            result["reference_preparation_sha256"],
            comparison_sha256=digest(run / "ls/comparison.json")), indent=2) + "\n")
        phase, status = "complete", 0
    finally:
        (run / "exit_status").write_text(f"{status}\n")
        (run / "final_phase").write_text(phase + "\n")
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
