"""Same-GPU FP32 inference timing for pooled baseline and SPMamba CTFs."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import torch

from espnet2.tasks.rir import RIRTask


def measure(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = function()
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite inference output")
    return dict(median_ms=statistics.median(times), min_ms=min(times),
                max_ms=max(times), samples_ms=times, shape=list(output.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 5:
        parser.error("Use a new output and at least five iterations")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES must identify the physical GPU")
    # Check BEFORE allocating a CUDA context. Never share an occupied GPU.
    rows = csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)))
    selected = next([v.strip() for v in row] for row in rows if int(row[0]) == args.gpu_index)
    applications = csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)))
    if int(selected[2]) >= 200 or any(row and row[0].strip() == selected[1] for row in applications):
        raise RuntimeError("Selected GPU is occupied")
    recipe = Path(__file__).resolve().parents[1]
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2"
    generator = torch.Generator().manual_seed(20261014)
    cpu_speech = torch.randn(64000, generator=generator)
    speech = cpu_speech.cuda()
    report = dict(host=socket.gethostname(), gpu_index=args.gpu_index,
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  sample_rate=16000, duration_seconds=4, batch_size=1, precision="float32",
                  input_sha256=hashlib.sha256(cpu_speech.numpy().tobytes()).hexdigest(),
                  weights="seed 0 random weights; architecture timing only",
                  timing="synchronized wall time; input already on GPU; excludes loading and disk IO",
                  ctf="waveform normalization + STFT + predictor + tap ordering",
                  rir="CTF inference + ordinary-sweep PIM yielding two 2-second RIRs",
                  warmup_ctf=10, iterations_ctf=args.iterations,
                  warmup_rir=2, iterations_rir=7, models={})
    for label in ("baseline", "spmamba_no_tfa", "spmamba_tfa", "baseline_repeat"):
        suffix = "" if label.startswith("baseline") else "_" + label
        path = recipe / "conf/tuning" / (prefix + suffix + ".yaml")
        config = RIRTask.get_parser().parse_args([
            "--config", str(path), "--output_dir", str(args.output.parent / "unused")])
        assert not config.use_amp and not config.init_param
        torch.manual_seed(0)
        model = RIRTask.build_model(config).cuda().eval()
        with torch.inference_mode():
            ctf = measure(lambda: model.estimate_ctf(speech), 10, args.iterations)
            rir = measure(lambda: model.estimate_rir(speech, rir_length=32000), 2, 7)
        assert ctf["shape"] == [2, 257, 60]
        assert rir["shape"] == [2, 32000]
        result = dict(parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      config=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      ctf=ctf, rir=rir)
        report["models"][label] = result
        print(label, result["parameters"], ctf["median_ms"], rir["median_ms"], flush=True)
        del model
        torch.cuda.empty_cache()
    base = report["models"]["baseline"]
    for result in report["models"].values():
        result["parameter_ratio"] = result["parameters"] / base["parameters"]
        result["ctf_time_ratio"] = result["ctf"]["median_ms"] / base["ctf"]["median_ms"]
        result["rir_time_ratio"] = result["rir"]["median_ms"] / base["rir"]["median_ms"]
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
