"""One synthetic batch verifies DAMSEP's CUDA forward/backward/optimizer paths.

This is a compatibility check, not training on a dataset or a speed benchmark.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from espnet2.enh.damsep.espnet_model import ESPnetDAMSEPModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("conf/tuning/train_damsep_nf_8k.yaml")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("exp/cuda_check/report.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = ESPnetDAMSEPModel(
        **yaml.safe_load(args.config.read_text())["model_conf"]
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    signals = [torch.randn(1, 32000, device="cuda") * 0.02 for _ in range(4)]
    clean1, clean2, reverb1, reverb2 = signals
    torch.cuda.reset_peak_memory_stats()
    loss, _, _ = model(
        speech_mix=reverb1 + reverb2,
        speech_mix_lengths=torch.tensor([32000], device="cuda"),
        speech_ref1=clean1,
        speech_ref2=clean2,
        speech_reverb1=reverb1,
        speech_reverb2=reverb2,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite CUDA loss")
    loss.backward()
    invalid = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    if invalid:
        raise RuntimeError(f"Missing or nonfinite CUDA gradients: {invalid}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    torch.cuda.synchronize()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite parameters after CUDA optimizer step")
    from importlib.metadata import version

    report = dict(
        status="passed",
        purpose="synthetic CUDA compatibility; no dataset training",
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        mamba=version("mamba-ssm"),
        causal_conv1d=version("causal-conv1d"),
        sample_rate=8000,
        samples=32000,
        batch_size=1,
        dtype="float32",
        parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        finite_gradients=True,
        optimizer_step=True,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
