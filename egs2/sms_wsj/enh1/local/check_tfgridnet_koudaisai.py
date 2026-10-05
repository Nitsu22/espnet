#!/usr/bin/env python3
"""Check the ablation objective and a disposable forward/backward, no training."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist

from espnet2.enh.loss.criterions.tfgridnet_roland import RolandTFL1, RolandTFMCPIT
from espnet2.tasks.enh import EnhancementTask


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_loss(roland_root):
    torch.manual_seed(7)
    criterion = RolandTFL1()
    wrapper = RolandTFMCPIT(criterion)
    refs = [torch.randn(2, 1024) for _ in range(2)]
    estimates = [torch.randn(2, 1024, requires_grad=True) for _ in range(2)]
    # Intentionally inconsistent spectra distinguish raw RI loss from re-STFT loss.
    spectra = [torch.randn(2, 17, 129, dtype=torch.complex64, requires_grad=True)
               for _ in range(2)]
    lens, scale = torch.tensor([1024, 1024]), torch.tensor([[0.7], [1.4]])
    others = dict(tfgridnet_mix_std=scale, tfgridnet_lengths=lens,
                  tfgridnet_spectra=spectra)
    loss, _, _ = wrapper(refs, estimates, others)
    swapped, _, _ = wrapper(refs[::-1], estimates, others)
    torch.testing.assert_close(loss, swapped)
    if roland_root:
        root = Path(roland_root) / "espnet2/enh/loss"
        original_criterion = load_module("original_criterion", root / "criterions/multitask_doa_sep_loss.py")
        original_wrapper = load_module("original_wrapper", root / "wrappers/pit_solver.py")
        baseline = original_wrapper.PITSolverMCDOAMultitask(
            original_criterion.DirectionalTFL1Multitask(),
            multitask_weights=[0.95, 0.0],
        )
        norm_refs = [r / scale for r in refs]
        old_loss, _, _ = baseline(refs, [s / scale for s in estimates], dict(
            ref_normalized=norm_refs,
            tf_ref=[criterion.encoder(r, lens)[0] for r in norm_refs],
            tf_out=spectra,
            ref_doa=[torch.zeros(2, dtype=torch.long) for _ in range(2)],
            inf_doa=[torch.zeros(2, 360) for _ in range(2)],
        ))
        torch.testing.assert_close(0.95 * loss, old_loss)
        new_grad = torch.autograd.grad(0.95 * loss, estimates + spectra, retain_graph=True)
        old_grad = torch.autograd.grad(old_loss, estimates + spectra)
        for new, old in zip(new_grad, old_grad):
            torch.testing.assert_close(new, old)
        print("Original Roland objective and gradients match with DOA weight zero")
    print("Permutation invariance OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--roland-root")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not args.ddp:
        check_loss(args.roland_root)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.ddp:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl")
        assert dist.get_world_size() == 4
    device = torch.device(f"cuda:{rank}" if args.ddp else "cpu")
    torch.manual_seed(0)
    config = "conf/tuning/train_tfgridnet_koudaisai_1ch.yaml"
    task_args = EnhancementTask.get_parser().parse_args([
        "--config", config, "--output_dir", "unused-smoke-output",
    ])
    model = EnhancementTask.build_model(task_args).to(device)
    assert model.separator.n_layers == 2 and model.separator.n_imics == 1
    assert not any("doa" in n for n, _ in model.named_parameters())
    preprocess = EnhancementTask.build_preprocess_fn(task_args, train=True)
    if args.ddp:
        root = Path("/gs/bs/tga-shinoda/nitsu/data/sms_wsj_4mic_koudaisai/dump_4mic/raw/train_si284_4mic")
        scps = [dict(line.split(maxsplit=1) for line in (root / n).read_text().splitlines())
                for n in ("wav.scp", "spk1.scp", "spk2.scp")]
        key = list(scps[0])[rank]
        arrays = [sf.read(scp[key], dtype="float32")[0] for scp in scps]
        arrays = [a[:32000] for a in arrays]
    else:
        arrays = [np.random.default_rng(i).normal(0, 0.01, (1024, 4)).astype("float32")
                  for i in range(3)]
        key = "synthetic"
    names = ("speech_mix", "speech_ref1", "speech_ref2")
    data = preprocess(key, dict(zip(names, arrays)))
    for name, array in zip(names, arrays):
        np.testing.assert_array_equal(data[name], array[:, 0])
    batch = {name: torch.from_numpy(data[name]).unsqueeze(0).to(device) for name in names}
    batch["speech_mix_lengths"] = torch.tensor([len(data["speech_mix"])], device=device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    loss, stats, _ = model(**batch)
    assert torch.isfinite(loss), stats
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    print(json.dumps({"rank": rank, "samples": len(data["speech_mix"]),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "loss": loss.item(), "finite_gradients": True,
                      "optimizer_steps": 0}), flush=True)
    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
