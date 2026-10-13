"""Verify Stage 5 artifacts and evaluate the longest real validation example."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from check_linked_dump import read_scp

from espnet2.enh.damsep.espnet_model import ESPnetDAMSEPModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("conf/tuning/train_damsep_nf_8k.yaml")
    )
    parser.add_argument("--statsdir", type=Path, default=Path("exp/damsep_stats_8k"))
    parser.add_argument(
        "--report", type=Path, default=Path("exp/damsep_nf_8k/input_check.json")
    )
    parser.add_argument("--check_backward", action="store_true")
    parser.add_argument("--check_inference", action="store_true")
    args = parser.parse_args()
    checked = json.loads((args.statsdir / "data_check.json").read_text())
    for path, digest in checked["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Stage 5 source list changed: {path}")
    fields = (
        "speech_mix",
        "speech_ref1",
        "speech_ref2",
        "speech_reverb1",
        "speech_reverb2",
    )
    for split, source_split in (("train", "tr"), ("valid", "cv")):
        shapes = []
        for field in fields:
            path = args.statsdir / split / (field + "_shape")
            shape = dict(
                line.split(maxsplit=1) for line in path.read_text().splitlines()
            )
            if len(shape) != checked["splits"][source_split]["count"]:
                raise RuntimeError(f"Incomplete Stage 5 shapes: {path}")
            shapes.append(shape)
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise RuntimeError(f"Unaligned Stage 5 shapes: {split}")
        if split == "train":
            train_uid = next(iter(shapes[0]))
    valid_shapes = shapes[0]
    uid = max(valid_shapes, key=lambda key: int(valid_shapes[key].split(",")[0]))
    samples = int(valid_shapes[uid].split(",")[0])
    base = Path("dump_clean/raw/cv_mix_clean_reverb_min_8k")
    rev = Path("dump_reverb/raw/cv_mix_both_reverb_min_8k")
    paths = (
        base / "wav.scp",
        base / "spk1.scp",
        base / "spk2.scp",
        rev / "spk1_reverb.scp",
        rev / "spk2_reverb.scp",
    )
    batch = {}
    for field, path in zip(fields, paths):
        audio, rate = sf.read(read_scp(path)[uid], dtype="float32", always_2d=True)
        if rate != 8000 or len(audio) != samples or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid validation audio: {field}, {uid}")
        batch[field] = torch.from_numpy(audio[:, 0].copy())[None].cuda()
    batch["speech_mix_lengths"] = torch.tensor([samples], device="cuda")
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["seed"])
    model = ESPnetDAMSEPModel(**config["model_conf"]).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        loss, stats, _ = model(**batch)
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or any(not torch.isfinite(v) for v in stats.values()):
        raise RuntimeError("Nonfinite full-length validation losses")
    backward_check = None
    if args.check_backward:
        del batch
        base_train = Path("dump_clean/raw/tr_mix_clean_reverb_min_8k")
        rev_train = Path("dump_reverb/raw/tr_mix_both_reverb_min_8k")
        train_paths = (
            base_train / "wav.scp",
            base_train / "spk1.scp",
            base_train / "spk2.scp",
            rev_train / "spk1_reverb.scp",
            rev_train / "spk2_reverb.scp",
        )
        train_batch = {}
        segment = config["preprocessor_conf"]["speech_segment"]
        for field, path in zip(fields, train_paths):
            audio, rate = sf.read(
                read_scp(path)[train_uid], dtype="float32", always_2d=True
            )
            if rate != 8000 or not np.isfinite(audio).all():
                raise RuntimeError(f"Invalid training audio: {field}, {train_uid}")
            train_batch[field] = torch.from_numpy(audio[:segment, 0].copy())[
                None
            ].cuda()
        train_samples = train_batch["speech_mix"].shape[1]
        train_batch["speech_mix_lengths"] = torch.tensor([train_samples], device="cuda")
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], **config["optim_conf"]
        )
        train_loss, _, _ = model(**train_batch)
        if not torch.isfinite(train_loss):
            raise RuntimeError("Nonfinite real training loss")
        train_loss.backward()
        invalid = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        if invalid:
            raise RuntimeError(f"Missing/nonfinite training gradients: {invalid}")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("Inactive CTF parameters received gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        torch.cuda.synchronize()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite parameters after real Adam update")
        backward_check = dict(
            status="passed",
            uid=train_uid,
            samples=train_samples,
            train_loss=float(train_loss.detach()),
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            frozen_parameters=sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            ),
            finite_gradients=True,
            optimizer_step=True,
        )
    inference_check = None
    if args.check_inference:
        # Test-set utterances can exceed the longest validation utterance.
        # Run the same CUDA separation path used by Stage 7 before training.
        test_paths = read_scp(Path("dump_clean/raw/tt_mix_clean_reverb_min_8k/wav.scp"))
        test_uid = max(test_paths, key=lambda key: sf.info(test_paths[key]).frames)
        audio, rate = sf.read(test_paths[test_uid], dtype="float32", always_2d=True)
        if rate != 8000 or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid test audio: {test_uid}")
        model.eval()
        with torch.no_grad():
            estimate = model.separate(
                torch.from_numpy(audio[:, 0].copy())[None].cuda()
            )["x_derev"]
        if estimate.shape != (1, 2, len(audio)) or not torch.isfinite(estimate).all():
            raise RuntimeError("Invalid longest-test CUDA separation output")
        torch.cuda.synchronize()
        inference_check = dict(
            status="passed", uid=test_uid, samples=len(audio), seconds=len(audio) / 8000
        )
    report = dict(
        status="passed",
        purpose="Stage 5 consistency and longest full-length validation CUDA check",
        device=torch.cuda.get_device_name(0),
        validation_id=uid,
        samples=samples,
        seconds=samples / 8000,
        dtype="float32",
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        stats={k: float(v) for k, v in stats.items()},
        clean_only=model.clean_only,
        architecture=model.network.architecture,
        backward_check=backward_check,
        inference_check=inference_check,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
