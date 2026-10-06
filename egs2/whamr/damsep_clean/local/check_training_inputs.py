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
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
