"""Check linked data/statistics and real batch-four CUDA training readiness."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from espnet2.tasks.enh import EnhancementTask


def read_scp(path):
    entries = {}
    for line in path.read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in entries or value.endswith("|"):
            raise ValueError(f"Duplicate ID or unsupported audio pipe: {path}")
        entries[uid] = value
    return entries


def audio(path):
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != 16000 or values.shape[1] not in (1, 2) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite mono/stereo 16 kHz audio: {path}")
    return values


def check_data(condition):
    if condition != "nf_whamr":
        raise ValueError("Only NF-WHAMR is prepared at 16 kHz")
    dump = Path("dump_nf16k")
    report = json.loads((dump / "preparation.json").read_text())
    maps = {}
    for split in ("tr", "cv", "tt"):
        folder = dump / "raw" / (split + "_mix_clean_reverb_min_16k")
        maps[split] = [read_scp(folder / name) for name in ("wav.scp", "spk1.scp", "spk2.scp")]
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            path = folder / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != report["hashes"][str(path)]:
                raise ValueError("Input SCP changed after CPU verification")
    for split, phase in (("tr", "train"), ("cv", "valid")):
        lengths = read_scp(dump / "raw" / (split + "_mix_clean_reverb_min_16k") / "utt2num_samples")
        for field in ("speech_mix", "speech_ref1", "speech_ref2"):
            shapes = read_scp(Path("exp/enh_stats_16k") / phase / (field + "_shape"))
            if shapes != lengths:
                raise ValueError("Statistics differ from verified lengths")
    return report, maps, read_scp(Path("exp/enh_stats_16k/valid/speech_mix_shape"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("whamr", "nf_whamr"), required=True)
    parser.add_argument("--expdir", type=Path, required=True)
    args = parser.parse_args()
    report, maps, valid_shape = check_data(args.condition)
    config = Path("conf/tuning/train_enh_tflocoformer_s_nf_16k.yaml")
    saved = args.expdir / "config.yaml"
    if saved.exists():
        previous = yaml.safe_load(saved.read_text())
        expected = (
            "mix_both_reverb" if args.condition == "whamr" else "mix_clean_reverb"
        )
        if (
            expected not in previous["train_data_path_and_name_and_type"][0][0]
            or previous["batch_size"] != 4
            or previous["accum_grad"] != 1
            or previous["speech_segment"] != 64000
            or previous["encoder_conf"]["n_fft"] != 512
        ):
            raise ValueError(
                "Existing checkpoint belongs to a different condition/batch"
            )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This check requires exactly one allocated CUDA device")
    options = EnhancementTask.get_parser().parse_args(
        ["--config", str(config), "--output_dir", str(args.expdir), "--ngpu", "1"]
    )
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = EnhancementTask.build_model(options).cuda().train()
    fields = ("speech_mix", "speech_ref1", "speech_ref2")

    def batch(split, ids, training):
        proc = EnhancementTask.build_preprocess_fn(options, training)
        items = [
            (
                uid,
                proc(
                    uid, {name: audio(m[uid]) for name, m in zip(fields, maps[split])}
                ),
            )
            for uid in ids
        ]
        _, values = EnhancementTask.build_collate_fn(options, training)(items)
        return {
            k: v.cuda().float() if v.is_floating_point() else v.cuda()
            for k, v in values.items()
        }

    train_ids = []
    for uid, path in maps["tr"][0].items():
        if sf.info(path).frames >= 64000:
            train_ids.append(uid)
        if len(train_ids) == 4:
            break
    if len(train_ids) != 4:
        raise ValueError("Cannot find four full training segments")
    values = batch("tr", train_ids, True)
    assert values["speech_mix"].shape == (4, 64000)
    optimizer = torch.optim.AdamW(model.parameters(), **options.optim_conf)
    torch.cuda.reset_peak_memory_stats()
    loss, _, _ = model(**values)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite batch-four training loss")
    loss.backward()
    if any(
        p.grad is None or not torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    ):
        raise RuntimeError("Missing or nonfinite batch-four gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite parameters after the AdamW step")
    report["cuda_train"] = dict(
        batch_size=4,
        samples_per_example=64000,
        loss=float(loss),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        optimizer_step=True,
    )
    model.zero_grad(set_to_none=True)
    del optimizer, values, loss
    torch.cuda.empty_cache()
    model.eval()
    uid = max(valid_shape, key=lambda key: int(valid_shape[key].split(",")[0]))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        loss, _, _ = model(**batch("cv", [uid], False))
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite full-length validation loss")
    report["cuda_valid"] = dict(
        batch_size=1,
        uid=uid,
        samples=int(valid_shape[uid].split(",")[0]),
        loss=float(loss),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    report.update(
        status="passed",
        device=torch.cuda.get_device_name(0),
        dtype="float32",
        parameters=sum(p.numel() for p in model.parameters()),
    )
    args.expdir.mkdir(parents=True, exist_ok=True)
    (args.expdir / "input_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("hashes", "splits")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
