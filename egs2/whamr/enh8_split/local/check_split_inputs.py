"""Check Split-S data/statistics and real batch-four CUDA training readiness."""

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
    if rate != 8000 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite stereo 8 kHz audio: {path}")
    return values


def check_data(condition):
    noisy = condition == "whamr"
    dump = Path("dump" if noisy else "dump_clean")
    subset = "mix_both_reverb_min_8k" if noisy else "mix_clean_reverb_min_8k"
    stats = Path("../enh1/exp" if noisy else "../enh4_clean/exp") / "enh_stats_8k"
    report = dict(
        condition=condition, sample_rate=8000, channel=0, splits={}, hashes={}
    )
    maps_by_split = {}
    for split, count in (("tr", 20000), ("cv", 5000), ("tt", 3000)):
        folder = dump / "raw" / f"{split}_{subset}"
        paths = [folder / name for name in ("wav.scp", "spk1.scp", "spk2.scp")]
        maps = [read_scp(path) for path in paths]
        ids = sorted(maps[0])
        if len(ids) != count or any(set(m) != set(ids) for m in maps[1:]):
            raise ValueError(f"Unaligned or incomplete {condition} {split} SCPs")
        for path in paths:
            report["hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if split != "tt":
            phase = "train" if split == "tr" else "valid"
            shapes = []
            for name in ("speech_mix", "speech_ref1", "speech_ref2"):
                path = stats / phase / f"{name}_shape"
                shape = read_scp(path)
                if set(shape) != set(ids):
                    raise ValueError(f"Statistics IDs differ from the SCPs: {path}")
                shapes.append(shape)
                report["hashes"][str(path)] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            if any(s != shapes[0] for s in shapes[1:]):
                raise ValueError(f"Unaligned statistics for {condition} {split}")
        else:
            shapes = None
        # Both inputs must use the same clean teachers. WHAMR must contain
        # the existing NF mixture plus its existing WHAMR noise image.
        nf = Path("dump_clean/raw") / f"{split}_mix_clean_reverb_min_8k"
        whamr = Path("dump/raw") / f"{split}_mix_both_reverb_min_8k"
        nf_maps = [read_scp(nf / name) for name in ("wav.scp", "spk1.scp", "spk2.scp")]
        whamr_maps = [
            read_scp(whamr / name) for name in ("wav.scp", "spk1.scp", "spk2.scp")
        ]
        # The original recipe formats noise teachers for train/valid only;
        # test clean-source scoring does not need an isolated noise image.
        noise_path = whamr / "noise1.scp"
        noise = read_scp(noise_path) if noise_path.exists() else None
        if noise is None and split != "tt":
            raise ValueError(f"Missing train/valid noise image: {noise_path}")
        if any(
            set(m) != set(ids)
            for m in nf_maps + whamr_maps + ([noise] if noise else [])
        ):
            raise ValueError(f"WHAMR / NF-WHAMR ID mismatch: {split}")
        worst_noise, worst_ref = 0.0, 0.0
        sample_ids = ids[:: max(1, len(ids) // 8)][:8]
        for uid in sample_ids:
            current = [audio(m[uid]) for m in maps]
            if any(x.shape != current[0].shape for x in current[1:]):
                raise ValueError(f"Waveform shape mismatch: {uid}")
            if shapes and len(current[0]) != int(shapes[0][uid].split(",")[0]):
                raise ValueError(f"Stale waveform length in statistics: {uid}")
            nf_audio = [audio(m[uid]) for m in nf_maps]
            whamr_audio = [audio(m[uid]) for m in whamr_maps]
            noise_audio = audio(noise[uid]) if noise else None
            if any(
                x.shape != current[0].shape
                for x in nf_audio
                + whamr_audio
                + ([noise_audio] if noise_audio is not None else [])
            ):
                raise ValueError(f"Cross-condition audio lengths differ: {uid}")
            noise_error = (
                float(np.max(np.abs(whamr_audio[0] - nf_audio[0] - noise_audio)))
                if noise_audio is not None
                else 0.0
            )
            ref_error = max(
                float(np.max(np.abs(whamr_audio[i] - nf_audio[i]))) for i in (1, 2)
            )
            if noise_error > 1e-4 or ref_error > 4e-5:
                raise ValueError(
                    f"WHAMR/NF inputs or clean teachers differ: {uid}: "
                    f"{noise_error}, {ref_error}"
                )
            worst_noise = max(worst_noise, noise_error)
            worst_ref = max(worst_ref, ref_error)
        report["splits"][split] = dict(
            count=count,
            sampled_ids=sample_ids,
            noise_sum_checked=noise is not None,
            noise_sum_error=worst_noise if noise is not None else None,
            clean_teacher_error=worst_ref,
        )
        maps_by_split[split] = maps
    valid_shape = read_scp(stats / "valid/speech_mix_shape")
    return report, maps_by_split, valid_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("whamr", "nf_whamr"), required=True)
    parser.add_argument("--expdir", type=Path, required=True)
    args = parser.parse_args()
    report, maps, valid_shape = check_data(args.condition)
    config = Path("conf/tuning") / (
        "train_enh_tflocoformer_split_s_whamr_8k.yaml"
        if args.condition == "whamr"
        else "train_enh_tflocoformer_split_s_nf_8k.yaml"
    )
    saved = args.expdir / "config.yaml"
    if saved.exists():
        previous = yaml.safe_load(saved.read_text())
        expected_separator = yaml.safe_load(config.read_text())["separator_conf"]
        expected = (
            "mix_both_reverb" if args.condition == "whamr" else "mix_clean_reverb"
        )
        if (
            expected not in previous["train_data_path_and_name_and_type"][0][0]
            or previous["batch_size"] != 4
            or previous["accum_grad"] != 1
            or previous["separator"] != "tflocoformer_split"
            or previous["separator_conf"] != expected_separator
        ):
            raise ValueError(
                "Existing checkpoint belongs to a different "
                "condition/batch/architecture"
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
        if sf.info(path).frames >= 32000:
            train_ids.append(uid)
        if len(train_ids) == 4:
            break
    if len(train_ids) != 4:
        raise ValueError("Cannot find four full training segments")
    values = batch("tr", train_ids, True)
    assert values["speech_mix"].shape == (4, 32000)
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
        samples_per_example=32000,
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
