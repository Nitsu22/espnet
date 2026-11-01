"""Prepare private WHAMR metadata and check the exact Eq. (10) training model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.enh import EnhancementTask

SETS = {"train": ("tr", 20000), "valid": ("cv", 5000), "test": ("tt", 3000)}
FILES = {"wav.scp": "speech_mix", "spk1.scp": "speech_ref1", "spk2.scp": "speech_ref2"}


def read_map(path):
    pairs = [line.strip().split(maxsplit=1) for line in path.open()]
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError(f"Duplicate keys in {path}")
    return result


def prepare(args):
    source = args.source_recipe.resolve()
    report, seen = {}, set()
    for split, (prefix, expected) in SETS.items():
        name = f"{prefix}_mix_both_reverb_min_8k"
        src = source / "dump/raw" / name
        dest = args.output / "dump/raw" / name
        dest.mkdir(parents=True, exist_ok=True)
        lengths = read_map(src / "utt2num_samples")
        if len(lengths) != expected or seen.intersection(lengths):
            raise ValueError(f"Unexpected or overlapping {split} keys")
        seen.update(lengths)
        shapes = args.output / "enh_stats_8k" / split
        shapes.mkdir(parents=True, exist_ok=True)
        sampled = {
            next(iter(lengths)),
            list(lengths)[len(lengths) // 2],
            list(lengths)[-1],
            max(lengths, key=lambda k: int(lengths[k])),
        }
        hashes = {}
        for filename, key in FILES.items():
            scp = read_map(src / filename)
            if set(scp) != set(lengths):
                raise ValueError(f"Mismatched keys in {src / filename}")
            lines = []
            for uid, wav in scp.items():
                path = Path(wav)
                if not path.is_absolute():
                    path = source / path
                # Keep the actual prepared waveform unchanged; only absolutize metadata.
                if not path.is_file():
                    raise FileNotFoundError(path)
                if uid in sampled:
                    info = sf.info(path)
                    if (info.samplerate, info.channels, info.frames) != (
                        8000, 2, int(lengths[uid])
                    ):
                        raise ValueError(f"Unexpected waveform header: {path}: {info}")
                lines.append(f"{uid} {path}\n")
            text = "".join(lines)
            (dest / filename).write_text(text)
            hashes[filename] = hashlib.sha256(text.encode()).hexdigest()
            (shapes / f"{key}_shape").write_text("".join(f"{u} {n}\n" for u, n in lengths.items()))
        (dest / "utt2num_samples").write_text("".join(f"{u} {n}\n" for u, n in lengths.items()))
        (dest / "feats_type").write_text("raw\n")
        report[split] = {
            "utterances": expected,
            "checked_all_paths": True,
            "header_checked_keys": sorted(sampled),
            "scp_sha256": hashes,
        }
        print(f"Prepared {split}: {expected} utterances", flush=True)
    (args.output / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")


def check(args):
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if args.gpu else "cpu")
    opts = EnhancementTask.get_parser().parse_args(
        ["--config", str(args.config), "--output_dir", str(args.output / "smoke")]
    )
    model = EnhancementTask.build_model(opts).to(device)
    assert sum(p.numel() for p in model.parameters()) == 5425552
    preprocess = EnhancementTask.build_preprocess_fn(opts, train=True)
    root = args.output / "dump/raw/tr_mix_both_reverb_min_8k"
    scps = {key: read_map(root / filename) for filename, key in FILES.items()}
    lengths = read_map(root / "utt2num_samples")
    keys = [u for u, n in lengths.items() if int(n) >= 32000][:4]
    data = []
    for uid in keys:
        raw = {}
        for key, scp in scps.items():
            raw[key], fs = sf.read(scp[uid], dtype="float32", always_2d=True)
            assert fs == 8000
        np.random.seed(0)
        excess = len(raw["speech_mix"]) - 32000
        start = np.random.randint(0, excess) if excess > 0 else 0
        peak = np.max(np.abs(raw["speech_mix"][start:start + 32000]))
        np.random.seed(0)
        processed = preprocess(uid, {k: v.copy() for k, v in raw.items()})
        for key in FILES.values():
            expected = raw[key][start:start + 32000, 0] / max(1.0, peak)
            np.testing.assert_allclose(processed[key], expected, atol=1e-7)
        data.append(processed)
    samples = 32000 if args.gpu else 1024
    batch = {
        key: torch.tensor(np.stack([d[key][:samples] for d in data]), device=device)
        for key in FILES.values()
    }
    batch["speech_mix_lengths"] = torch.full((4,), samples, device=device)
    optimizer = torch.optim.Adam(model.parameters(), **opts.optim_conf)
    timings = []
    for step in range(args.steps if args.gpu else 1):
        optimizer.zero_grad(set_to_none=True)
        if args.gpu:
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss = model(**batch)[0]
        if not torch.isfinite(loss).all():
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        if not all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        ):
            raise RuntimeError("Missing or nonfinite gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), opts.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        if args.gpu:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        print(
            f"step={step + 1} loss={loss.item():.6f} "
            f"grad_norm={norm.item():.6f} seconds={timings[-1]:.3f}", flush=True
        )
    model.eval()
    with torch.no_grad():
        loss = model(**batch)[0]
        swapped = dict(batch, speech_ref1=batch["speech_ref2"], speech_ref2=batch["speech_ref1"])
        torch.testing.assert_close(loss, model(**swapped)[0])
        scaled = {k: (v * 3 if k != "speech_mix_lengths" else v) for k, v in batch.items()}
        torch.testing.assert_close(loss, model(**scaled)[0], atol=2e-4, rtol=2e-4)
        # Full-length validation, including the longest validation utterance.
        validation = args.output / "dump/raw/cv_mix_both_reverb_min_8k"
        valid_lengths = read_map(validation / "utt2num_samples")
        uid = max(valid_lengths, key=lambda u: int(valid_lengths[u]))
        valid_batch = {}
        for filename, key in FILES.items():
            wav, fs = sf.read(
                read_map(validation / filename)[uid],
                dtype="float32", always_2d=True,
            )
            if not args.gpu:
                wav = wav[:1024]
            valid_batch[key] = torch.tensor(wav[:, 0], device=device)[None]
        valid_batch["speech_mix_lengths"] = torch.tensor(
            [valid_batch["speech_mix"].shape[1]], device=device
        )
        valid_loss = model(**valid_batch)[0]
        assert torch.isfinite(valid_loss).all()
    result = {"device": str(device), "parameters": 5425552, "batch_size": 4, "samples": samples,
              "median_step_seconds": float(np.median(timings[2:] or timings)),
              "validation_samples": int(valid_batch["speech_mix_lengths"][0]),
              "validation_loss": valid_loss.item(), "passed": True}
    if args.gpu:
        result["gpu_name"] = torch.cuda.get_device_name()
        result["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
    dest = args.output / "smoke"
    dest.mkdir(exist_ok=True)
    checkpoint = dest / ("gpu.pth" if args.gpu else "cpu.pth")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored["model"][key])
    result_path = dest / ("gpu.json" if args.gpu else "cpu.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "check"])
    parser.add_argument("--source-recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (prepare if args.mode == "prepare" else check)(args)
