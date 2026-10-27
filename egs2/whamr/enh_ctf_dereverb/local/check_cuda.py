#!/usr/bin/env python3
"""Charged full-size CUDA readiness/timing check; never use the trial queue."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.enh.layers.bidirectional_mamba import BidirectionalMamba
from espnet2.tasks.enh import EnhancementTask


FIELDS = ("speech_mix", "speech_ref1", "speech_ref2", "speech_reverb1", "speech_reverb2")
SCPS = ("wav.scp", "spk1.scp", "spk2.scp", "spk1_reverb.scp", "spk2_reverb.scp")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(config, paired_dump):
    return dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        config_sha256=digest(config),
        paired_dump=str(Path(paired_dump).resolve()),
        manifest_sha256=digest(Path(paired_dump) / "manifest.json"),
    )


def verify_report(path, config, paired_dump):
    report = json.loads(Path(path).read_text())
    for key, expected in identity(config, paired_dump).items():
        if report.get(key) != expected:
            raise ValueError(f"CUDA check {key} differs from this run")
    if not report.get("passed") or report.get("train", {}).get("batch_size") != 4:
        raise ValueError("A successful physical batch-four CUDA check is required")
    print("Verified CUDA report for this commit, configuration and paired data.")


def index(path):
    entries = {}
    for line in Path(path).read_text().splitlines():
        key, value = line.split(maxsplit=1)
        if key in entries:
            raise ValueError(f"Duplicate ID in {path}: {key}")
        entries[key] = value
    return entries


def kernel_check():
    """Compare the actual Mamba CUDA kernel with the official reference scan."""
    mixer = BidirectionalMamba(8, d_state=4, d_conv=4, backend="cuda").cuda()
    x = torch.randn(2, 13, 8, device="cuda", requires_grad=True)
    actual = mixer(x)
    actual_grad = torch.autograd.grad(actual.square().sum(), x)[0]
    mixer.backend = "torch"
    expected = mixer(x)
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=2e-4)
    return dict(
        output_max_error=float((actual - expected).abs().max()),
        gradient_max_error=float((actual_grad - expected_grad).abs().max()),
    )


def check(config, paired_dump, output, steps):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a CUDA report: {output}")
    if steps < 2:
        raise ValueError("Use at least two steps, including a warmup step")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(identity(config, paired_dump), passed=False)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one allocated CUDA device is required")
        options = EnhancementTask.get_parser().parse_args([
            "--config", str(config), "--output_dir", str(output.parent), "--ngpu", "1"
        ])
        np.random.seed(options.seed)
        torch.manual_seed(options.seed)
        torch.cuda.manual_seed_all(options.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        report["gpu"] = dict(
            name=torch.cuda.get_device_name(),
            total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        report["dependencies"] = {
            package: importlib.metadata.version(package)
            for package in ("mamba-ssm", "causal-conv1d", "transformers")
        }
        report["mamba_kernel"] = kernel_check()
        model = EnhancementTask.build_model(options).cuda().train()
        if len(model.separator.blocks) != 3 or hasattr(model.separator, "deconv"):
            raise ValueError("Expected three front-end blocks and no pre-CG Clean head")
        report["parameters"] = sum(p.numel() for p in model.parameters())
        paired_dump = Path(paired_dump)
        maps = {
            split: [index(paired_dump / "raw" / f"{split}_mix_both_reverb_min_8k" / scp)
                    for scp in SCPS]
            for split in ("tr", "cv")
        }
        train_shapes = index(paired_dump / "stats/train/speech_mix_shape")
        valid_shapes = index(paired_dump / "stats/valid/speech_mix_shape")

        def make_batch(split, ids, train):
            proc = EnhancementTask.build_preprocess_fn(options, train)
            examples = []
            for uid in ids:
                signals = {}
                for name, mapping in zip(FIELDS, maps[split]):
                    values, rate = sf.read(mapping[uid], dtype="float32", always_2d=True)
                    if rate != 8000 or not np.isfinite(values).all():
                        raise ValueError(f"Expected finite actual 8-kHz audio: {uid}")
                    signals[name] = values
                # Exercise the largest permitted training crop, not an edge
                # crop with less history. Every retry uses real waveform data.
                for _ in range(100):
                    example = proc(uid, {key: value.copy() for key, value in signals.items()})
                    if not train or len(example["speech_mix"]) == 40640:
                        break
                else:
                    raise RuntimeError(f"Could not obtain full context for {uid}")
                examples.append((uid, example))
            _, batch = EnhancementTask.build_collate_fn(options, train)(examples)
            return {key: value.cuda() for key, value in batch.items()}

        ids = [uid for uid, shape in train_shapes.items() if int(shape.split(",")[0]) >= 64000][:4]
        if len(ids) != 4:
            raise ValueError("Four long training utterances are required for the probe")
        batch = make_batch("tr", ids, True)
        if batch["speech_mix"].shape != (4, 40640):
            raise ValueError("Expected a physical batch of four full context crops")
        optimizer = torch.optim.AdamW(model.parameters(), **options.optim_conf)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, stats, _ = model(**batch)
            if not torch.isfinite(loss) or any(
                not torch.isfinite(value) for value in stats.values()
            ):
                raise RuntimeError("Nonfinite training loss or statistics")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError(f"Missing or nonfinite gradient: {name}")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), options.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite parameters after AdamW")
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            print(f"CUDA step {step+1}/{steps}: {durations[-1]:.3f}s", flush=True)
        report["train"] = dict(
            batch_size=4, input_samples=40640, target_samples=32000,
            utterances=ids, seconds_per_step=durations,
            mean_seconds_after_warmup=float(np.mean(durations[1:])),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            final_stats={key: float(value) for key, value in stats.items()},
        )
        optimizer.zero_grad(set_to_none=True)
        del optimizer, batch, loss, stats
        torch.cuda.empty_cache()
        model.eval()
        longest = max(valid_shapes, key=lambda uid: int(valid_shapes[uid].split(",")[0]))
        batch = make_batch("cv", [longest], False)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            loss, stats, _ = model(**batch)
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or any(not torch.isfinite(x) for x in stats.values()):
            raise RuntimeError("Nonfinite full-length validation statistics")
        report["validation"] = dict(
            utterance=longest, samples=int(batch["speech_mix_lengths"][0]),
            seconds=time.perf_counter() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            stats={key: float(value) for key, value in stats.items()},
        )
        report["passed"] = True
        print("CUDA readiness check PASSED", flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"CUDA report: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paired_dump", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify_report", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.verify_report:
        verify_report(args.verify_report, args.config, args.paired_dump)
    else:
        check(args.config, args.paired_dump, args.output, args.steps)


if __name__ == "__main__":
    main()
