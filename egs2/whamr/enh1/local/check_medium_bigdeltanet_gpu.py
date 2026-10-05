#!/usr/bin/env python3
"""Check real-data training and the longest validation utterance on one GPU."""

from pathlib import Path

import soundfile as sf
import torch

from espnet2.tasks.enh import EnhancementTask


def read_scp(path):
    return dict(line.rstrip().split(maxsplit=1) for line in path.open())


def load_batch(dataset, key, limit=None):
    batch = {}
    for scp, name in (
        ("wav.scp", "speech_mix"),
        ("spk1.scp", "speech_ref1"),
        ("spk2.scp", "speech_ref2"),
    ):
        filename = read_scp(Path("dump/raw") / dataset / scp)[key]
        audio, rate = sf.read(filename, dtype="float32", always_2d=False)
        if rate != 8000 or audio.ndim != 1:
            raise ValueError(f"Expected mono 8 kHz audio: {filename}")
        if limit is not None:
            audio = audio[:limit]
        batch[name] = torch.from_numpy(audio).unsqueeze(0).cuda()
    batch["speech_mix_lengths"] = torch.tensor(
        [batch["speech_mix"].shape[1]], device="cuda"
    )
    return batch


def memory(label):
    torch.cuda.synchronize()
    print(
        f"{label}: peak allocated={torch.cuda.max_memory_allocated() / 2**30:.3f} GiB, "
        f"peak reserved={torch.cuda.max_memory_reserved() / 2**30:.3f} GiB",
        flush=True,
    )


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    config = "conf/tuning/train_enh_tflocoformer_medium_nocashe_bigdeltanet_1gpu.yaml"
    args = EnhancementTask.get_parser().parse_args([
        "--config", config, "--output_dir", "/tmp/deltanet-medium-memory-check",
    ])
    model = EnhancementTask.build_model(args).cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    lengths = read_scp(Path("exp/enh_stats_8k/train/speech_mix_shape"))
    key = next(k for k, n in lengths.items() if int(n.split(",")[0]) >= 32000)
    batch = load_batch("tr_mix_both_reverb_min_8k", key, limit=32000)
    optimizer = torch.optim.AdamW(model.parameters(), **args.optim_conf)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.accum_grad):
        loss, _, _ = model(**batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        (loss / args.accum_grad).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"Training loss: {loss.item():.6f}", flush=True)
    memory("Four accumulated real 4-second microbatches")
    del batch, optimizer, loss
    torch.cuda.empty_cache()
    lengths = read_scp(Path("exp/enh_stats_8k/valid/speech_mix_shape"))
    key = max(lengths, key=lambda k: int(lengths[k].split(",")[0]))
    batch = load_batch("cv_mix_both_reverb_min_8k", key)
    print(f"Longest validation: {key}, {batch['speech_mix'].shape[1] / 8000:.3f}s", flush=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        loss, _, _ = model(**batch)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite validation loss")
    print(f"Validation loss: {loss.item():.6f}", flush=True)
    memory("Longest real validation utterance")
    print("GPU_CHECK_SUCCESS", flush=True)


if __name__ == "__main__":
    main()
