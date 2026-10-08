"""Check actual pretrained-Mamba FP32 training and full-length validation on CUDA.

Run inside a one-GPU allocation before starting the long experiment. The probe
does not save trained separation weights or modify the source BiMamba model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from check_bimamba_conditioning_data import read_scp
from espnet2.tasks.enh import EnhancementTask


def digest_parameters(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("dump_nf_2spk_16k_min"))
    parser.add_argument("--config", default="conf/tuning/train_enh_tflocoformer_s_nf16k_bimamba_ctf_film_all.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one allocated CUDA device for this probe")
    options = EnhancementTask.get_parser().parse_args(
        ["--config", args.config, "--output_dir", str(args.output.parent), "--ngpu", "1"]
    )
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = EnhancementTask.build_model(options).cuda().train()
    before = digest_parameters(model.frozen_ctf)

    def read_split(split):
        folder = args.dump / "raw" / f"{split}_rir_2spk_nf_min_16k"
        maps = [read_scp(folder / name) for name in (
            "wav.scp", "speech_direct1.scp", "speech_direct2.scp")]
        return maps, read_scp(folder / "utt2num_samples")

    def batch(maps, ids, training):
        preprocess = EnhancementTask.build_preprocess_fn(options, training)
        rows = []
        for uid in ids:
            values = {}
            for name, mapping in zip(("speech_mix", "speech_ref1", "speech_ref2"), maps):
                wave, sr = sf.read(mapping[uid], dtype="float32", always_2d=True)
                if sr != 16000:
                    raise ValueError("Input is not native 16 kHz")
                values[name] = wave
            rows.append((uid, preprocess(uid, values)))
        _, data = EnhancementTask.build_collate_fn(options, training)(rows)
        return {name: value.cuda() for name, value in data.items()}

    maps, lengths = read_split("tr")
    ids = [uid for uid, length in lengths.items() if int(length) >= 64000][:4]
    if len(ids) != 4:
        raise ValueError("Cannot find four four-second training observations")
    train = batch(maps, ids, True)
    if train["speech_mix"].shape != (4, 64000):
        raise ValueError("Expected physical batch four and aligned four-second crops")
    optimizer = torch.optim.AdamW(model.parameters(), **options.optim_conf)
    torch.cuda.reset_peak_memory_stats()
    loss, _, _ = model(**train)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite conditioned separation loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("frozen_ctf."):
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError("BiMamba is not frozen")
        elif parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Missing/nonfinite separation gradient: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
    optimizer.step()
    if before != digest_parameters(model.frozen_ctf):
        raise RuntimeError("Pretrained BiMamba changed during the separation update")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite parameters after the separation update")
    report = dict(
        train_loss=float(loss), train_shape=list(train["speech_mix"].shape),
        train_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        frozen_parameters=sum(p.numel() for p in model.frozen_ctf.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_weights_unchanged=True,
        source_sha256=model.bimamba_sha256,
    )
    model.zero_grad(set_to_none=True)
    del optimizer, loss, train
    torch.cuda.empty_cache()
    model.eval()
    maps, lengths = read_split("cv")
    uid = max(lengths, key=lambda key: int(lengths[key]))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        loss, _, _ = model(**batch(maps, [uid], False))
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite full-length validation loss")
    report.update(
        validation_uid=uid, validation_samples=int(lengths[uid]),
        validation_loss=float(loss),
        validation_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        status="passed", dtype="float32", device=torch.cuda.get_device_name(0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
