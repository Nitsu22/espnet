#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

from espnet2.fileio.sound_scp import SoundScpReader, SoundScpWriter
from espnet2.tasks.rir import RIRTask


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-source Rec-RIR PIT inference")
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--rir_length", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    return parser


def main(cmd=None):
    args = get_parser().parse_args(cmd)
    device = args.device
    dtype = getattr(torch, args.dtype)

    model, _ = RIRTask.build_model_from_file(
        config_file=args.train_config,
        model_file=args.model_file,
        device=device,
    )
    if not hasattr(model, "estimate_rir"):
        raise RuntimeError("The loaded model does not support estimate_rir()")
    model = model.to(dtype=dtype, device=device).eval()

    output_dir = Path(args.output_dir)
    reader = SoundScpReader(args.wav_scp, dtype=np.float32, always_2d=False)
    uids = list(reader.keys())
    with SoundScpWriter(output_dir / "rir1", output_dir / "rir1" / "wav.scp") as writer1, \
            SoundScpWriter(output_dir / "rir2", output_dir / "rir2" / "wav.scp") as writer2:
        for uid in uids:
            sample_rate, wav = reader[uid]
            if sample_rate != args.sample_rate:
                raise ValueError(
                    f"{uid}: expected {args.sample_rate} Hz, got {sample_rate} Hz"
                )
            if wav.ndim == 2:
                wav = wav[:, 0]
            speech = torch.as_tensor(wav, dtype=dtype, device=device)
            with torch.no_grad():
                rir = model.estimate_rir(speech, rir_length=args.rir_length)
            rir = rir.detach().cpu().float().numpy()
            if rir.ndim == 3:
                rir = rir[0]
            if rir.ndim != 2 or rir.shape[0] != 2:
                raise RuntimeError(f"{uid}: expected estimated RIR shape [2, T], got {rir.shape}")
            writer1[uid] = (args.sample_rate, rir[0])
            writer2[uid] = (args.sample_rate, rir[1])

        with (output_dir / "rir.scp").open("w", encoding="utf-8") as f:
            for uid in uids:
                f.write(f"{uid} {writer1.get_path(uid)} {writer2.get_path(uid)}\n")


if __name__ == "__main__":
    main()
