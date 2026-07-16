#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.tasks.rir import RIRTask


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-source Rec-RIR PIT CTF inference")
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
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
    if not hasattr(model, "estimate_ctf"):
        raise RuntimeError("The loaded model does not support estimate_ctf()")
    model = model.to(dtype=dtype, device=device).eval()

    output_dir = Path(args.output_dir)
    reader = SoundScpReader(args.wav_scp, dtype=np.float32, always_2d=False)
    uids = list(reader.keys())
    with NpyScpWriter(output_dir / "rir_ctf", output_dir / "rir_ctf.scp") as writer:
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
                ctf = model.estimate_ctf(speech)
            ctf = ctf.detach().cpu().to(dtype=torch.complex64)
            if ctf.ndim == 4:
                ctf = ctf[0]
            if ctf.ndim != 3 or ctf.shape[0] != 2:
                raise RuntimeError(
                    f"{uid}: expected estimated CTF shape [2, F, taps], "
                    f"got {tuple(ctf.shape)}"
                )
            ctf = ctf.permute(2, 0, 1).contiguous()  # [taps, spk, F]
            writer[uid] = torch.view_as_real(ctf).numpy().astype(np.float32)


if __name__ == "__main__":
    main()
