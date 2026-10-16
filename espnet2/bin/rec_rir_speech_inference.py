#!/usr/bin/env python3
"""Write Rec-RIR clean/reverb audio using DAMSEP's WAV normalization."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.rir.rec_rir.speech_inference import estimate_speech
from espnet2.tasks.rir import RIRTask


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--wav_scp", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    model, config = RIRTask.build_model_from_file(
        args.train_config, args.model_file, args.device
    )
    if config.sample_rate != 8000 or config.rir_model_type != "rec_rir_pit":
        raise ValueError("Expected native 8 kHz two-speaker Rec-RIR")
    model = model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [line.split(maxsplit=1) for line in args.wav_scp.read_text().splitlines()]
    if args.limit < 0 or len({uid for uid, _ in rows}) != len(rows) or not rows:
        raise ValueError("Empty/duplicate mixture IDs or negative limit")
    rows = rows[: args.limit or None]
    indexes = {(branch, spk): [] for branch in ("clean", "reverb") for spk in (1, 2)}
    for number, (uid, path) in enumerate(rows, 1):
        if Path(uid).name != uid or uid in (".", ".."):
            raise ValueError(f"Unsafe ID: {uid}")
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != 8000 or not np.isfinite(audio).all():
            raise ValueError(f"Expected finite 8 kHz mixture: {uid}")
        estimates = estimate_speech(
            model, torch.from_numpy(audio[:, 0].copy()).to(args.device)
        )
        for branch, values in estimates.items():
            for spk, signal in enumerate(values.cpu().numpy(), 1):
                # Match damsep_inference exactly: normalize each source/branch
                # to 0.9 peak before PCM16, with no estimated delay correction.
                signal = signal * 0.9 / max(float(np.max(np.abs(signal))), 1e-8)
                dest = args.output_dir / branch / f"wav_spk{spk}" / f"{uid}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dest, signal, sr, subtype="PCM_16")
                indexes[branch, spk].append(f"{uid} {dest}\n")
        if number % 100 == 0:
            print(f"Inferred speech {number}/{len(rows)} utterances", flush=True)
    for (branch, spk), lines in indexes.items():
        (args.output_dir / branch / f"spk{spk}.scp").write_text("".join(lines))
    (args.output_dir / "inference_complete.json").write_text(
        json.dumps(
            dict(
                count=len(rows),
                sample_rate=8000,
                channel=0,
                normalize_output_wav=True,
                output_subtype="PCM_16",
                exact_input_length=True,
                train_config=args.train_config,
                model_file=args.model_file,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
