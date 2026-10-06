#!/usr/bin/env python3
"""Full-utterance DAMSEP inference; write clean/reverb WAVs and complex CTFs."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.damsep import DAMSEPTask
from espnet2.utils.types import str2bool


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--wav_scp", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--normalize_output_wav", type=str2bool, default=True)
    parser.add_argument("--key_file", type=Path)
    return parser


def inference(args):
    model, _ = DAMSEPTask.build_model_from_file(
        args.train_config, args.model_file, args.device
    )
    model.eval()
    mixtures = {}
    for line in args.wav_scp.read_text().splitlines():
        uid, path = line.split(maxsplit=1)
        if uid in mixtures or Path(uid).name != uid or uid in (".", ".."):
            raise ValueError(f"Duplicate or unsafe utterance ID: {uid}")
        mixtures[uid] = path
    keys = list(mixtures)
    if args.key_file is not None:
        keys = [line.split()[0] for line in args.key_file.read_text().splitlines()]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError("Inference keys must be nonempty and unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lists = {
        (branch, speaker): [] for branch in ("clean", "reverb") for speaker in (1, 2)
    }
    ctf_list = []
    with torch.no_grad():
        for i, uid in enumerate(keys):
            audio, sr = sf.read(mixtures[uid], dtype="float32", always_2d=True)
            if sr != 8000 or len(audio) <= 256 or not np.isfinite(audio).all():
                raise ValueError(
                    f"{uid}: expected finite 8 kHz audio longer than 256 samples"
                )
            waveform = torch.from_numpy(audio[:, 0].copy()).unsqueeze(0).to(args.device)
            output = model.separate(waveform)
            for branch, field in (("clean", "x_derev"), ("reverb", "x_sep")):
                values = output[field][0].cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{uid}: nonfinite {branch} estimates")
                for speaker in (1, 2):
                    signal = values[speaker - 1]
                    if args.normalize_output_wav:
                        # Match enh4_clean's 0.9 peak normalization before PCM16.
                        signal = signal * 0.9 / max(float(np.max(np.abs(signal))), 1e-8)
                    dest = args.output_dir / branch / f"wav_spk{speaker}" / f"{uid}.wav"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        dest,
                        signal,
                        sr,
                        subtype="PCM_16" if args.normalize_output_wav else "FLOAT",
                    )
                    lists[branch, speaker].append(f"{uid} {dest}\n")
            ctf_ri = output["rir"].reshape(2, 2, 257, 60).cpu().numpy()
            ctf = ctf_ri[:, 0] + 1j * ctf_ri[:, 1]
            if not np.isfinite(ctf).all():
                raise ValueError(f"{uid}: nonfinite CTF estimates")
            dest = args.output_dir / "ctf" / f"{uid}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, ctf=ctf)
            ctf_list.append(f"{uid} {dest}\n")
            if (i + 1) % 100 == 0:
                print(f"Inferred {i + 1}/{len(keys)} utterances", flush=True)
    for (branch, speaker), lines in lists.items():
        (args.output_dir / branch / f"spk{speaker}.scp").write_text("".join(lines))
    (args.output_dir / "ctf.scp").write_text("".join(ctf_list))
    (args.output_dir / "inference.json").write_text(
        json.dumps(
            dict(
                train_config=str(args.train_config),
                model_file=str(args.model_file),
                sample_rate=8000,
                channel=0,
                count=len(keys),
                normalize_output_wav=args.normalize_output_wav,
                ctf_shape=[2, 257, 60],
                ctf_source_order="matches output WAVs; no distance ordering",
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    inference(get_parser().parse_args())


if __name__ == "__main__":
    main()
