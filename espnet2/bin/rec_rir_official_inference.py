"""Run released Rec-RIR weights using ESPnet's port, optionally checking parity."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import toml
import torch

from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.model import BiSpatialNet
from espnet2.rir.rec_rir.pim import RecRIRPIM


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scp(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in result or Path(uid).name != uid:
            raise ValueError(f"Invalid or duplicate utterance ID: {uid}")
        result[uid] = value
    return result


def normalize(x):
    if not torch.isfinite(x).all() or x.abs().max() == 0:
        raise ValueError("Nonfinite or silent RIR")
    return x / x.abs().max()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wav-scp", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-repo", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    config = toml.load(args.config)
    transform = RecRIRTransforms(**config["acoustic"]["args"])
    sr = transform.sr
    model = BiSpatialNet(**config["model"]["args"]).to(args.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {
        k.removeprefix("module."): v
        for k, v in checkpoint["model"].items()
        if not any(word in k for word in ("ops", "params"))
    }
    model.load_state_dict(state, strict=True)
    pim = RecRIRPIM(sr=sr)
    entries = list(read_scp(args.wav_scp).items())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        entries = entries[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "config": config,
        "checkpoint_sha256": digest(args.checkpoint),
        "config_sha256": digest(args.config),
        "input_scp_sha256": digest(args.wav_scp),
        "torch": torch.__version__,
        "device": args.device,
        "samples": len(entries),
        "rir_samples": sr * 2,
        "output_normalization": "absolute peak, matching official inference.py",
        "script_sha256": digest(__file__),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.parity_repo:
        sys.path.insert(0, str(args.parity_repo.resolve()))
        from acoustics.feature import transforms
        from method.pim import PIM
        from model.RecRIR import BiSpatialNet as OfficialModel

        official = OfficialModel(**config["model"]["args"]).to(args.device).eval()
        official.load_state_dict(state, strict=True)
        official_tf = transforms(**config["acoustic"]["args"])
        official_pim = PIM(**config["EM_algo"]["args"])
    parity = []
    with (args.output_dir / "wav.scp").open("w") as scp:
        for index, (uid, path) in enumerate(entries):
            wav, fs = sf.read(path, dtype="float32")
            if fs != sr or wav.ndim != 1 or not np.isfinite(wav).all():
                raise ValueError(f"Expected finite mono {sr} Hz audio: {path}")
            x = torch.from_numpy(wav).to(args.device)
            spec = transform.stft(transform.norm_amplitude(x), "complex")
            _, features, _ = model(transform.preprocess(spec[None, None]))
            ctf = transform.postprocess(features).squeeze(0).squeeze(0).flip(-1)
            rir = normalize(pim.ctf_to_rir(ctf, transform, x.device, sr * 2))
            if args.parity_repo:
                official_ctf = official_pim.init_seg(official_tf, official, x)
                official_rir = normalize(official_pim.process(x, official, official_tf, x.device))
                torch.testing.assert_close(ctf, official_ctf, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(rir.cpu(), official_rir.cpu(), atol=1e-5, rtol=1e-4)
                parity.append({
                    "uid": uid,
                    "ctf_max_abs_error": float((ctf - official_ctf).abs().max()),
                    "rir_max_abs_error": float((rir.cpu() - official_rir.cpu()).abs().max()),
                })
            output = args.output_dir.resolve() / f"{uid}.wav"
            sf.write(output, rir.cpu().numpy(), sr, subtype="FLOAT")
            scp.write(f"{uid} {output}\n")
            scp.flush()
            if index % 50 == 0:
                print(f"{index + 1}/{len(entries)} {uid}", flush=True)
    if args.parity_repo:
        (args.output_dir / "parity.json").write_text(json.dumps({
            "passed": True, "atol": 1e-5, "rtol": 1e-4, "utterances": parity,
            "official_source_sha256": {
                str(p): digest(args.parity_repo / p)
                for p in ("model/RecRIR.py", "method/pim.py", "acoustics/feature.py")
            },
        }, indent=2))
    (args.output_dir / "complete.json").write_text(json.dumps({"utterances": len(entries)}))


if __name__ == "__main__":
    main()
