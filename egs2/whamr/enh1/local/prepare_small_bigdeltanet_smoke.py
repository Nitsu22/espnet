#!/usr/bin/env python3
"""Create a tiny real-WHAMR subset for the full ESPnet 4-GPU startup check."""

from pathlib import Path


def read_scp(path):
    return dict(line.rstrip().split(maxsplit=1) for line in path.open())


def main():
    target = Path("dump/bigdeltanet_4gpu_smoke")
    for split, dataset in (
        ("train", "tr_mix_both_reverb_min_8k"),
        ("valid", "cv_mix_both_reverb_min_8k"),
    ):
        src = Path("dump/raw") / dataset
        shapes = Path("exp/enh_stats_8k") / split
        mappings = {
            "wav.scp": read_scp(src / "wav.scp"),
            "spk1.scp": read_scp(src / "spk1.scp"),
            "spk2.scp": read_scp(src / "spk2.scp"),
        }
        for name in ("speech_mix_shape", "speech_ref1_shape", "speech_ref2_shape"):
            mappings[name] = read_scp(shapes / name)
        keys = [
            key for key, length in mappings["speech_mix_shape"].items()
            if 32000 <= int(length.split(",")[0]) <= 40000
            and all(key in mapping for mapping in mappings.values())
        ][:8]
        if len(keys) != 8:
            raise RuntimeError(f"Need 8 four-to-five-second utterances for {split}")
        out = target / split
        out.mkdir(parents=True, exist_ok=True)
        for name, mapping in mappings.items():
            (out / name).write_text("".join(f"{key} {mapping[key]}\n" for key in keys))
        print(f"Prepared {len(keys)} real WHAMR utterances: {out}", flush=True)


if __name__ == "__main__":
    main()
