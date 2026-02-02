#!/usr/bin/env python3
"""Generate WHAMR mixtures from NPZ inputs using NpzPreprocessor."""

import argparse
import os
from typing import Dict

import numpy as np
import soundfile as sf

from espnet2.train.preprocessor_npz import NpzPreprocessor


def _read_scp(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, value = line.split(maxsplit=1)
            mapping[utt_id] = value
    return mapping


def _require_scp(data_dir: str, name: str) -> Dict[str, str]:
    path = os.path.join(data_dir, f"{name}.scp")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scp: {path}")
    return _read_scp(path)


def _as_int(value):
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=(
            "/net/midgar/work/nitsu/learning/tf-locoformer/espnet/"
            "egs2/whamr/se_npz/data/tt_mix_clean_reverb_min_8k"
        ),
        help="Kaldi-style data dir with *_npz scp files.",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "/net/midgar/work/nitsu/learning/tf-locoformer/espnet/"
            "egs2/whamr/se_npz/tmp_mixture"
        ),
        help="Output directory for generated wavs.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz_scp = _require_scp(args.data_dir, "npz")
    s1_base_scp = _require_scp(args.data_dir, "spk1_base_npz")
    s2_base_scp = _require_scp(args.data_dir, "spk2_base_npz")
    s1_temp_scp = _require_scp(args.data_dir, "spk1_temp_npz")
    s2_temp_scp = _require_scp(args.data_dir, "spk2_temp_npz")
    noise_base_scp = _require_scp(args.data_dir, "noise_base_npz")
    rir_scp = _require_scp(args.data_dir, "rir_npz")
    room_param_scp = _require_scp(args.data_dir, "room_param_npz")

    preprocessor = NpzPreprocessor(
        train=False,
        mix_type="clean",
        ref_condition="reverb",
        num_spk=2,
        sample_rate=8000,
        force_single_channel=False,
        channel_reordering=False,
        rir_apply_prob=0.0,
        noise_apply_prob=0.0,
    )

    for uid in sorted(npz_scp.keys()):
        missing = []
        for name, scp in (
            ("spk1_base_npz", s1_base_scp),
            ("spk2_base_npz", s2_base_scp),
            ("spk1_temp_npz", s1_temp_scp),
            ("spk2_temp_npz", s2_temp_scp),
            ("noise_base_npz", noise_base_scp),
            ("rir_npz", rir_scp),
            ("room_param_npz", room_param_scp),
        ):
            if uid not in scp:
                missing.append(name)
        if missing:
            raise KeyError(f"{uid} missing scp entries: {', '.join(missing)}")

        data = {
            "npz_path": npz_scp[uid],
            "s1_base": np.load(s1_base_scp[uid], allow_pickle=False),
            "s2_base": np.load(s2_base_scp[uid], allow_pickle=False),
            "s1_temp": np.load(s1_temp_scp[uid], allow_pickle=False),
            "s2_temp": np.load(s2_temp_scp[uid], allow_pickle=False),
            "noise_base": np.load(noise_base_scp[uid], allow_pickle=False),
            "rir_path": rir_scp[uid],
            "room_param_path": room_param_scp[uid],
        }

        # Use the stored sample rate for saving.
        meta = np.load(npz_scp[uid], allow_pickle=True)
        sample_rate = _as_int(meta["sample_rate"])

        out = preprocessor(uid, data)
        speech_mix = out["speech_mix"]
        out_path = os.path.join(args.out_dir, uid + ".wav")
        sf.write(out_path, speech_mix, sample_rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
