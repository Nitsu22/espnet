#!/usr/bin/env python3
"""Create anchor/positive/negative samples for NpzSwapRirPreprocessor."""

import argparse
import os
from typing import Dict, List

import numpy as np
import soundfile as sf

from espnet2.train.preprocessor_npz import NpzSwapRirPreprocessor


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


def _as_int(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _to_time_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D waveform, got shape={x.shape}")
    if x.shape[0] >= x.shape[1]:
        return x
    return x.T


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    audio = _to_time_channels(audio)
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def _build_sample(
    uid: str,
    npz_scp: Dict[str, str],
    s1_base_scp: Dict[str, str],
    s2_base_scp: Dict[str, str],
    s1_temp_scp: Dict[str, str],
    s2_temp_scp: Dict[str, str],
    noise_base_scp: Dict[str, str],
    rir_scp: Dict[str, str],
    room_param_scp: Dict[str, str],
) -> Dict[str, np.ndarray]:
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

    return {
        "npz_path": npz_scp[uid],
        "s1_base": np.load(s1_base_scp[uid], allow_pickle=False),
        "s2_base": np.load(s2_base_scp[uid], allow_pickle=False),
        "s1_temp": np.load(s1_temp_scp[uid], allow_pickle=False),
        "s2_temp": np.load(s2_temp_scp[uid], allow_pickle=False),
        "noise_base": np.load(noise_base_scp[uid], allow_pickle=False),
        "rir_path": rir_scp[uid],
        "room_param_path": room_param_scp[uid],
    }


def _select_uids(uids: List[str], num_samples: int, shuffle: bool, seed: int) -> List[str]:
    uids = sorted(uids)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(uids)
    if num_samples > 0:
        uids = uids[:num_samples]
    return uids


def main() -> None:
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
            "egs2/whamr/se_npz/tmp_swap_rir_samples"
        ),
        help="Output directory for generated wavs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of utterances to process (<=0 means all).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle uids before sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for shuffling and contrastive RNG (deterministic mode).",
    )
    parser.add_argument(
        "--mix-type",
        default="both",
        choices=("both", "clean", "single"),
        help="Mix type passed to NpzSwapRirPreprocessor.",
    )
    parser.add_argument(
        "--ref-condition",
        default="reverb",
        choices=("anechoic", "reverb"),
        help="Reference condition passed to NpzSwapRirPreprocessor.",
    )
    parser.add_argument(
        "--speech-segment",
        type=int,
        default=32000,
        help="Crop length in samples (set <=0 to disable).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
        help="Sample rate expected by the preprocessor.",
    )
    parser.add_argument(
        "--anchor-single-channel",
        action="store_true",
        help="If set, collapse anchor to single channel.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic RNG per uid (disables random_each_call).",
    )
    parser.add_argument(
        "--pool-npz-scp",
        default=None,
        help="NPZ pool scp for sampling data2 (default: data-dir/npz.scp).",
    )
    parser.add_argument(
        "--pool-rir-scp",
        default=None,
        help="RIR pool scp (default: data-dir/rir_npz.scp).",
    )
    parser.add_argument(
        "--pool-room-param-scp",
        default=None,
        help="Room-param pool scp (default: data-dir/room_param_npz.scp).",
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

    pool_npz_scp = args.pool_npz_scp or os.path.join(args.data_dir, "npz.scp")
    pool_rir_scp = args.pool_rir_scp or os.path.join(args.data_dir, "rir_npz.scp")
    pool_room_param_scp = args.pool_room_param_scp or os.path.join(
        args.data_dir, "room_param_npz.scp"
    )
    if not os.path.exists(pool_npz_scp):
        raise FileNotFoundError(f"Missing scp: {pool_npz_scp}")
    if not os.path.exists(pool_rir_scp):
        raise FileNotFoundError(f"Missing scp: {pool_rir_scp}")
    if not os.path.exists(pool_room_param_scp):
        raise FileNotFoundError(f"Missing scp: {pool_room_param_scp}")

    uids = _select_uids(
        list(npz_scp.keys()), args.num_samples, args.shuffle, args.seed
    )

    preprocessor = NpzSwapRirPreprocessor(
        train=True,
        mix_type=args.mix_type,
        ref_condition=args.ref_condition,
        num_spk=2,
        sample_rate=args.sample_rate,
        force_single_channel=False,
        channel_reordering=False,
        rir_apply_prob=0.0,
        noise_apply_prob=0.0,
        speech_segment=None if args.speech_segment <= 0 else args.speech_segment,
        anchor_single_channel=args.anchor_single_channel,
        contrastive_enable=True,
        contrastive_num_neg=1,
        contrastive_seed=args.seed,
        contrastive_epoch=0,
        contrastive_random_each_call=(not args.deterministic),
        contrastive_scale_min=1.0,
        contrastive_scale_max=1.0,
        contrastive_pool_rir_scp=pool_rir_scp,
        contrastive_pool_room_param_scp=pool_room_param_scp,
        contrastive_pool_npz_scp=pool_npz_scp,
    )

    for uid in uids:
        data = _build_sample(
            uid,
            npz_scp,
            s1_base_scp,
            s2_base_scp,
            s1_temp_scp,
            s2_temp_scp,
            noise_base_scp,
            rir_scp,
            room_param_scp,
        )

        meta = np.load(npz_scp[uid], allow_pickle=True)
        sample_rate = _as_int(meta["sample_rate"])

        out = preprocessor(uid, data)
        out_dir = os.path.join(args.out_dir, uid)
        os.makedirs(out_dir, exist_ok=True)

        _write_wav(os.path.join(out_dir, "anchor.wav"), out["speech_anchor"], sample_rate)
        _write_wav(os.path.join(out_dir, "positive.wav"), out["speech_pos"], sample_rate)
        _write_wav(os.path.join(out_dir, "negative.wav"), out["speech_neg1"], sample_rate)

        print(f"Wrote samples for {uid} -> {out_dir}")


if __name__ == "__main__":
    main()
