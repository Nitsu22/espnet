#!/usr/bin/env python3
"""WHAMR NPZ utilities using NpzPreprocessor.

Modes:
  - dump_mix: Save `speech_mix` wavs from NPZ inputs (legacy behavior).
  - contrastive_corr: Generate `speech_anchor/speech_pos/speech_neg1` and print correlations.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _corrcoef_1d(x: np.ndarray, y: np.ndarray, eps: float = 1.0e-12) -> float:
    """Pearson correlation (0-lag) for 1-D signals."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y)) + eps
    return float(np.sum(x * y) / denom)


def _as_time_channels(x: np.ndarray, target_len: int) -> np.ndarray:
    """Return (T, C) array, cropped to target_len."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:target_len].reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D waveform, got shape={x.shape}")
    if x.shape[0] == target_len:
        return x[:target_len]
    if x.shape[1] == target_len:
        return x.T[:target_len]
    # Fall back: assume time axis is the longer one.
    if x.shape[0] >= x.shape[1]:
        return x[:target_len]
    return x.T[:target_len]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


@dataclass
class _CorrRow:
    uid: str
    anc_pos_ch0: float
    anc_neg_ch0: float
    pos_neg_ch0: float


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
    parser.add_argument(
        "--mode",
        default="dump_mix",
        choices=("dump_mix", "contrastive_corr"),
        help="dump_mix: save speech_mix wavs. contrastive_corr: print correlations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of utterances to process in contrastive_corr mode.",
    )
    parser.add_argument(
        "--mix-type",
        default="both",
        choices=("both", "clean", "single"),
        help="Mix type passed to NpzPreprocessor.",
    )
    parser.add_argument(
        "--ref-condition",
        default="reverb",
        choices=("anechoic", "reverb"),
        help="Reference condition passed to NpzPreprocessor.",
    )
    parser.add_argument(
        "--speech-segment",
        type=int,
        default=32000,
        help="Crop length in samples for contrastive_corr (set <=0 to disable).",
    )
    parser.add_argument(
        "--contrastive-scale-min",
        type=float,
        default=0.9,
        help="Uniform scale lower bound for contrastive positives/negatives.",
    )
    parser.add_argument(
        "--contrastive-scale-max",
        type=float,
        default=1.1,
        help="Uniform scale upper bound for contrastive positives/negatives.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic RNG per uid (disables random_each_call).",
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

    uids = sorted(npz_scp.keys())

    if args.mode == "contrastive_corr":
        uids = uids[: max(0, args.num_samples)]

        preprocessor = NpzPreprocessor(
            train=True,
            mix_type=args.mix_type,
            ref_condition=args.ref_condition,
            num_spk=2,
            sample_rate=8000,
            force_single_channel=False,
            channel_reordering=False,
            rir_apply_prob=0.0,
            noise_apply_prob=0.0,
            speech_segment=None if args.speech_segment <= 0 else args.speech_segment,
            anchor_single_channel=True,
            contrastive_enable=True,
            contrastive_num_neg=1,
            contrastive_seed=1234,
            contrastive_epoch=0,
            contrastive_random_each_call=(not args.deterministic),
            contrastive_scale_min=args.contrastive_scale_min,
            contrastive_scale_max=args.contrastive_scale_max,
            contrastive_pool_rir_scp=os.path.join(args.data_dir, "rir_npz.scp"),
            contrastive_pool_room_param_scp=os.path.join(
                args.data_dir, "room_param_npz.scp"
            ),
        )

        rows: List[_CorrRow] = []
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
            out = preprocessor(uid, data)

            anchor = np.asarray(out["speech_anchor"]).reshape(-1)
            t = anchor.size
            pos = _as_time_channels(out["speech_pos"], t)
            neg = _as_time_channels(out["speech_neg1"], t)

            anc_pos0 = (
                _corrcoef_1d(anchor, pos[:, 0]) if pos.shape[1] >= 1 else float("nan")
            )
            anc_neg0 = (
                _corrcoef_1d(anchor, neg[:, 0]) if neg.shape[1] >= 1 else float("nan")
            )
            pos_neg0 = (
                _corrcoef_1d(pos[:, 0], neg[:, 0])
                if (pos.shape[1] >= 1 and neg.shape[1] >= 1)
                else float("nan")
            )

            rows.append(
                _CorrRow(uid=uid, anc_pos_ch0=anc_pos0, anc_neg_ch0=anc_neg0, pos_neg_ch0=pos_neg0)
            )

        print("uid\tanc_vs_pos(ch0)\tanc_vs_neg(ch0)\tpos_vs_neg(ch0)")
        for r in rows:
            print(
                f"{r.uid}\t{r.anc_pos_ch0:+.4f}\t{r.anc_neg_ch0:+.4f}\t{r.pos_neg_ch0:+.4f}"
            )

        anc_pos0_m, anc_pos0_s = _mean_std([r.anc_pos_ch0 for r in rows])
        anc_neg0_m, anc_neg0_s = _mean_std([r.anc_neg_ch0 for r in rows])
        pos_neg0_m, pos_neg0_s = _mean_std([r.pos_neg_ch0 for r in rows])

        print("")
        print(f"anc_vs_pos(ch0) mean±std: {anc_pos0_m:+.4f} ± {anc_pos0_s:.4f}")
        print(f"anc_vs_neg(ch0) mean±std: {anc_neg0_m:+.4f} ± {anc_neg0_s:.4f}")
        print(f"pos_vs_neg(ch0) mean±std: {pos_neg0_m:+.4f} ± {pos_neg0_s:.4f}")
        return

    # dump_mix (legacy)
    preprocessor = NpzPreprocessor(
        train=False,
        mix_type=args.mix_type,
        ref_condition=args.ref_condition,
        num_spk=2,
        sample_rate=8000,
        force_single_channel=False,
        channel_reordering=False,
        rir_apply_prob=0.0,
        noise_apply_prob=0.0,
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

        # Use the stored sample rate for saving.
        meta = np.load(npz_scp[uid], allow_pickle=True)
        sample_rate = _as_int(meta["sample_rate"])

        out = preprocessor(uid, data)
        speech_mix = out["speech_mix"]
        out_path = os.path.join(args.out_dir, uid + ".wav")
        sf.write(out_path, speech_mix, sample_rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
