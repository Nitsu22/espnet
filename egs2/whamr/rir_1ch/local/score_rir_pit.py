#!/usr/bin/env python3

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from score_rir import (
    c50_db,
    drr_db,
    finite_mean,
    finite_pearson,
    finite_rmse,
    fix_length,
    load_audio,
    peak_align,
    pearson,
    read_2col,
    resample_signal,
    rt60_schroeder,
    scale_signals,
)


def read_3col(path: Path) -> Dict[str, Tuple[str, str]]:
    entries = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected 3 columns at {path}:{lineno}: {line}")
            entries[parts[0]] = (parts[1], parts[2])
    return entries


def load_rir_audio(path: Path, sample_rate: int, rir_length: int) -> np.ndarray:
    rir, fs = load_audio(path)
    if fs not in (0, sample_rate):
        rir = resample_signal(rir, fs, sample_rate)
    return fix_length(rir, rir_length)


def direct_indices_from_room_param(path: Path, sample_rate: int) -> Tuple[int, int]:
    with np.load(path, allow_pickle=True) as room:
        mic_pos = np.asarray(room["mic_pos"], dtype=np.float64)[0]
        generation_fs = int(np.asarray(room["fs"]).item())
        source_positions = (
            np.asarray(room["s1_pos"], dtype=np.float64),
            np.asarray(room["s2_pos"], dtype=np.float64),
        )

    # WHAMR uses pyroomacoustics with c=343 m/s and an 81-tap fractional-delay
    # filter. Its 40-sample group delay at generation_fs is part of the saved RIR.
    fractional_delay = int(round(40 * sample_rate / generation_fs))
    return tuple(
        int(round(np.linalg.norm(source_pos - mic_pos) / 343.0 * sample_rate))
        + fractional_delay
        for source_pos in source_positions
    )


def align_pred_to_index(pred: np.ndarray, target_index: int) -> np.ndarray:
    length = pred.shape[0]
    pred_peak = int(np.argmax(np.abs(pred))) if pred.size else 0
    shift = int(target_index) - pred_peak
    if shift > 0:
        pred = np.pad(pred, (shift, 0))
    elif shift < 0:
        pred = pred[-shift:]
    return fix_length(pred, length)


def score_pair(
    pred: np.ndarray,
    ref: np.ndarray,
    sample_rate: int,
    rir_length: int,
    align: str,
    scale_mode: str,
    direct_window_ms: float,
    direct_index: Optional[int] = None,
) -> Dict[str, float]:
    pred = fix_length(pred, rir_length)
    ref = fix_length(ref, rir_length)
    if direct_index is None:
        direct_index = int(np.argmax(np.abs(ref))) if ref.size else 0
    direct_index = max(0, min(int(direct_index), rir_length - 1))
    if align == "peak":
        pred = peak_align(pred, ref)
    elif align == "direct":
        pred = align_pred_to_index(pred, direct_index)
    pred_s, ref_s = scale_signals(pred, ref, scale_mode)
    n50 = min(rir_length, int(round(0.050 * sample_rate)))
    direct_end = min(rir_length, direct_index + n50)
    pred_direct_50ms = pred_s[direct_index:direct_end]
    ref_direct_50ms = ref_s[direct_index:direct_end]
    pred_rt60 = rt60_schroeder(pred_s, sample_rate)
    ref_rt60 = rt60_schroeder(ref_s, sample_rate)
    pred_drr = drr_db(pred_s, sample_rate, direct_window_ms)
    ref_drr = drr_db(ref_s, sample_rate, direct_window_ms)
    pred_c50 = c50_db(pred_s, sample_rate)
    ref_c50 = c50_db(ref_s, sample_rate)
    return {
        "rmse": float(np.sqrt(np.mean((pred_s - ref_s) ** 2))),
        "rmse_50ms": float(np.sqrt(np.mean((pred_s[:n50] - ref_s[:n50]) ** 2))),
        "rmse_direct_50ms": float(
            np.sqrt(np.mean((pred_direct_50ms - ref_direct_50ms) ** 2))
        ),
        "corr": pearson(pred_s, ref_s),
        "corr_direct_50ms": pearson(pred_direct_50ms, ref_direct_50ms),
        "rt60_pred": pred_rt60,
        "rt60_ref": ref_rt60,
        "rt60_err": pred_rt60 - ref_rt60,
        "drr_pred": pred_drr,
        "drr_ref": ref_drr,
        "drr_err": pred_drr - ref_drr,
        "c50_pred": pred_c50,
        "c50_ref": ref_c50,
        "c50_err": pred_c50 - ref_c50,
    }


def aggregate_pair_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics = list(metrics)
    return {
        "rmse": finite_mean(m["rmse"] for m in metrics),
        "rmse_50ms": finite_mean(m["rmse_50ms"] for m in metrics),
        "rmse_direct_50ms": finite_mean(m["rmse_direct_50ms"] for m in metrics),
        "corr": finite_mean(m["corr"] for m in metrics),
        "corr_direct_50ms": finite_mean(m["corr_direct_50ms"] for m in metrics),
        "rt60_mae": finite_mean(abs(m["rt60_err"]) for m in metrics),
        "rt60_rmse": finite_rmse(m["rt60_err"] for m in metrics),
        "drr_mae": finite_mean(abs(m["drr_err"]) for m in metrics),
        "drr_rmse": finite_rmse(m["drr_err"] for m in metrics),
        "c50_mae": finite_mean(abs(m["c50_err"]) for m in metrics),
        "c50_rmse": finite_rmse(m["c50_err"] for m in metrics),
    }


def pit_score(row: Dict[str, float], metric: str) -> float:
    value = row[metric]
    if metric.startswith("corr"):
        return -value if math.isfinite(value) else float("inf")
    return value if math.isfinite(value) else float("inf")


def build_pred_map(args) -> Dict[str, Tuple[str, str]]:
    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
        if (pred_dir / "rir.scp").is_file():
            return read_3col(pred_dir / "rir.scp")
        rir1_scp = pred_dir / "rir1" / "wav.scp"
        rir2_scp = pred_dir / "rir2" / "wav.scp"
        if not rir1_scp.is_file() or not rir2_scp.is_file():
            raise FileNotFoundError(
                f"Expected {pred_dir}/rir.scp or rir1/wav.scp and rir2/wav.scp"
            )
        pred1 = read_2col(rir1_scp)
        pred2 = read_2col(rir2_scp)
    elif args.pred_rir_scp:
        return read_3col(Path(args.pred_rir_scp))
    else:
        if not args.pred_rir1_scp or not args.pred_rir2_scp:
            raise ValueError(
                "Use --pred_dir, --pred_rir_scp, or both --pred_rir1_scp and "
                "--pred_rir2_scp"
            )
        pred1 = read_2col(Path(args.pred_rir1_scp))
        pred2 = read_2col(Path(args.pred_rir2_scp))

    missing = sorted(set(pred1) ^ set(pred2))
    if missing:
        raise KeyError(f"Prediction key mismatch; first={missing[0]}")
    return {uid: (pred1[uid], pred2[uid]) for uid in pred1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_rir_scp", required=True)
    pred_group = parser.add_mutually_exclusive_group()
    pred_group.add_argument("--pred_dir")
    pred_group.add_argument("--pred_rir_scp")
    parser.add_argument("--pred_rir1_scp")
    parser.add_argument("--pred_rir2_scp")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--rir_length", type=int, default=8192)
    parser.add_argument("--align", choices=["none", "peak", "direct"], default="peak")
    parser.add_argument("--room_param_scp")
    parser.add_argument(
        "--scale_mode",
        choices=["none", "peak", "ref_peak", "optimal"],
        default="peak",
    )
    parser.add_argument("--direct_window_ms", type=float, default=2.5)
    parser.add_argument(
        "--pit_metric",
        choices=[
            "rmse",
            "rmse_50ms",
            "rmse_direct_50ms",
            "corr",
            "corr_direct_50ms",
        ],
        default="rmse_50ms",
    )
    args = parser.parse_args()

    ref_map = read_3col(Path(args.ref_rir_scp))
    pred_map = build_pred_map(args)
    room_param_map = (
        read_2col(Path(args.room_param_scp)) if args.room_param_scp else None
    )
    if args.align == "direct" and room_param_map is None:
        raise ValueError("--room_param_scp is required when --align=direct")
    missing = sorted(set(ref_map) - set(pred_map))
    if missing:
        raise KeyError(
            f"{len(missing)} reference utterances missing in predictions; first={missing[0]}"
        )
    if room_param_map is not None:
        missing_room = sorted(set(ref_map) - set(room_param_map))
        if missing_room:
            raise KeyError(
                f"{len(missing_room)} room parameters missing; first={missing_room[0]}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_utt_rows = []
    per_source_rows = []
    permutations = tuple(itertools.permutations(range(2)))
    for uid, ref_paths in sorted(ref_map.items()):
        pred_paths = pred_map[uid]
        direct_indices = (
            direct_indices_from_room_param(Path(room_param_map[uid]), args.sample_rate)
            if room_param_map is not None
            else (None, None)
        )
        refs = [
            load_rir_audio(Path(ref_path), args.sample_rate, args.rir_length)
            for ref_path in ref_paths
        ]
        preds = [
            load_rir_audio(Path(pred_path), args.sample_rate, args.rir_length)
            for pred_path in pred_paths
        ]

        candidates = []
        for perm_index, perm in enumerate(permutations):
            pair_rows = []
            for pred_index, ref_index in enumerate(perm):
                metrics = score_pair(
                    preds[pred_index],
                    refs[ref_index],
                    args.sample_rate,
                    args.rir_length,
                    args.align,
                    args.scale_mode,
                    args.direct_window_ms,
                    direct_indices[ref_index],
                )
                pair_rows.append((pred_index, ref_index, metrics))
            agg = aggregate_pair_metrics(metrics for _, _, metrics in pair_rows)
            candidates.append((pit_score(agg, args.pit_metric), perm_index, perm, pair_rows, agg))

        _, perm_index, perm, pair_rows, agg = min(candidates, key=lambda item: item[0])
        per_utt_rows.append(
            {
                "utt_id": uid,
                "perm_index": perm_index,
                "perm": ",".join(str(i) for i in perm),
                **agg,
            }
        )
        for pred_index, ref_index, metrics in pair_rows:
            per_source_rows.append(
                {
                    "utt_id": uid,
                    "perm_index": perm_index,
                    "pred_index": pred_index,
                    "ref_index": ref_index,
                    **metrics,
                }
            )

    per_utt_path = out_dir / "per_utt.csv"
    with per_utt_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(per_utt_rows[0].keys()) if per_utt_rows else ["utt_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_utt_rows)

    per_source_path = out_dir / "per_source.csv"
    with per_source_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = (
            list(per_source_rows[0].keys()) if per_source_rows else ["utt_id"]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_source_rows)

    summary = {
        "num_utts": len(per_utt_rows),
        "num_source_pairs": len(per_source_rows),
        "sample_rate": args.sample_rate,
        "rir_length": args.rir_length,
        "align": args.align,
        "direct_index_source": (
            "room_geometry" if room_param_map is not None else "reference_peak"
        ),
        "scale_mode": args.scale_mode,
        "pit_metric": args.pit_metric,
        "perm0_ratio": finite_mean(
            1.0 if row["perm_index"] == 0 else 0.0 for row in per_utt_rows
        ),
        "rmse_mean": finite_mean(r["rmse"] for r in per_source_rows),
        "rmse_50ms_mean": finite_mean(r["rmse_50ms"] for r in per_source_rows),
        "rmse_direct_50ms_mean": finite_mean(
            r["rmse_direct_50ms"] for r in per_source_rows
        ),
        "corr_mean": finite_mean(r["corr"] for r in per_source_rows),
        "corr_direct_50ms_mean": finite_mean(
            r["corr_direct_50ms"] for r in per_source_rows
        ),
        "rt60_mae": finite_mean(abs(r["rt60_err"]) for r in per_source_rows),
        "rt60_rmse": finite_rmse(r["rt60_err"] for r in per_source_rows),
        "rt60_pearson": finite_pearson(
            (r["rt60_pred"] for r in per_source_rows),
            (r["rt60_ref"] for r in per_source_rows),
        ),
        "drr_mae": finite_mean(abs(r["drr_err"]) for r in per_source_rows),
        "drr_rmse": finite_rmse(r["drr_err"] for r in per_source_rows),
        "drr_pearson": finite_pearson(
            (r["drr_pred"] for r in per_source_rows),
            (r["drr_ref"] for r in per_source_rows),
        ),
        "c50_mae": finite_mean(abs(r["c50_err"]) for r in per_source_rows),
        "c50_rmse": finite_rmse(r["c50_err"] for r in per_source_rows),
        "c50_pearson": finite_pearson(
            (r["c50_pred"] for r in per_source_rows),
            (r["c50_ref"] for r in per_source_rows),
        ),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
