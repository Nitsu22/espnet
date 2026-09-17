#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def resample_signal(x: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    from scipy.signal import resample_poly

    gcd = math.gcd(orig_fs, target_fs)
    return resample_poly(x, target_fs // gcd, orig_fs // gcd)


def read_2col(path: Path) -> Dict[str, str]:
    entries = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Expected 2 columns at {path}:{lineno}: {line}")
            entries[parts[0]] = parts[1]
    return entries


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        wav = np.load(path)
        return np.asarray(wav, dtype=np.float64).reshape(-1), 0
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            if "rir" in z:
                wav = z["rir"]
            elif "rir_pred" in z:
                wav = z["rir_pred"]
            else:
                keys = ", ".join(z.files)
                raise KeyError(f"{path} has no rir/rir_pred key: {keys}")
            fs = int(np.asarray(z["fs"]).item()) if "fs" in z else 0
        return np.asarray(wav, dtype=np.float64).reshape(-1), fs
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        if suffix != ".wav":
            raise
        import struct

        with path.open("rb") as f:
            header = f.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise ValueError(f"Unsupported WAV header: {path}")
            fmt = None
            data = None
            while True:
                chunk_header = f.read(8)
                if not chunk_header:
                    break
                if len(chunk_header) != 8:
                    raise ValueError(f"Truncated WAV chunk header: {path}")
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                chunk = f.read(chunk_size)
                if len(chunk) != chunk_size:
                    raise ValueError(f"Truncated WAV chunk: {path}")
                if chunk_size % 2:
                    f.read(1)
                if chunk_id == b"fmt ":
                    fmt = chunk
                elif chunk_id == b"data":
                    data = chunk
            if fmt is None or data is None:
                raise ValueError(f"Missing fmt/data chunk in WAV: {path}")
            if len(fmt) < 16:
                raise ValueError(f"Invalid fmt chunk in WAV: {path}")
            audio_format, channels, fs, _, _, bits_per_sample = struct.unpack(
                "<HHIIHH", fmt[:16]
            )
        if audio_format == 1 and bits_per_sample == 8:
            wav = (
                np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0
            ) / 128.0
        elif audio_format == 1 and bits_per_sample == 16:
            wav = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
        elif audio_format == 1 and bits_per_sample == 24:
            raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            vals = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            vals = np.where(vals & 0x800000, vals | ~0xFFFFFF, vals)
            wav = vals.astype(np.float64) / 8388608.0
        elif audio_format == 1 and bits_per_sample == 32:
            wav = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
        elif audio_format == 3 and bits_per_sample == 32:
            wav = np.frombuffer(data, dtype="<f4").astype(np.float64)
        elif audio_format == 3 and bits_per_sample == 64:
            wav = np.frombuffer(data, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(
                f"Unsupported WAV format: format={audio_format}, bits={bits_per_sample}"
            )
        if channels > 1:
            wav = wav.reshape(-1, channels)
    else:
        wav, fs = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim == 2:
        wav = wav[:, 0]
    return wav.reshape(-1), int(fs)


def load_ref_rir(
    path: Path,
    target: str,
    source_index: int,
    mic_index: int,
    sample_rate: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        key = f"rir_{target}"
        if key not in z:
            raise KeyError(f"{key} is not found in {path}")
        rir = np.asarray(z[key], dtype=np.float64)
        fs = int(np.asarray(z["fs"]).item()) if "fs" in z else sample_rate
    if rir.ndim != 3:
        raise ValueError(f"Expected [source, mic, time] RIR in {path}: {rir.shape}")
    rir = rir[source_index, mic_index]
    if fs != sample_rate:
        rir = resample_signal(rir, fs, sample_rate)
    return np.asarray(rir, dtype=np.float64).reshape(-1)


def load_ref_rir_audio(path: Path, sample_rate: int) -> np.ndarray:
    rir, fs = load_audio(path)
    if fs not in (0, sample_rate):
        rir = resample_signal(rir, fs, sample_rate)
    return np.asarray(rir, dtype=np.float64).reshape(-1)


def fix_length(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] >= length:
        return x[:length]
    return np.pad(x, (0, length - x.shape[0]))


def peak_align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred_peak = int(np.argmax(np.abs(pred))) if pred.size else 0
    ref_peak = int(np.argmax(np.abs(ref))) if ref.size else 0
    shift = ref_peak - pred_peak
    if shift > 0:
        pred = np.pad(pred, (shift, 0))
    elif shift < 0:
        pred = pred[-shift:]
    return fix_length(pred, ref.shape[0])


def scale_signals(pred: np.ndarray, ref: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1.0e-12
    pred = pred.copy()
    ref = ref.copy()
    if mode == "none":
        return pred, ref
    if mode == "peak":
        pred = pred / max(np.max(np.abs(pred)), eps)
        ref = ref / max(np.max(np.abs(ref)), eps)
    elif mode == "ref_peak":
        scale = max(np.max(np.abs(ref)), eps)
        pred = pred / scale
        ref = ref / scale
    elif mode == "optimal":
        alpha = float(np.dot(pred, ref) / max(np.dot(pred, pred), eps))
        pred = pred * alpha
    else:
        raise ValueError(f"Unsupported scale mode: {mode}")
    return pred, ref


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1) - float(np.mean(x))
    y = y.reshape(-1) - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def drr_db(rir: np.ndarray, fs: int, direct_window_ms: float) -> float:
    peak = int(np.argmax(np.abs(rir))) if rir.size else 0
    half = int(round(fs * direct_window_ms / 1000.0))
    b = max(0, peak - half)
    e = min(rir.shape[0], peak + half + 1)
    direct = float(np.sum(rir[b:e] ** 2))
    late = float(np.sum(rir[:b] ** 2) + np.sum(rir[e:] ** 2))
    if direct <= 0.0 or late <= 0.0:
        return float("nan")
    return 10.0 * math.log10(direct / late)


def c50_db(rir: np.ndarray, fs: int) -> float:
    peak = int(np.argmax(np.abs(rir))) if rir.size else 0
    boundary = min(rir.shape[0], peak + int(round(0.050 * fs)))
    early = float(np.sum(rir[peak:boundary] ** 2))
    late = float(np.sum(rir[boundary:] ** 2))
    if early <= 0.0 or late <= 0.0:
        return float("nan")
    return 10.0 * math.log10(early / late)


def rt60_schroeder(rir: np.ndarray, fs: int) -> float:
    energy = np.asarray(rir, dtype=np.float64) ** 2
    if not np.any(energy > 0):
        return float("nan")
    sch = np.cumsum(energy[::-1])[::-1]
    sch = sch / max(float(sch[0]), 1.0e-12)
    db = 10.0 * np.log10(np.maximum(sch, 1.0e-12))
    t = np.arange(db.shape[0], dtype=np.float64) / float(fs)
    for lo, hi, factor in [(-35.0, -5.0, 2.0), (-25.0, -5.0, 3.0)]:
        mask = (db <= hi) & (db >= lo)
        if int(np.sum(mask)) >= 2:
            slope, _ = np.polyfit(t[mask], db[mask], 1)
            if slope < 0:
                return float(-60.0 / slope)
    return float("nan")


def finite_mean(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_rmse(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def finite_pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")
    return pearson(x[mask], y[mask])


def pred_path_for_uid(uid: str, pred_map: Dict[str, str], pred_dir: Path) -> Path:
    if pred_map is not None:
        return Path(pred_map[uid])
    for suffix in [".wav", ".flac", ".npy", ".npz"]:
        path = pred_dir / f"{uid}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No prediction found for {uid} in {pred_dir}")


def main():
    parser = argparse.ArgumentParser()
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref_rir_npz_scp")
    ref_group.add_argument("--ref_rir_scp")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_scp")
    group.add_argument("--pred_dir")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--rir_length", type=int, default=8192)
    parser.add_argument("--target", choices=["reverberant", "anechoic"], default="reverberant")
    parser.add_argument("--source_index", type=int, default=0)
    parser.add_argument("--mic_index", type=int, default=0)
    parser.add_argument("--align", choices=["none", "peak"], default="peak")
    parser.add_argument(
        "--scale_mode",
        choices=["none", "peak", "ref_peak", "optimal"],
        default="peak",
    )
    parser.add_argument("--direct_window_ms", type=float, default=2.5)
    args = parser.parse_args()

    ref_map = read_2col(Path(args.ref_rir_npz_scp or args.ref_rir_scp))
    pred_map = read_2col(Path(args.pred_scp)) if args.pred_scp else None
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if pred_map is not None:
        missing = sorted(set(ref_map) - set(pred_map))
        if missing:
            raise KeyError(f"{len(missing)} reference utterances missing in pred_scp; first={missing[0]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for uid, ref_path in sorted(ref_map.items()):
        if args.ref_rir_npz_scp:
            ref = load_ref_rir(
                Path(ref_path),
                args.target,
                args.source_index,
                args.mic_index,
                args.sample_rate,
            )
        else:
            ref = load_ref_rir_audio(Path(ref_path), args.sample_rate)
        ref = fix_length(ref, args.rir_length)
        pred_path = pred_path_for_uid(uid, pred_map, pred_dir)
        pred, pred_fs = load_audio(pred_path)
        if pred_fs not in (0, args.sample_rate):
            pred = resample_signal(pred, pred_fs, args.sample_rate)
        pred = fix_length(pred, args.rir_length)
        if args.align == "peak":
            pred = peak_align(pred, ref)
        pred_s, ref_s = scale_signals(pred, ref, args.scale_mode)

        n50 = min(args.rir_length, int(round(0.050 * args.sample_rate)))
        pred_rt60 = rt60_schroeder(pred_s, args.sample_rate)
        ref_rt60 = rt60_schroeder(ref_s, args.sample_rate)
        pred_drr = drr_db(pred_s, args.sample_rate, args.direct_window_ms)
        ref_drr = drr_db(ref_s, args.sample_rate, args.direct_window_ms)
        pred_c50 = c50_db(pred_s, args.sample_rate)
        ref_c50 = c50_db(ref_s, args.sample_rate)
        rows.append(
            {
                "utt_id": uid,
                "rmse": float(np.sqrt(np.mean((pred_s - ref_s) ** 2))),
                "rmse_50ms": float(np.sqrt(np.mean((pred_s[:n50] - ref_s[:n50]) ** 2))),
                "corr": pearson(pred_s, ref_s),
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
        )

    csv_path = out_dir / "per_utt.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["utt_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_utts": len(rows),
        "sample_rate": args.sample_rate,
        "rir_length": args.rir_length,
        "align": args.align,
        "scale_mode": args.scale_mode,
        "rmse_mean": finite_mean(r["rmse"] for r in rows),
        "rmse_50ms_mean": finite_mean(r["rmse_50ms"] for r in rows),
        "corr_mean": finite_mean(r["corr"] for r in rows),
        "rt60_mae": finite_mean(abs(r["rt60_err"]) for r in rows),
        "rt60_rmse": finite_rmse(r["rt60_err"] for r in rows),
        "rt60_pearson": finite_pearson((r["rt60_pred"] for r in rows), (r["rt60_ref"] for r in rows)),
        "drr_mae": finite_mean(abs(r["drr_err"]) for r in rows),
        "drr_rmse": finite_rmse(r["drr_err"] for r in rows),
        "drr_pearson": finite_pearson((r["drr_pred"] for r in rows), (r["drr_ref"] for r in rows)),
        "c50_mae": finite_mean(abs(r["c50_err"]) for r in rows),
        "c50_rmse": finite_rmse(r["c50_err"] for r in rows),
        "c50_pearson": finite_pearson((r["c50_pred"] for r in rows), (r["c50_ref"] for r in rows)),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
