#!/usr/bin/env python3
"""Create boxplots of score deltas against T60/DOA/SNR-proxy bins."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_params = (
        script_dir.parent / "se2" / "whamr_scripts" / "data" / "reverb_params_tt.csv"
    )
    default_out = script_dir / "dump"

    parser = argparse.ArgumentParser(
        description=(
            "Generate boxplots for delta scores "
            "(proposed - baseline) by T60 / absolute DOA / SNR-proxy bins."
        )
    )
    parser.add_argument(
        "--baseline-scoring-dir",
        type=Path,
        required=True,
        help="Baseline scoring dir, or parent containing ./scoring.",
    )
    parser.add_argument(
        "--proposed-scoring-dir",
        type=Path,
        required=True,
        help="Proposed scoring dir, or parent containing ./scoring.",
    )
    parser.add_argument(
        "--params-csv",
        type=Path,
        default=default_params,
        help=f"Parameter CSV path (default: {default_params})",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="SI_SNR,SDR",
        help="Comma-separated metric prefixes (default: SI_SNR,SDR).",
    )
    parser.add_argument(
        "--t60-bins",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated edges for T60 bins.",
    )
    parser.add_argument(
        "--doa-bins",
        type=str,
        default="0,30,60,90,120,150,180",
        help="Comma-separated edges for absolute DOA-average bins (degree).",
    )
    parser.add_argument(
        "--snr-bins",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated edges for SNR-proxy bins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Output directory for PNGs (default: {default_out})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="baseline_vs_proposed",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--hide-fliers",
        action="store_true",
        help="Hide outlier points in boxplots.",
    )
    return parser.parse_args()


def resolve_scoring_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Scoring path does not exist: {path}")
    if path.is_dir() and (path / "scoring").is_dir():
        nested = path / "scoring"
        if (nested / "SI_SNR_spk1").is_file():
            return nested
    return path


def parse_edges(text: str) -> List[float]:
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if len(parts) < 2:
        raise ValueError(f"Need at least 2 bin edges: {text}")
    edges = [float(x) for x in parts]
    for i in range(1, len(edges)):
        if not edges[i] > edges[i - 1]:
            raise ValueError(f"Edges must be strictly increasing: {edges}")
    return edges


def read_metric_file(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric file not found: {path}")
    out: Dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2:
                raise ValueError(f"Unexpected format at {path}:{line_no}: {text}")
            key, value = parts
            if key in out:
                raise ValueError(f"Duplicate key in {path}: {key}")
            out[key] = float(value)
    return out


def read_metric_avg(scoring_dir: Path, metric: str) -> Dict[str, float]:
    spk1 = read_metric_file(scoring_dir / f"{metric}_spk1")
    spk2 = read_metric_file(scoring_dir / f"{metric}_spk2")
    if set(spk1.keys()) != set(spk2.keys()):
        only1 = sorted(set(spk1.keys()) - set(spk2.keys()))[:5]
        only2 = sorted(set(spk2.keys()) - set(spk1.keys()))[:5]
        raise ValueError(
            f"Key mismatch for {metric}: only_spk1_sample={only1}, only_spk2_sample={only2}"
        )
    return {k: (spk1[k] + spk2[k]) / 2.0 for k in spk1.keys()}


def strip_reverb_suffix(score_key: str) -> str:
    return score_key[:-7] if score_key.endswith("_reverb") else score_key


def map_score_key(score_key: str, param_keys: set[str]) -> Optional[str]:
    base = strip_reverb_suffix(score_key)
    direct = f"{base}.wav"
    if direct in param_keys:
        return direct
    parts = base.split("_")
    if len(parts) >= 3:
        drop2 = f"{'_'.join(parts[2:])}.wav"
        if drop2 in param_keys:
            return drop2
    return None


def parse_snr_proxy_abs(score_key: str) -> Optional[float]:
    # Score key layout includes "..._<v1>_..._<v2>_reverb"
    base = strip_reverb_suffix(score_key)
    parts = base.split("_")
    if len(parts) < 6:
        return None
    try:
        v1 = float(parts[3])
        v2 = float(parts[5])
    except ValueError:
        return None
    return abs(v1 - v2)


def abs_doa_wrt_array_axis(
    source_x: float,
    source_y: float,
    mic_l_x: float,
    mic_l_y: float,
    mic_r_x: float,
    mic_r_y: float,
) -> float:
    mic_c_x = (mic_l_x + mic_r_x) / 2.0
    mic_c_y = (mic_l_y + mic_r_y) / 2.0
    axis_deg = math.degrees(math.atan2(mic_r_y - mic_l_y, mic_r_x - mic_l_x))
    src_deg = math.degrees(math.atan2(source_y - mic_c_y, source_x - mic_c_x))
    diff = ((src_deg - axis_deg + 180.0) % 360.0) - 180.0
    return abs(diff)


def read_param_features(params_csv: Path) -> Dict[str, Dict[str, float]]:
    if not params_csv.is_file():
        raise FileNotFoundError(f"Parameter CSV not found: {params_csv}")
    out: Dict[str, Dict[str, float]] = {}
    with params_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {params_csv}")
        required = [
            "utterance_id",
            "T60",
            "micL_x",
            "micL_y",
            "micR_x",
            "micR_y",
            "s1_x",
            "s1_y",
            "s2_x",
            "s2_y",
        ]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"Missing columns in {params_csv}: {missing}")

        for row in reader:
            utt = row["utterance_id"].strip()
            if not utt:
                continue
            t60 = float(row["T60"])
            mlx = float(row["micL_x"])
            mly = float(row["micL_y"])
            mrx = float(row["micR_x"])
            mry = float(row["micR_y"])
            d1 = abs_doa_wrt_array_axis(
                float(row["s1_x"]),
                float(row["s1_y"]),
                mlx,
                mly,
                mrx,
                mry,
            )
            d2 = abs_doa_wrt_array_axis(
                float(row["s2_x"]),
                float(row["s2_y"]),
                mlx,
                mly,
                mrx,
                mry,
            )
            out[utt] = {"T60": t60, "doa_avg_abs": (d1 + d2) / 2.0}
    return out


def find_bin_index(value: float, edges: Sequence[float]) -> Optional[int]:
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            if lo <= value <= hi:
                return i
        else:
            if lo <= value < hi:
                return i
    return None


def make_bin_label(i: int, edges: Sequence[float], value_digits: int = 3) -> str:
    lo = edges[i]
    hi = edges[i + 1]
    lo_s = f"{lo:.{value_digits}g}"
    hi_s = f"{hi:.{value_digits}g}"
    if i == len(edges) - 2:
        return f"[{lo_s}, {hi_s}]"
    return f"[{lo_s}, {hi_s})"


def collect_records(
    baseline_scoring: Path,
    proposed_scoring: Path,
    params: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
) -> List[Dict[str, object]]:
    base_metric = {m: read_metric_avg(baseline_scoring, m) for m in metrics}
    prop_metric = {m: read_metric_avg(proposed_scoring, m) for m in metrics}

    common_keys = set.intersection(*(set(base_metric[m].keys()) for m in metrics))
    common_keys &= set.intersection(*(set(prop_metric[m].keys()) for m in metrics))
    if not common_keys:
        raise ValueError("No common score keys between baseline/proposed metrics.")

    params_keys = set(params.keys())
    unresolved: List[str] = []
    records: List[Dict[str, object]] = []
    for key in sorted(common_keys):
        utt = map_score_key(key, params_keys)
        if utt is None:
            unresolved.append(key)
            continue

        rec: Dict[str, object] = {
            "score_key": key,
            "utterance_id": utt,
            "T60": params[utt]["T60"],
            "doa_avg_abs": params[utt]["doa_avg_abs"],
            "snr_proxy_abs": parse_snr_proxy_abs(key),
        }
        for m in metrics:
            rec[f"delta_{m}"] = prop_metric[m][key] - base_metric[m][key]
        records.append(rec)

    if unresolved:
        sample = unresolved[:5]
        raise ValueError(
            f"Could not map {len(unresolved)} score keys to params CSV. sample={sample}"
        )
    return records


def bin_metric_values(
    records: Sequence[Dict[str, object]],
    feature_key: str,
    metric_key: str,
    edges: Sequence[float],
) -> Tuple[List[str], List[List[float]], List[int], int]:
    grouped: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    dropped = 0
    for rec in records:
        feature = rec.get(feature_key)
        metric = rec.get(metric_key)
        if feature is None or metric is None:
            dropped += 1
            continue
        idx = find_bin_index(float(feature), edges)
        if idx is None:
            dropped += 1
            continue
        grouped[idx].append(float(metric))

    labels = [make_bin_label(i, edges) for i in range(len(edges) - 1)]
    counts = [len(v) for v in grouped]
    return labels, grouped, counts, dropped


def plot_boxplot(
    labels: Sequence[str],
    values: Sequence[Sequence[float]],
    counts: Sequence[int],
    title: str,
    ylabel: str,
    output_png: Path,
    showfliers: bool,
) -> None:
    compact = [
        (labels[i], values[i], counts[i]) for i in range(len(labels)) if len(values[i]) > 0
    ]
    if not compact:
        raise ValueError(f"No values to plot for {output_png}")

    plot_labels = [f"{lab}\n(n={cnt})" for lab, _, cnt in compact]
    plot_values = [vals for _, vals, _ in compact]

    width = max(10.0, 1.3 * len(plot_labels))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.boxplot(plot_values, tick_labels=plot_labels, showfliers=showfliers)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Condition bins")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def write_summary(
    summary_path: Path,
    records: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    t60_counts: Sequence[int],
    doa_counts: Sequence[int],
    snr_counts: Sequence[int],
    dropped_t60: int,
    dropped_doa: int,
    dropped_snr: int,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"records={len(records)}\n")
        f.write(f"metrics={','.join(metrics)}\n")
        for m in metrics:
            key = f"delta_{m}"
            vals = [float(r[key]) for r in records]
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            median = vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
            f.write(
                f"{key}: mean={sum(vals)/n:.6f}, median={median:.6f}, "
                f"min={vals_sorted[0]:.6f}, max={vals_sorted[-1]:.6f}\n"
            )
        f.write(f"T60 counts={list(t60_counts)}, dropped={dropped_t60}\n")
        f.write(f"DOA counts={list(doa_counts)}, dropped={dropped_doa}\n")
        f.write(f"SNRproxy counts={list(snr_counts)}, dropped={dropped_snr}\n")


def main() -> int:
    args = parse_args()
    baseline_scoring = resolve_scoring_dir(args.baseline_scoring_dir.resolve())
    proposed_scoring = resolve_scoring_dir(args.proposed_scoring_dir.resolve())
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    if not metrics:
        raise ValueError("No metrics specified.")

    t60_edges = parse_edges(args.t60_bins)
    doa_edges = parse_edges(args.doa_bins)
    snr_edges = parse_edges(args.snr_bins)
    params = read_param_features(args.params_csv.resolve())
    records = collect_records(baseline_scoring, proposed_scoring, params, metrics)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t60_labels: List[str] = []
    doa_labels: List[str] = []
    snr_labels: List[str] = []
    t60_counts: List[int] = []
    doa_counts: List[int] = []
    snr_counts: List[int] = []
    dropped_t60 = 0
    dropped_doa = 0
    dropped_snr = 0

    for metric in metrics:
        metric_key = f"delta_{metric}"

        t60_labels, t60_values, t60_counts, dropped_t60 = bin_metric_values(
            records, "T60", metric_key, t60_edges
        )
        plot_boxplot(
            labels=t60_labels,
            values=t60_values,
            counts=t60_counts,
            title=f"{metric_key} by T60 bins",
            ylabel=f"{metric_key} [dB]",
            output_png=output_dir / f"{args.prefix}_{metric_key}_boxplot_t60.png",
            showfliers=not args.hide_fliers,
        )

        doa_labels, doa_values, doa_counts, dropped_doa = bin_metric_values(
            records, "doa_avg_abs", metric_key, doa_edges
        )
        plot_boxplot(
            labels=doa_labels,
            values=doa_values,
            counts=doa_counts,
            title=f"{metric_key} by absolute DOA-average bins",
            ylabel=f"{metric_key} [dB]",
            output_png=output_dir / f"{args.prefix}_{metric_key}_boxplot_doa.png",
            showfliers=not args.hide_fliers,
        )

        snr_labels, snr_values, snr_counts, dropped_snr = bin_metric_values(
            records, "snr_proxy_abs", metric_key, snr_edges
        )
        plot_boxplot(
            labels=snr_labels,
            values=snr_values,
            counts=snr_counts,
            title=f"{metric_key} by SNR-proxy bins (|v1-v2|)",
            ylabel=f"{metric_key} [dB]",
            output_png=output_dir / f"{args.prefix}_{metric_key}_boxplot_snrproxy.png",
            showfliers=not args.hide_fliers,
        )

    summary = output_dir / f"{args.prefix}_boxplot_summary.txt"
    write_summary(
        summary_path=summary,
        records=records,
        metrics=metrics,
        t60_counts=t60_counts,
        doa_counts=doa_counts,
        snr_counts=snr_counts,
        dropped_t60=dropped_t60,
        dropped_doa=dropped_doa,
        dropped_snr=dropped_snr,
    )

    print(f"Saved boxplots and summary to: {output_dir}")
    print(f"records={len(records)}")
    print(f"metrics={metrics}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
