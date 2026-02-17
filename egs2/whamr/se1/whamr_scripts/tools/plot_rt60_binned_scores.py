#!/usr/bin/env python3
"""Plot SI-SNR/SDR mean scores by RT60(T60) bins."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_params = script_dir.parent / "data" / "reverb_params_tt.csv"
    default_output = script_dir / "png" / "rt60_binned_scores.png"

    parser = argparse.ArgumentParser(
        description="Create RT60(T60)-binned bar plots."
    )
    parser.add_argument(
        "--scoring-dir",
        type=Path,
        required=True,
        help="Path to scoring directory, or parent containing ./scoring.",
    )
    parser.add_argument(
        "--params-csv",
        type=Path,
        default=default_params,
        help=f"Parameter CSV path (default: {default_params})",
    )
    parser.add_argument(
        "--param-column",
        type=str,
        default="RT60",
        help="Binning column in params CSV. RT60 is resolved to T60 if present.",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="0.1,0.3,0.6,1.0",
        help="Comma-separated bin edges. [a,b) except last [a,b].",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=default_output,
        help=f"Output PNG path (default: {default_output})",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RT60-binned Mean Scores",
        help="Figure title.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="BOTH",
        choices=["BOTH", "SI_SNR", "SDR"],
        help="Plot metric selection (default: BOTH).",
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


def parse_bin_edges(bin_text: str) -> List[float]:
    parts = [x.strip() for x in bin_text.split(",") if x.strip()]
    if len(parts) < 2:
        raise ValueError("At least 2 bin edges are required.")
    edges = [float(x) for x in parts]
    for idx in range(1, len(edges)):
        if not edges[idx] > edges[idx - 1]:
            raise ValueError(f"Bin edges must be strictly increasing: {edges}")
    return edges


def resolve_param_column(header: Sequence[str], requested: str) -> str:
    if requested in header:
        return requested
    if requested.upper() == "RT60" and "T60" in header:
        return "T60"
    raise KeyError(
        f"Parameter column '{requested}' not found in CSV header: {list(header)}"
    )


def read_params(params_csv: Path, requested_column: str) -> Tuple[Dict[str, float], str]:
    if not params_csv.is_file():
        raise FileNotFoundError(f"Parameter CSV not found: {params_csv}")

    with params_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {params_csv}")
        param_column = resolve_param_column(reader.fieldnames, requested_column)
        if "utterance_id" not in reader.fieldnames:
            raise KeyError("'utterance_id' column not found in parameter CSV")

        out: Dict[str, float] = {}
        for row in reader:
            utt = row["utterance_id"].strip()
            if not utt:
                continue
            if utt in out:
                raise ValueError(f"Duplicate utterance_id in parameter CSV: {utt}")
            out[utt] = float(row[param_column])
    return out, param_column


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
                raise ValueError(f"Unexpected format at {path}:{line_no}: '{text}'")
            key, value = parts
            if key in out:
                raise ValueError(f"Duplicate key in {path}: {key}")
            out[key] = float(value)
    return out


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


def bin_label(i: int, edges: Sequence[float]) -> str:
    lo = edges[i]
    hi = edges[i + 1]
    if i == len(edges) - 2:
        return f"[{lo:.3g}, {hi:.3g}]"
    return f"[{lo:.3g}, {hi:.3g})"


def aggregate_metric(
    scoring_dir: Path,
    metric_name: str,
    param_map: Dict[str, float],
    edges: Sequence[float],
) -> Tuple[List[str], List[int], List[float]]:
    spk1 = read_metric_file(scoring_dir / f"{metric_name}_spk1")
    spk2 = read_metric_file(scoring_dir / f"{metric_name}_spk2")
    if set(spk1.keys()) != set(spk2.keys()):
        raise ValueError(f"Key mismatch between {metric_name}_spk1 and {metric_name}_spk2")

    param_keys = set(param_map.keys())
    sums = [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)

    unresolved = 0
    out_of_bin = 0
    for key in spk1.keys():
        utt = map_score_key(key, param_keys)
        if utt is None:
            unresolved += 1
            continue
        idx = find_bin_index(param_map[utt], edges)
        if idx is None:
            out_of_bin += 1
            continue
        mean_spk = (spk1[key] + spk2[key]) / 2.0
        sums[idx] += mean_spk
        counts[idx] += 1

    if unresolved > 0:
        raise ValueError(f"{metric_name}: {unresolved} keys could not be mapped to params CSV")
    if out_of_bin > 0:
        raise ValueError(f"{metric_name}: {out_of_bin} keys are out of bin range")

    means = [sums[i] / counts[i] if counts[i] > 0 else float("nan") for i in range(len(counts))]
    labels = [bin_label(i, edges) for i in range(len(edges) - 1)]
    return labels, counts, means


def plot_results(
    labels: Sequence[str],
    counts: Sequence[int],
    metric_series: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    xticks = [f"{labels[i]}\n(n={counts[i]})" for i in range(len(labels))]
    x = list(range(len(labels)))

    ncols = len(metric_series)
    if ncols < 1:
        raise ValueError("No metric series to plot.")
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.8), dpi=180, constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    if ncols == 1:
        axes = [axes]

    for i, (display_name, means, color) in enumerate(metric_series):
        bars = axes[i].bar(x, means, color=color)
        axes[i].set_title(f"{display_name} (mean_spk_avg)")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(xticks)
        axes[i].set_ylabel("Score (dB)")
        axes[i].grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars:
            h = bar.get_height()
            axes[i].annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.savefig(output_png)
    plt.close(fig)


def print_table(
    labels: Sequence[str],
    counts: Sequence[int],
    si_snr_means: Optional[Sequence[float]] = None,
    sdr_means: Optional[Sequence[float]] = None,
) -> None:
    columns = ["bin", "count"]
    if si_snr_means is not None:
        columns.append("SI-SNR_mean_spk_avg")
    if sdr_means is not None:
        columns.append("SDR_mean_spk_avg")
    print("\t".join(columns))
    for i in range(len(labels)):
        row = [labels[i], str(counts[i])]
        if si_snr_means is not None:
            row.append(f"{si_snr_means[i]:.6f}")
        if sdr_means is not None:
            row.append(f"{sdr_means[i]:.6f}")
        print("\t".join(row))


def main() -> int:
    args = parse_args()
    scoring_dir = resolve_scoring_dir(args.scoring_dir.resolve())
    edges = parse_bin_edges(args.bins)
    param_map, param_col = read_params(args.params_csv.resolve(), args.param_column)

    print(f"parameter_column: {param_col}")
    metric_series: List[Tuple[str, Sequence[float], str]] = []

    if args.metric in ("BOTH", "SI_SNR"):
        labels_si, counts_si, si_snr_means = aggregate_metric(scoring_dir, "SI_SNR", param_map, edges)
        metric_series.append(("SI-SNR", si_snr_means, "#2a9d8f"))
    else:
        labels_si, counts_si, si_snr_means = None, None, None

    if args.metric in ("BOTH", "SDR"):
        labels_sdr, counts_sdr, sdr_means = aggregate_metric(scoring_dir, "SDR", param_map, edges)
        metric_series.append(("SDR", sdr_means, "#457b9d"))
    else:
        labels_sdr, counts_sdr, sdr_means = None, None, None

    labels = labels_si if labels_si is not None else labels_sdr
    counts = counts_si if counts_si is not None else counts_sdr
    if labels is None or counts is None:
        raise ValueError("No metric selected.")

    if labels_si is not None and labels_sdr is not None and (labels_si != labels_sdr or counts_si != counts_sdr):
        raise ValueError("SI_SNR and SDR aggregation produced inconsistent bins/counts.")

    print_table(labels, counts, si_snr_means=si_snr_means, sdr_means=sdr_means)
    plot_results(labels, counts, metric_series, args.title, args.output_png.resolve())
    print(f"saved_png: {args.output_png.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
