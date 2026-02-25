#!/usr/bin/env python3
"""Plot a simple 2-box boxplot: baseline vs proposed."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create a single PNG with two boxplots: baseline and proposed."
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
        "--metric",
        type=str,
        default="SI_SNR",
        help="Metric prefix (e.g., SI_SNR or SDR).",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=script_dir / "dump" / "baseline_vs_proposed_SI_SNR_boxplot.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--hide-fliers",
        action="store_true",
        help="Hide outlier points in boxplot.",
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
    keys1 = set(spk1.keys())
    keys2 = set(spk2.keys())
    if keys1 != keys2:
        only1 = sorted(keys1 - keys2)[:5]
        only2 = sorted(keys2 - keys1)[:5]
        raise ValueError(
            f"Key mismatch for {metric}: only_spk1_sample={only1}, only_spk2_sample={only2}"
        )
    return {k: (spk1[k] + spk2[k]) / 2.0 for k in spk1}


def align_values(
    baseline: Dict[str, float], proposed: Dict[str, float]
) -> Tuple[List[float], List[float]]:
    common = sorted(set(baseline.keys()) & set(proposed.keys()))
    if not common:
        raise ValueError("No common keys between baseline and proposed.")
    bvals = [baseline[k] for k in common]
    pvals = [proposed[k] for k in common]
    return bvals, pvals


def format_stats(values: List[float]) -> str:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return "n=0"
    median = vals[n // 2] if (n % 2 == 1) else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return (
        f"n={n}, mean={statistics.fmean(vals):.6f}, median={median:.6f}, "
        f"min={vals[0]:.6f}, max={vals[-1]:.6f}"
    )


def main() -> int:
    args = parse_args()
    baseline_scoring = resolve_scoring_dir(args.baseline_scoring_dir.resolve())
    proposed_scoring = resolve_scoring_dir(args.proposed_scoring_dir.resolve())

    baseline = read_metric_avg(baseline_scoring, args.metric)
    proposed = read_metric_avg(proposed_scoring, args.metric)
    bvals, pvals = align_values(baseline, proposed)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    labels = [f"baseline\n(n={len(bvals)})", f"proposed\n(n={len(pvals)})"]
    ax.boxplot([bvals, pvals], tick_labels=labels, showfliers=not args.hide_fliers)
    ax.set_title(f"{args.metric} distribution: baseline vs proposed")
    ax.set_ylabel(f"{args.metric} [dB]")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()

    out = args.output_png.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print(f"saved_png={out}")
    print(f"baseline_stats: {format_stats(bvals)}")
    print(f"proposed_stats: {format_stats(pvals)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
