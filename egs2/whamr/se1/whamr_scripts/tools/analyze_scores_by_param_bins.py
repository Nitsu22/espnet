#!/usr/bin/env python3
"""Analyze SI-SNR/SDR by parameter bins (e.g., T60/RT60)."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_params = script_dir.parent / "data" / "reverb_params_tt.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Read scoring files (SI_SNR_spk1/2, SDR_spk1/2), join with parameter CSV, "
            "and report bin-wise means."
        )
    )
    parser.add_argument(
        "--scoring-dir",
        type=Path,
        required=True,
        help=(
            "Path to scoring directory, or its parent directory that contains ./scoring. "
            "Example: .../enhanced_tt_mix_both_reverb_min_8k or .../scoring"
        ),
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
        default="T60",
        help=(
            "Column name in params CSV for binning. "
            "If RT60 is given and T60 exists, T60 is used."
        ),
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="0.1,0.3,0.6,1.0",
        help=(
            "Comma-separated bin edges. "
            "Intervals are [a,b) except the last [a,b]."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="SI_SNR,SDR",
        help="Comma-separated metric prefixes to analyze (default: SI_SNR,SDR).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for aggregated results.",
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
            raise KeyError(
                f"'utterance_id' column not found in parameter CSV: {params_csv}"
            )

        param_map: Dict[str, float] = {}
        for row in reader:
            utt = row["utterance_id"].strip()
            if not utt:
                continue
            value_text = row[param_column]
            if value_text is None or str(value_text).strip() == "":
                raise ValueError(
                    f"Empty value for parameter column '{param_column}' at utterance_id='{utt}'"
                )
            value = float(value_text)
            if utt in param_map:
                raise ValueError(f"Duplicate utterance_id in CSV: {utt}")
            param_map[utt] = value

    return param_map, param_column


def read_metric_file(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric file not found: {path}")
    metric: Dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected metric file format at {path}:{line_no}: '{text}'"
                )
            utt, score_text = parts
            if utt in metric:
                raise ValueError(f"Duplicate utterance key in {path}: {utt}")
            metric[utt] = float(score_text)
    return metric


def strip_reverb_suffix(score_key: str) -> str:
    return score_key[:-7] if score_key.endswith("_reverb") else score_key


def map_score_key_to_utterance_id(
    score_key: str, params_set: Collection[str]
) -> Tuple[Optional[str], Optional[str]]:
    base = strip_reverb_suffix(score_key)
    candidate_direct = f"{base}.wav"
    if candidate_direct in params_set:
        return candidate_direct, "direct"

    parts = base.split("_")
    if len(parts) >= 3:
        candidate_drop2 = f"{'_'.join(parts[2:])}.wav"
        if candidate_drop2 in params_set:
            return candidate_drop2, "drop2prefix"

    return None, None


def find_bin_index(value: float, edges: Sequence[float]) -> Optional[int]:
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        if i == len(edges) - 2:
            if lower <= value <= upper:
                return i
        else:
            if lower <= value < upper:
                return i
    return None


def format_bin_label(index: int, edges: Sequence[float]) -> str:
    lower = edges[index]
    upper = edges[index + 1]
    if index == len(edges) - 2:
        return f"[{lower:.6g}, {upper:.6g}]"
    return f"[{lower:.6g}, {upper:.6g})"


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def analyze_metric(
    scoring_dir: Path,
    metric_name: str,
    param_map: Dict[str, float],
    edges: Sequence[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    spk1_path = scoring_dir / f"{metric_name}_spk1"
    spk2_path = scoring_dir / f"{metric_name}_spk2"

    spk1 = read_metric_file(spk1_path)
    spk2 = read_metric_file(spk2_path)

    keys1 = set(spk1.keys())
    keys2 = set(spk2.keys())
    if keys1 != keys2:
        only1 = sorted(keys1 - keys2)[:5]
        only2 = sorted(keys2 - keys1)[:5]
        raise ValueError(
            f"Key mismatch between {spk1_path.name} and {spk2_path.name}. "
            f"only_spk1_sample={only1}, only_spk2_sample={only2}"
        )

    params_set = set(param_map.keys())
    by_bin_spk1: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    by_bin_spk2: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    by_bin_avg: List[List[float]] = [[] for _ in range(len(edges) - 1)]

    unresolved: List[str] = []
    out_of_bin = 0
    mapping_counts = {"direct": 0, "drop2prefix": 0}

    for score_key in sorted(keys1):
        utterance_id, mapping_kind = map_score_key_to_utterance_id(score_key, params_set)
        if utterance_id is None:
            unresolved.append(score_key)
            continue

        if mapping_kind is not None:
            mapping_counts[mapping_kind] += 1

        param_value = param_map[utterance_id]
        bin_idx = find_bin_index(param_value, edges)
        if bin_idx is None:
            out_of_bin += 1
            continue

        v1 = spk1[score_key]
        v2 = spk2[score_key]
        by_bin_spk1[bin_idx].append(v1)
        by_bin_spk2[bin_idx].append(v2)
        by_bin_avg[bin_idx].append((v1 + v2) / 2.0)

    if unresolved:
        sample = unresolved[:5]
        raise ValueError(
            f"Could not map {len(unresolved)} scoring keys to params CSV. sample={sample}"
        )

    rows: List[Dict[str, object]] = []
    for idx in range(len(edges) - 1):
        rows.append(
            {
                "metric": metric_name,
                "bin_index": idx,
                "bin_label": format_bin_label(idx, edges),
                "bin_lower": edges[idx],
                "bin_upper": edges[idx + 1],
                "count": len(by_bin_avg[idx]),
                "mean_spk1": mean_or_none(by_bin_spk1[idx]),
                "mean_spk2": mean_or_none(by_bin_spk2[idx]),
                "mean_spk_avg": mean_or_none(by_bin_avg[idx]),
            }
        )

    meta: Dict[str, object] = {
        "total_scored": len(keys1),
        "out_of_bin": out_of_bin,
        "mapping_counts": mapping_counts,
    }
    return rows, meta


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def print_report(
    rows: Sequence[Dict[str, object]],
    meta: Dict[str, object],
    param_column: str,
) -> None:
    if not rows:
        return
    metric_name = str(rows[0]["metric"])
    print(f"\n=== {metric_name} by {param_column} bins ===")
    print(f"total_scored: {meta['total_scored']}")
    print(f"out_of_bin:   {meta['out_of_bin']}")
    mapping_counts = meta["mapping_counts"]
    print(f"mapping:      {mapping_counts}")
    print("bin\tcount\tmean_spk1\tmean_spk2\tmean_spk_avg")
    for row in rows:
        print(
            f"{row['bin_label']}\t{row['count']}\t"
            f"{format_value(row['mean_spk1'])}\t"
            f"{format_value(row['mean_spk2'])}\t"
            f"{format_value(row['mean_spk_avg'])}"
        )


def write_output_csv(
    output_csv: Path,
    rows: Sequence[Dict[str, object]],
    param_column: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "parameter",
        "bin_index",
        "bin_label",
        "bin_lower",
        "bin_upper",
        "count",
        "mean_spk1",
        "mean_spk2",
        "mean_spk_avg",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["parameter"] = param_column
            writer.writerow(out_row)


def main() -> int:
    args = parse_args()
    scoring_dir = resolve_scoring_dir(args.scoring_dir.resolve())
    edges = parse_bin_edges(args.bins)
    metric_names = [x.strip() for x in args.metrics.split(",") if x.strip()]
    if not metric_names:
        raise ValueError("No metrics specified.")

    param_map, resolved_param_col = read_params(args.params_csv.resolve(), args.param_column)

    all_rows: List[Dict[str, object]] = []
    for metric_name in metric_names:
        rows, meta = analyze_metric(scoring_dir, metric_name, param_map, edges)
        print_report(rows, meta, resolved_param_col)
        all_rows.extend(rows)

    if args.output_csv is not None:
        write_output_csv(args.output_csv.resolve(), all_rows, resolved_param_col)
        print(f"\nSaved aggregated results to: {args.output_csv.resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
