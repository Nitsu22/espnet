#!/usr/bin/env python3
"""Paired geometry subgroup analysis of normalized original/same-speaker scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_bimamba_condition_gains import (
    FEATURES, METRICS, bin_index, bootstrap_means, canonical_uid, geometry,
    read_scores, write_csv,
)


SELECTED = ("rt60_s", "mean_distance_m", "distance_gap_m",
            "source_separation_m", "doa_separation_deg", "volume_m3")


def load_scores(folder, expected):
    result = {}
    for metric in METRICS:
        speakers = [read_scores(folder / f"{metric}_spk{i}") for i in (1, 2)]
        if any(set(values) != set(expected) for values in speakers):
            raise ValueError(f"Score key mismatch: {folder}/{metric}")
        result[metric] = np.array([
            (speakers[0][uid] + speakers[1][uid]) / 2 for uid in expected
        ])
    return result


def summarize(baseline, conditioned, mask, rng, reps):
    n = int(mask.sum())
    if n == 0:
        raise ValueError("Empty subgroup")
    row = {"n_mixtures": n}
    for metric in METRICS:
        a, b = baseline[metric][mask], conditioned[metric][mask]
        row[f"baseline_{metric}"] = float(a.mean())
        row[f"concat_{metric}"] = float(b.mean())
        row[f"gain_{metric}"] = float((b-a).mean())
    gain = (conditioned["SI_SNR"] - baseline["SI_SNR"])[mask]
    lo, hi = np.quantile(bootstrap_means(gain, rng, reps), [.025, .975])
    row.update(gain_ci_low=float(lo), gain_ci_high=float(hi),
               win_rate=float((gain > 0).mean()))
    return row


def analyze(args):
    if args.output.exists():
        raise FileExistsError(f"Never overwrite an analysis: {args.output}")
    meta_rows = list(csv.DictReader(args.metadata.open()))
    metadata = {r["utterance_id"]: r for r in meta_rows}
    if len(metadata) != len(meta_rows):
        raise ValueError("Duplicate metadata UID")
    keys = sorted(read_scores(args.original_baseline / "SI_SNR_spk1"))
    if len(keys) != args.expected_count:
        raise ValueError("Unexpected mixture count")
    rows = []
    for uid in keys:
        values = geometry(metadata[canonical_uid(uid)])
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite geometry: {uid}")
        rows.append(dict(uid=uid, **values))
    data = {}
    sources = [args.metadata]
    for condition in ("original", "same"):
        for model in ("baseline", "concat"):
            folder = getattr(args, f"{condition}_{model}")
            data[condition, model] = load_scores(folder, keys)
            sources.extend(folder / f"{metric}_spk{i}"
                           for metric in METRICS for i in (1, 2))
    groups = [("all", "All", np.ones(len(keys), dtype=bool))]
    for feature in SELECTED:
        edges = FEATURES[feature][1]
        idx = bin_index([r[feature] for r in rows], edges)
        for i in range(len(edges)-1):
            label = f"[{edges[i]:g},{edges[i+1]:g}{']' if i == len(edges)-2 else ')'}"
            groups.append((feature, label, idx == i))
    rng = np.random.default_rng(args.seed)
    summaries, contrasts = [], []
    for feature, label, mask in groups:
        for condition in ("original", "same"):
            summaries.append(dict(condition=condition, feature=feature, bin=label,
                                  **summarize(data[condition, "baseline"],
                                              data[condition, "concat"], mask,
                                              rng, args.bootstrap)))
        change = ((data["same", "concat"]["SI_SNR"] - data["same", "baseline"]["SI_SNR"])
                  - (data["original", "concat"]["SI_SNR"] - data["original", "baseline"]["SI_SNR"]))[mask]
        lo, hi = np.quantile(bootstrap_means(change, rng, args.bootstrap), [.025, .975])
        contrasts.append(dict(feature=feature, bin=label, n_mixtures=int(mask.sum()),
                              gain_change_db=float(change.mean()),
                              ci_low=float(lo), ci_high=float(hi)))
    for i, row in enumerate(rows):
        for (condition, model), metrics in data.items():
            for metric, values in metrics.items():
                row[f"{condition}_{model}_{metric}"] = float(values[i])
    args.output.mkdir(parents=True)
    write_csv(args.output / "condition_summary.csv", summaries)
    write_csv(args.output / "gain_changes.csv", contrasts)
    write_csv(args.output / "paired_scenes.csv", rows)
    provenance = dict(
        seed=args.seed, bootstrap=args.bootstrap, count=len(keys),
        metric="ESPnet SI_SNR: zero-mean SI-SDR in dB",
        unit="one paired scene, mean of two reference-aligned source scores",
        distance_definition="3D Euclidean distance; left microphone; mean_distance_m=(d1+d2)/2",
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        boundaries={k: [None if np.isinf(x) else x for x in FEATURES[k][1]] for k in SELECTED},
        caveats=["One training seed; confidence intervals resample scenes, not training runs.",
                 "Pointwise exploratory subgroup intervals; no multiple-comparison correction.",
                 "Scenes share utterances/speakers; intervals condition on this fixed speech pool.",
                 "Room factors are correlated; subgroup trends do not establish causality.",
                 "RT60 is nominal generation T60, not a measured decay estimate.",
                 "Same-speaker scenes change utterance content as well as speaker pairing.",
                 "Both models use normalized output WAVs in both conditions."])
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for row in summaries:
        print(f'{row["condition"]:8s} {row["feature"]:24s} {row["bin"]:14s} '
              f'N={row["n_mixtures"]:4d} baseline={row["baseline_SI_SNR"]:.3f} '
              f'concat={row["concat_SI_SNR"]:.3f} gain={row["gain_SI_SNR"]:+.3f} '
              f'CI=[{row["gain_ci_low"]:+.3f},{row["gain_ci_high"]:+.3f}]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-baseline", "original-concat", "same-baseline", "same-concat", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.expected_count < 1:
        parser.error("Positive bootstrap count and expected mixture count required")
    analyze(args)
