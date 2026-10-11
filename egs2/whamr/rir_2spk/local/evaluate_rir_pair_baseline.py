#!/usr/bin/env python3
"""Compare two reference RIRs and evaluate duplicated single-RIR outputs.

Reuse score_rir_pit preprocessing exactly. Reference duplication is an oracle
diagnostic, not a realizable estimator. Output duplication uses each existing
model slot in turn; the two slot variants are reported separately and averaged.
"""

import argparse
import csv
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from score_rir_pit import (
    aggregate_pair_metrics,
    load_rir_audio,
    read_3col,
    score_pair,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison_json.read_text())
    fs, length = comparison["sample_rate"], comparison["rir_samples"]
    results = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition, cfg in comparison["conditions"].items():
        models = cfg["models"]
        first = Path(next(iter(models.values()))["summary"]).parent.parent
        refs = read_3col(first / "reference.scp")
        assert len(refs) == comparison["counts"]["mixtures_per_condition"]

        @lru_cache(maxsize=512)
        def load(path):
            return load_rir_audio(Path(path), fs, length)

        @lru_cache(maxsize=2048)
        def pair(pred_path, ref_path):
            return score_pair(
                load(pred_path),
                load(ref_path),
                fs,
                length,
                comparison["align"],
                comparison["scale_mode"],
                2.5,
                polarity_align=cfg["polarity_align"],
            )

        def aggregate(rows):
            return aggregate_pair_metrics(rows)

        rows = []
        cross, oracle = [], [[], []]
        for uid, paths in sorted(refs.items()):
            both = [pair(paths[0], paths[1]), pair(paths[1], paths[0])]
            cross.extend(both)
            rows.append({"utt_id": uid, **aggregate(both)})
            for slot in range(2):
                diagonal = pair(paths[slot], paths[slot])
                assert diagonal["rmse"] == 0
                assert np.isclose(diagonal["corr"], 1)
                oracle[slot].extend(pair(paths[slot], p) for p in paths)
        result = {
            "num_utts": len(refs),
            "sample_rate": fs,
            "rir_length": length,
            "align": comparison["align"],
            "scale_mode": comparison["scale_mode"],
            "polarity_align": cfg["polarity_align"],
            "reference_cross_bidirectional": aggregate(cross),
            "oracle_duplicate_reference1": aggregate(oracle[0]),
            "oracle_duplicate_reference2": aggregate(oracle[1]),
            "oracle_duplicate_average_over_slots": aggregate(oracle[0] + oracle[1]),
            "models": {},
        }
        for name, model in models.items():
            score_dir = Path(model["summary"]).parent
            root = score_dir.parent
            assert read_3col(root / "reference.scp") == refs
            predictions = read_3col(root / "inference" / "rir.scp")
            assert set(predictions) == set(refs)
            normal, duplicate = [], [[], []]
            for uid, ref_paths in sorted(refs.items()):
                pred_paths = predictions[uid]
                matrix = [[pair(p, r) for r in ref_paths] for p in pred_paths]
                candidates = [
                    [matrix[i][j] for i, j in enumerate(perm)]
                    for perm in itertools.permutations(range(2))
                ]
                normal.extend(
                    min(candidates, key=lambda x: aggregate(x)["rmse"])
                )
                for slot in range(2):
                    duplicate[slot].extend(matrix[slot])
            normal_summary = aggregate(normal)
            stored = model["metrics"]
            for key, stored_key in [
                ("rmse", "rmse_mean"),
                ("corr", "corr_mean"),
                ("rt60_mae", "rt60_mae"),
                ("drr_mae", "drr_mae"),
                ("c50_mae", "c50_mae"),
            ]:
                np.testing.assert_allclose(
                    normal_summary[key], stored[stored_key],
                    rtol=1e-10, atol=1e-12,
                )
            result["models"][name] = {
                "normal_two_outputs": normal_summary,
                "duplicate_output1": aggregate(duplicate[0]),
                "duplicate_output2": aggregate(duplicate[1]),
                "duplicate_average_over_slots": aggregate(
                    duplicate[0] + duplicate[1]
                ),
            }
            print(condition, name, result["models"][name], flush=True)
        results[condition] = result
        with (args.out_dir / (condition + "_reference_pairs.csv")).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.out_dir / (condition + ".json")).write_text(
            json.dumps(result, indent=2)
        )
        print(
            condition, "reference cross",
            result["reference_cross_bidirectional"], flush=True,
        )
        load.cache_clear()
        pair.cache_clear()
    (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
