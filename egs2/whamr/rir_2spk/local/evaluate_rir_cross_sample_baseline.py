#!/usr/bin/env python3
"""Compare reference RIRs with disjoint RIR pairs from other utterances.

Report fixed-slot and RMSE-PIT matching separately. Partners are drawn uniformly
from eligible utterances, so repeated BUT RIRs retain test-set weighting. Exclude
both original RIR paths, rather than only excluding the original utterance ID.
"""

import argparse
import csv
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


def room_labels(refs, inventory, room_params):
    if inventory:
        by_name = {r["name"]: r["room"] for r in json.loads(inventory.read_text())}
        labels = {}
        for uid, paths in refs.items():
            pair = [by_name[Path(p).name] for p in paths]
            assert pair[0] == pair[1]
            labels[uid] = pair[0]
        return labels
    with room_params.open() as f:
        params = {
            Path(r["utterance_id"]).stem: tuple(
                r[k] for k in ("room_x", "room_y", "room_z")
            )
            for r in csv.DictReader(f)
        }
    return {uid: params[uid] for uid in refs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison_json", type=Path, required=True)
    parser.add_argument("--but_inventory", type=Path, required=True)
    parser.add_argument("--whamr_room_params", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    assert len(set(args.seeds)) == len(args.seeds)
    comparison = json.loads(args.comparison_json.read_text())
    fs, length = comparison["sample_rate"], comparison["rir_samples"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for condition in ("whamr", "but_clean"):
        cfg = comparison["conditions"][condition]
        root = Path(next(iter(cfg["models"].values()))["summary"]).parent.parent
        refs = read_3col(root / "reference.scp")
        assert len(refs) == comparison["counts"]["mixtures_per_condition"]
        rooms = room_labels(
            refs,
            args.but_inventory if condition == "but_clean" else None,
            args.whamr_room_params,
        )
        if condition == "but_clean":
            noisy = comparison["conditions"]["but_noisy"]
            noisy_root = Path(next(iter(noisy["models"].values()))["summary"])
            assert read_3col(noisy_root.parent.parent / "reference.scp") == refs

        @lru_cache(maxsize=512)
        def load(path):
            return load_rir_audio(Path(path), fs, length)

        @lru_cache(maxsize=4096)
        def pair(p, r):
            return score_pair(
                load(p), load(r), fs, length,
                comparison["align"], comparison["scale_mode"], 2.5,
                polarity_align=cfg["polarity_align"],
            )

        all_uids = sorted(refs)
        rir_sets = {uid: frozenset(paths) for uid, paths in refs.items()}
        scopes = (
            ["different_room"] if condition == "whamr"
            else ["any_room", "same_room", "different_room"]
        )
        results[condition] = {}
        for scope in scopes:
            candidates = {}
            by_rir_set = {}
            for uid in all_uids:
                cache_key = rir_sets[uid]
                if cache_key not in by_rir_set:
                    by_rir_set[cache_key] = tuple(
                        other for other in all_uids
                        if uid != other and not (rir_sets[uid] & rir_sets[other])
                        and (
                            scope == "any_room"
                            or (rooms[uid] == rooms[other]) == (scope == "same_room")
                        )
                    )
                candidates[uid] = by_rir_set[cache_key]
            eligible = [uid for uid in all_uids if candidates[uid]]
            assert eligible
            excluded = [uid for uid in all_uids if not candidates[uid]]
            same_sample, fixed, pit = [], [], []
            rows, seed_summaries = [], {}
            for uid in eligible:
                p = refs[uid]
                same_sample.extend([pair(p[0], p[1]), pair(p[1], p[0])])
            print(
                condition, scope, len(eligible), "eligible utterances", flush=True
            )
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                seed_fixed, seed_pit = [], []
                for index, uid in enumerate(eligible):
                    pool = candidates[uid]
                    other = pool[int(rng.randint(len(pool)))]
                    assert uid != other and not (rir_sets[uid] & rir_sets[other])
                    p, r = refs[other], refs[uid]
                    matrix = [[pair(a, b) for b in r] for a in p]
                    fixed_rows = [matrix[0][0], matrix[1][1]]
                    alternatives = [fixed_rows, [matrix[0][1], matrix[1][0]]]
                    best = min(
                        range(2),
                        key=lambda i: aggregate_pair_metrics(alternatives[i])["rmse"],
                    )
                    pit_rows = alternatives[best]
                    assert aggregate_pair_metrics(pit_rows)["rmse"] <= (
                        aggregate_pair_metrics(fixed_rows)["rmse"] + 1e-15
                    )
                    seed_fixed.extend(fixed_rows)
                    seed_pit.extend(pit_rows)
                    rows.append({
                        "seed": seed, "utt_id": uid, "partner_utt_id": other,
                        "same_room": rooms[uid] == rooms[other], "perm_index": best,
                        "partner_rir1": p[0], "partner_rir2": p[1],
                        **aggregate_pair_metrics(pit_rows),
                    })
                    if (index + 1) % 1000 == 0:
                        print(condition, scope, seed, index + 1, flush=True)
                fixed.extend(seed_fixed)
                pit.extend(seed_pit)
                seed_summaries[str(seed)] = {
                    "fixed_slots": aggregate_pair_metrics(seed_fixed),
                    "pit": aggregate_pair_metrics(seed_pit),
                }
            subset_models = {}
            metric_names = tuple(pair(refs[eligible[0]][0], refs[eligible[0]][0]))
            eligible_set = set(eligible)
            for name, model in cfg["models"].items():
                with (Path(model["summary"]).parent / "per_source.csv").open() as f:
                    metrics = [
                        {k: float(row[k]) for k in metric_names}
                        for row in csv.DictReader(f) if row["utt_id"] in eligible_set
                    ]
                assert len(metrics) == 2 * len(eligible)
                subset_models[name] = aggregate_pair_metrics(metrics)
                if len(eligible) == len(refs):
                    for key, saved in [
                        ("rmse", "rmse_mean"), ("corr", "corr_mean")
                    ]:
                        np.testing.assert_allclose(
                            subset_models[name][key], model["metrics"][saved],
                            rtol=1e-10, atol=1e-12,
                        )
            result = {
                "num_utts": len(eligible), "num_excluded_utts": len(excluded),
                "excluded_rooms": sorted({str(rooms[u]) for u in excluded}),
                "seeds": args.seeds, "num_source_comparisons": len(pit),
                "sample_rate": fs, "rir_length": length,
                "align": comparison["align"],
                "scale_mode": comparison["scale_mode"],
                "polarity_align": cfg["polarity_align"],
                "sampling": (
                    "uniform eligible utterances; both original RIR paths excluded"
                ),
                "same_sample_cross": aggregate_pair_metrics(same_sample),
                "other_sample_fixed_slots": aggregate_pair_metrics(fixed),
                "other_sample_pit": aggregate_pair_metrics(pit),
                "per_seed": seed_summaries, "models_on_same_subset": subset_models,
                "rt60_finite_comparisons": sum(
                    bool(np.isfinite(m["rt60_err"])) for m in pit
                ),
            }
            results[condition][scope] = result
            prefix = condition + "_" + scope
            with (args.out_dir / (prefix + "_pairs.csv")).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (args.out_dir / (prefix + ".json")).write_text(
                json.dumps(result, indent=2)
            )
            print(prefix, result["other_sample_pit"], flush=True)
        load.cache_clear()
        pair.cache_clear()
    (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
