#!/usr/bin/env python3
"""Compare fixed models on original and same-speaker mixtures of the same scenes.

The original scene UID is retained by the test builder; it does NOT identify
the new speaker identities. Scores are reference-aligned by ESPnet beforehand.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from analyze_bimamba_condition_gains import (
    METRICS, bootstrap_means, canonical_uid, read_scores, write_csv,
)


def load_pair(baseline, bimamba, expected):
    data = {}
    for model, root in (("baseline", baseline), ("bimamba", bimamba)):
        for metric in METRICS:
            values = [read_scores(root / f"{metric}_spk{i}") for i in (1, 2)]
            if any(set(v) != set(expected) for v in values):
                raise ValueError(f"Score keys differ: {root}/{metric}")
            data[model, metric] = np.array([
                (values[0][uid] + values[1][uid]) / 2 for uid in expected
            ])
    return data


def validate_generation(test_dir, keys):
    root = test_dir.parent.parent
    manifest_path, audit_path = root / "preparation.json", root / "audit.tsv"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["scene_count"] != len(keys)
            or not manifest["original_scene_ids_retained"]
            or manifest["train_valid_created"]
            or not manifest["test_train_valid_utterances_and_speakers_disjoint"]
            or manifest["sample_rate"] != 8000 or manifest["subtype"] != "PCM_16"):
        raise ValueError("Not a verified test-only same-speaker generation")
    with audit_path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(keys) or {r["scene_uid"] for r in rows} != set(keys):
        raise ValueError("Same-speaker generation audit keys differ")
    for row in rows:
        a, b = Path(row["source1"]), Path(row["source2"])
        if (a == b or a.parent.name != b.parent.name
                or a.parent.name != row["actual_speaker"]
                or row["anchor_slot"] not in ("1", "2")
                or row[f'source{row["anchor_slot"]}'] != row[f'original_source{row["anchor_slot"]}']):
            raise ValueError("Same-person/distinct-utterance/anchor invariant failed")
    for path in (audit_path, *(test_dir / name for name in ("wav.scp", "spk1.scp", "spk2.scp"))):
        relative = str(path.relative_to(root))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["output_hashes"][relative]:
            raise ValueError(f"Generation metadata changed: {path}")
    return [manifest_path, audit_path]


def summarize(original, same, mask, rng, reps):
    count = int(np.sum(mask))
    if not count:
        raise ValueError("Empty analysis group")
    result = {"n_mixtures": count}
    for condition, data in (("original", original), ("same_speaker", same)):
        for metric in METRICS:
            for model in ("baseline", "bimamba"):
                result[f"{condition}_{model}_{metric}"] = float(data[model, metric][mask].mean())
            gain = data["bimamba", metric][mask] - data["baseline", metric][mask]
            result[f"{condition}_gain_{metric}"] = float(gain.mean())
        gain = data["bimamba", "SI_SNR"][mask] - data["baseline", "SI_SNR"][mask]
        lo, hi = np.quantile(bootstrap_means(gain, rng, reps), [.025, .975])
        result.update({f"{condition}_gain_ci_low": float(lo),
                       f"{condition}_gain_ci_high": float(hi),
                       f"{condition}_win_rate": float(np.mean(gain > 0))})
    # Pair the scene before bootstrapping: independent resamples would discard
    # the deliberate matching of room, geometry, duration and reference power.
    contrast = ((same["bimamba", "SI_SNR"] - same["baseline", "SI_SNR"])
                - (original["bimamba", "SI_SNR"] - original["baseline", "SI_SNR"]))[mask]
    lo, hi = np.quantile(bootstrap_means(contrast, rng, reps), [.025, .975])
    result.update(gain_change_db=float(contrast.mean()),
                  gain_change_ci_low=float(lo), gain_change_ci_high=float(hi))
    return result


def analyze(args):
    if args.bootstrap < 1 or args.expected_count < 1:
        raise ValueError("Positive bootstrap and expected count required")
    if args.output.exists():
        raise FileExistsError(f"Never overwrite an analysis: {args.output}")
    lines = [line.split(maxsplit=1) for line in (args.test_dir / "wav.scp").read_text().splitlines() if line.strip()]
    keys = sorted(uid for uid, _ in lines)
    if len(keys) != args.expected_count or len(set(keys)) != len(keys):
        raise ValueError("Wrong number of test keys or duplicate keys")
    generation_sources = validate_generation(args.test_dir, keys)
    with args.metadata.open() as stream:
        meta_rows = list(csv.DictReader(stream))
    metadata = {r["utterance_id"]: r for r in meta_rows}
    if len(metadata) != len(meta_rows):
        raise ValueError("Duplicate room metadata")
    rt = np.array([float(metadata[canonical_uid(uid)]["T60"]) for uid in keys])
    if np.any(~np.isfinite(rt)) or np.any((rt < .1) | (rt > 1)):
        raise ValueError("Invalid RT60")
    original = load_pair(args.original_baseline, args.original_bimamba, keys)
    same = load_pair(args.baseline, args.bimamba, keys)
    rng = np.random.default_rng(args.seed)
    groups = [("All", np.ones(len(keys), bool))]
    for lo, hi in ((.1, .3), (.3, .6), (.6, 1.0)):
        groups.append((f"{lo:g}–{hi:g}", (rt >= lo) & ((rt <= hi) if hi == 1 else (rt < hi))))
    rows = [dict(rt60_s=label, **summarize(original, same, mask, rng, args.bootstrap))
            for label, mask in groups]
    paired = []
    for i, uid in enumerate(keys):
        row = dict(scene_uid=uid, rt60_s=float(rt[i]))
        for condition, data in (("original", original), ("same_speaker", same)):
            for metric in METRICS:
                for model in ("baseline", "bimamba"):
                    row[f"{condition}_{model}_{metric}"] = float(data[model, metric][i])
                row[f"{condition}_gain_{metric}"] = float(data["bimamba", metric][i] - data["baseline", metric][i])
        row["gain_change_db"] = row["same_speaker_gain_SI_SNR"] - row["original_gain_SI_SNR"]
        paired.append(row)
    args.output.mkdir(parents=True)
    write_csv(args.output / "summary.csv", rows)
    write_csv(args.output / "paired_scenes.csv", paired)
    sources = [args.metadata, *generation_sources, *(args.test_dir / n for n in ("wav.scp", "spk1.scp", "spk2.scp"))]
    for root in (args.baseline, args.bimamba, args.original_baseline, args.original_bimamba):
        sources.extend(root / f"{metric}_spk{i}" for metric in METRICS for i in (1, 2))
    provenance = dict(seed=args.seed, bootstrap=args.bootstrap,
                      unit="scene; mean of two already reference-aligned speaker scores",
                      source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      caveats=["Single training seed; intervals do not quantify training reproducibility.",
                               "Exploratory same-speaker stress test, not the standard NF-WHAMR benchmark.",
                               "New utterance content also changes; speaker identity is not isolated causally.",
                               "Original scene keys encode the OLD speaker pair; see dump assignment audit for new identities.",
                               "Speech utterances/speakers recur across scenes; bootstrap conditions on this fixed speech pool.",
                               "CTF is re-estimated from each new mixture, not transferred from original audio."])
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    report = ["# 同一話者Mixture：固定Baseline S / BiMamba FiLM+ALL S\n",
              "元testと同じ部屋・2音源位置・RIR・長さ・参照パワー比。片方の元発話を保持し、もう片方を同一人物の別発話に交換。PCM16ピーク超過時はMixtureと両参照を共通減衰（両モデルは推論前に分散正規化）。CTFは新Mixtureから再推定。学習のやり直しなし。\n",
              "Metric: ESPnet SI_SNR [dB]（平均除去SI-SDR）。各Mixture内の2音源平均。区間は左閉右開、最終区間のみ右閉。\n",
              "| RT60 [s] | N | 元test Baseline | 元test BiMamba | 元test差 | 同一話者 Baseline | 同一話者 BiMamba | 同一話者差 | 改善幅の変化 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f'| {r["rt60_s"]} | {r["n_mixtures"]} | {r["original_baseline_SI_SNR"]:.3f} | {r["original_bimamba_SI_SNR"]:.3f} | {r["original_gain_SI_SNR"]:+.3f} | {r["same_speaker_baseline_SI_SNR"]:.3f} | {r["same_speaker_bimamba_SI_SNR"]:.3f} | {r["same_speaker_gain_SI_SNR"]:+.3f} | {r["gain_change_db"]:+.3f} |')
    report.extend(["", f"Paired scene bootstrap: {args.bootstrap}回、seed={args.seed}。点ごとの95%区間（学習seed間のばらつきは含まない）。", ""])
    for r in rows:
        report.append(f'- {r["rt60_s"]}: 同一話者でのモデル差CI [{r["same_speaker_gain_ci_low"]:+.3f}, {r["same_speaker_gain_ci_high"]:+.3f}]、元testからの改善幅変化CI [{r["gain_change_ci_low"]:+.3f}, {r["gain_change_ci_high"]:+.3f}] dB。')
    report.extend(["", "元testと同一話者testの双方で同じモデルを使用。性能を見てモデル・seed・データ選択は変更していない。これは発話内容も変わる探索的なストレステストであり、話者同一性だけの因果効果やCTFが部屋情報だけを捉えた証明にはならない。", "", "全5指標（SI_SNR, SDR, SAR, SIR, STOI）はsummary.csv、個別結果はpaired_scenes.csvを参照。STOIの値はESPnet出力の百分率。"])
    (args.output / "RESULTS.md").write_text("\n".join(report) + "\n")
    (args.output / "report.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>同一話者Mixture評価</title><style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap}a{margin-right:20px}</style><h1>同一話者Mixture評価</h1><pre>' + html.escape("\n".join(report)) + '</pre><p><a href="summary.csv">全指標CSV</a><a href="paired_scenes.csv">scene別CSV</a></p></html>')
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "bimamba", "original-baseline", "original-bimamba", "test-dir", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--expected-count", type=int, default=3000)
    analyze(parser.parse_args())
