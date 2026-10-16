#!/usr/bin/env python3
"""Paired, exploratory condition analysis of two-speaker separation scores.

No model selection or threshold optimization uses test outcomes. One bootstrap
unit is one mixture (both reference speakers together), not an isolated speaker.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from scipy import stats


FEATURES = {
    "rt60_s": ("Nominal RT60 [s]", [0.1, 0.3, 0.6, 1.0]),
    "volume_m3": ("Room volume [m³]", [0, 150, 225, np.inf]),
    "doa_separation_deg": ("Horizontal source-angle gap [deg]", [0, 30, 60, 120, 180]),
    "mean_distance_m": ("Mean source–left-mic distance [m]", [0, 1, 1.5, np.inf]),
    "distance_gap_m": ("Source-distance difference [m]", [0, 0.25, 0.5, np.inf]),
    "source_separation_m": ("Distance between sources [m]", [0, 1, 2, np.inf]),
    "power_imbalance_db": ("Clean-source power imbalance [dB]", [0, 2, 5, np.inf]),
    "duration_s": ("Mixture duration [s]", [0, 3, 5, np.inf]),
}
METRICS = ("SI_SNR", "SDR", "SAR", "SIR", "STOI")


def read_scores(path):
    out = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        uid, value = line.split()
        if uid in out:
            raise ValueError(f"Duplicate score key: {path}:{lineno}")
        out[uid] = float(value)
        if not np.isfinite(out[uid]):
            raise ValueError(f"Nonfinite score: {path}:{lineno}")
    return out


def canonical_uid(uid):
    fields = uid.split("_")
    if len(fields) != 7 or fields[-1] != "reverb":
        raise ValueError(f"Unexpected mixture UID: {uid}")
    if fields[0] != fields[2][:3] or fields[1] != fields[4][:3]:
        raise ValueError(f"Speaker prefix mismatch: {uid}")
    return "_".join(fields[2:-1]) + ".wav"


def geometry(row):
    vector = lambda prefix: np.array([float(row[f"{prefix}_{c}"]) for c in "xyz"])
    s1, s2 = vector("s1"), vector("s2")
    mic = np.array([float(row["micL_x"]), float(row["micL_y"]), float(row["mic_z"])])
    v1, v2 = s1 - mic, s2 - mic
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if min(d1, d2) <= 0 or min(np.linalg.norm(v1[:2]), np.linalg.norm(v2[:2])) <= 0:
        raise ValueError("Undefined source geometry")
    angle = np.degrees(np.arccos(np.clip(np.dot(v1[:2], v2[:2]) / (np.linalg.norm(v1[:2]) * np.linalg.norm(v2[:2])), -1, 1)))
    return {
        "rt60_s": float(row["T60"]),
        "volume_m3": float(np.prod(vector("room"))),
        "doa_separation_deg": float(angle),
        "mean_distance_m": float((d1 + d2) / 2),
        "distance_gap_m": float(abs(d1 - d2)),
        "source_separation_m": float(np.linalg.norm(s1 - s2)),
    }


def bin_index(values, edges):
    values = np.asarray(values)
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 2
    if np.any(indices < 0) or np.any(indices >= len(edges) - 1):
        raise ValueError("Feature lies outside declared bins")
    return indices


def bootstrap_means(values, rng, reps):
    values = np.asarray(values)
    if len(values) == 0:
        raise ValueError("Empty group")
    out = np.empty(reps)
    for first in range(0, reps, 100):
        last = min(first + 100, reps)
        out[first:last] = values[rng.integers(0, len(values), (last - first, len(values)))].mean(axis=1)
    return out


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    q = np.empty(len(order))
    q[order] = np.minimum.accumulate((pvalues[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, mask, rng, reps):
    subset = [r for r, keep in zip(rows, mask) if keep]
    gain = np.array([r["gain_SI_SNR"] for r in subset])
    ci = np.quantile(bootstrap_means(gain, rng, reps), [.025, .975])
    summary = {"n_mixtures": len(subset), "n_sources": 2 * len(subset)}
    for metric in METRICS:
        for model in ("baseline", "bimamba", "gain"):
            summary[f"{model}_{metric}"] = float(np.mean([r[f"{model}_{metric}"] for r in subset]))
    summary.update(gain_ci_low=float(ci[0]), gain_ci_high=float(ci[1]),
                   median_gain=float(np.median(gain)), win_rate=float(np.mean(gain > 0)),
                   trimmed_mean_gain_1pct=float(stats.trim_mean(gain, .01)),
                   gain_p10=float(np.quantile(gain, .1)), gain_p90=float(np.quantile(gain, .9)))
    return summary


def source_features(audio_root, uids):
    import soundfile as sf
    def load_scp(name):
        pairs = [line.split(maxsplit=1) for line in (audio_root / name).read_text().splitlines() if line.strip()]
        if len(pairs) != len(dict(pairs)):
            raise ValueError(f"Duplicate {name} key")
        data = dict(pairs)
        if set(data) != set(uids):
            raise ValueError(f"Audio/score keys differ: {name}")
        return data
    scps = [load_scp(f"spk{i}.scp") for i in (1, 2)]
    result = {}
    for uid in uids:
        powers, lengths = [], []
        for scp in scps:
            wav, fs = sf.read(scp[uid], dtype="float64", always_2d=True)
            if fs != 8000:
                raise ValueError(f"Expected native 8k input: {uid}")
            wav = wav[:, 0]
            powers.append(np.mean((wav - wav.mean()) ** 2))
            lengths.append(len(wav))
        if lengths[0] != lengths[1] or min(powers) <= 0:
            raise ValueError(f"Invalid references: {uid}")
        result[uid] = dict(power_imbalance_db=float(abs(10 * np.log10(powers[0] / powers[1]))), duration_s=lengths[0] / 8000)
    return result


def regression(rows):
    # Geometry and source-power factors considered jointly; no baseline-score
    # predictor, because regressing a difference on its baseline induces coupling.
    names = ["rt60_s", "log_volume", "doa_separation_deg", "mean_distance_m", "distance_gap_m", "power_imbalance_db", "duration_s"]
    raw = np.array([[r["rt60_s"], np.log(r["volume_m3"]), r["doa_separation_deg"], r["mean_distance_m"], r["distance_gap_m"], r["power_imbalance_db"], r["duration_s"]] for r in rows])
    x = np.column_stack([np.ones(len(rows)), (raw - raw.mean(axis=0)) / raw.std(axis=0)])
    y = np.array([r["gain_SI_SNR"] for r in rows])
    xtx = np.linalg.inv(x.T @ x)
    beta = xtx @ x.T @ y
    resid = y - x @ beta
    leverage = np.sum((x @ xtx) * x, axis=1)
    adjusted = resid / (1 - leverage)
    cov = xtx @ ((x * adjusted[:, None]).T @ (x * adjusted[:, None])) @ xtx
    se = np.sqrt(np.diag(cov))
    p = 2 * stats.norm.sf(np.abs(beta[1:] / se[1:]))
    q = bh_adjust(p)
    return [{"feature": name, "coefficient_db_per_sd": float(beta[i]), "hc3_se": float(se[i]),
             "ci_low": float(beta[i] - 1.96 * se[i]), "ci_high": float(beta[i] + 1.96 * se[i]),
             "p_exploratory": float(p[i-1]), "q_bh": float(q[i-1])} for i, name in enumerate(names, 1)], float(1 - (resid @ resid) / np.sum((y-y.mean())**2))


def analyze(args):
    rng = np.random.default_rng(args.seed)
    meta_rows = list(csv.DictReader(args.metadata.open()))
    metadata = {r["utterance_id"]: r for r in meta_rows}
    if len(metadata) != len(meta_rows):
        raise ValueError("Duplicate metadata key")
    data = {}
    keys = None
    sources = [args.metadata, args.audio_root / "spk1.scp", args.audio_root / "spk2.scp"]
    for model, root in (("baseline", args.baseline), ("bimamba", args.bimamba)):
        for metric in METRICS:
            for speaker in (1, 2):
                path = root / f"{metric}_spk{speaker}"
                values = read_scores(path)
                if keys is None:
                    keys = set(values)
                if set(values) != keys:
                    raise ValueError(f"Score key mismatch: {path}")
                data[model, metric, speaker] = values
                sources.append(path)
    if len(keys) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} mixtures, got {len(keys)}")
    audio = source_features(args.audio_root, keys)
    rows = []
    for uid in sorted(keys):
        key = canonical_uid(uid)
        if key not in metadata:
            raise ValueError(f"Missing metadata: {uid}")
        row = dict(uid=uid, metadata_key=key, **geometry(metadata[key]), **audio[uid])
        for metric in METRICS:
            for model in ("baseline", "bimamba"):
                row[f"{model}_{metric}"] = (data[model, metric, 1][uid] + data[model, metric, 2][uid]) / 2
            row[f"gain_{metric}"] = row[f"bimamba_{metric}"] - row[f"baseline_{metric}"]
        rows.append(row)
    summaries = [dict(feature="all", bin="All", **summarize(rows, np.ones(len(rows), dtype=bool), rng, args.bootstrap))]
    for feature, (_, edges) in FEATURES.items():
        idx = bin_index([r[feature] for r in rows], edges)
        for i in range(len(edges) - 1):
            label = f"[{edges[i]:g}, {edges[i+1]:g}{']' if i == len(edges)-2 else ')'}"
            summaries.append(dict(feature=feature, bin=label, **summarize(rows, idx == i, rng, args.bootstrap)))
    joint = []
    rt = bin_index([r["rt60_s"] for r in rows], FEATURES["rt60_s"][1])
    volume = bin_index([r["volume_m3"] for r in rows], FEATURES["volume_m3"][1])
    for i in range(3):
        for j in range(3):
            joint.append(dict(rt60_bin=i, volume_bin=j, **summarize(rows, (rt == i) & (volume == j), rng, args.bootstrap)))
    reg, r2 = regression(rows)
    delta = np.array([r["gain_SI_SNR"] for r in rows])
    contrast = bootstrap_means(delta[rt == 2], rng, args.bootstrap) - bootstrap_means(delta[rt == 0], rng, args.bootstrap)
    contrast_ci = np.quantile(contrast, [.025, .975])
    write_contrast = [{"contrast": "high_rt60_minus_low_rt60_gain", "gain_difference_db": float(delta[rt == 2].mean()-delta[rt == 0].mean()),
                       "ci_low": float(contrast_ci[0]), "ci_high": float(contrast_ci[1])}]
    correlations = []
    for feature in FEATURES:
        rho, p = stats.spearmanr([r[feature] for r in rows], [r["gain_SI_SNR"] for r in rows])
        correlations.append(dict(feature=feature, spearman=float(rho), p_exploratory=float(p)))
    for row, q in zip(correlations, bh_adjust([r["p_exploratory"] for r in correlations])):
        row["q_bh"] = float(q)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "paired_utterances.csv", rows)
    write_csv(args.output / "condition_summary.csv", summaries)
    write_csv(args.output / "rt60_volume.csv", joint)
    write_csv(args.output / "adjusted_regression.csv", reg)
    write_csv(args.output / "exploratory_correlations.csv", correlations)
    write_csv(args.output / "rt60_contrast.csv", write_contrast)
    provenance = dict(seed=args.seed, bootstrap_repetitions=args.bootstrap, n_mixtures=len(rows),
                      unit="mixture, mean over two reference speakers", regression_r_squared=r2,
                      source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      boundaries={name: [None if np.isinf(x) else x for x in edges] for name, (_, edges) in FEATURES.items()},
                      caveats=["Single training seed; CIs resample test mixtures, not model seeds.",
                               "Nominal generation T60, not independently measured decay.",
                               "DOA is geometric horizontal source-angle gap at left mic; this is mono, not evidence of spatial cue use.",
                               "All subgroup/interaction analyses are exploratory; no causal claims or test-driven model selection.",
                               "Individual subgroup CIs are pointwise, not familywise simultaneous.",
                               "Power imbalance uses zero-mean original clean references, left channel, not nominal filename gains."])
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot(args.output, summaries, joint)
    make_report(args.output, summaries, reg, r2, write_contrast[0], args.bootstrap, args.seed)
    print(json.dumps({"overall": summaries[0], "adjusted_r2": r2, "regression": reg}, indent=2))


def plot(output, summaries, joint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = ["rt60_s", "volume_m3", "doa_separation_deg", "mean_distance_m", "distance_gap_m", "power_imbalance_db"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    overall = summaries[0]["gain_SI_SNR"]
    for ax, feature in zip(axes.flat, chosen):
        groups = [r for r in summaries if r["feature"] == feature]
        y = np.array([r["gain_SI_SNR"] for r in groups])
        errors = np.array([[r["gain_ci_low"] for r in groups], [r["gain_ci_high"] for r in groups]])
        ax.errorbar(range(len(groups)), y, yerr=np.abs(errors - y), fmt="o", capsize=5, color="#116b8b", linewidth=2)
        ax.axhline(0, color="black", lw=.8)
        ax.axhline(overall, color="#999999", ls="--", lw=1)
        ax.set_xticks(range(len(groups)), [f'{r["bin"]}\nn={r["n_mixtures"]}' for r in groups], fontsize=9)
        ax.set_title(FEATURES[feature][0])
        ax.set_ylabel("SI-SNR gain over Baseline [dB]")
        ax.set_ylim(-0.75, 0.65)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("BiMamba CTF + FiLM ALL S vs Baseline S | NF-WHAMR 8 kHz test", fontsize=15)
    fig.text(.5, .015, "Paired mixture bootstrap, pointwise 95% CIs; dashed line = overall gain. Exploratory, one training seed.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(output / "condition_gains.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.3, 5))
    grid = np.array([r["gain_SI_SNR"] for r in joint]).reshape(3, 3)
    limit = max(abs(grid.min()), abs(grid.max()))
    im = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit)
    for row in joint:
        ax.text(row["volume_bin"], row["rt60_bin"], f'{row["gain_SI_SNR"]:+.3f} dB\nn={row["n_mixtures"]}', ha="center", va="center", color="black", bbox=dict(facecolor="white", alpha=.65, edgecolor="none"))
    ax.set_xticks(range(3), ["<150", "150–225", "≥225"])
    ax.set_yticks(range(3), ["0.1–0.3", "0.3–0.6", "0.6–1.0"])
    ax.set_xlabel("Room volume [m³]")
    ax.set_ylabel("Nominal RT60 [s]")
    ax.set_title("Joint RT60 × room-volume strata (exploratory)")
    fig.colorbar(im, ax=ax, label="SI-SNR gain [dB]")
    fig.tight_layout()
    fig.savefig(output / "rt60_volume_gains.png", dpi=180)
    plt.close(fig)


def make_report(output, summaries, reg, r2, contrast, bootstrap=5000, seed=0):
    def table(items):
        header = '<tr><th>区間</th><th>混合音数</th><th>Baseline</th><th>BiMamba</th><th>差 [dB]</th><th>差の95% CI</th><th>改善率</th><th>上下1%除外の差</th></tr>'
        lines = []
        for r in items:
            lines.append(f'<tr><td>{html.escape(r["bin"])}</td><td>{r["n_mixtures"]:,}</td><td>{r["baseline_SI_SNR"]:.3f}</td><td>{r["bimamba_SI_SNR"]:.3f}</td><td>{r["gain_SI_SNR"]:+.3f}</td><td>[{r["gain_ci_low"]:+.3f}, {r["gain_ci_high"]:+.3f}]</td><td>{100*r["win_rate"]:.1f}%</td><td>{r["trimmed_mean_gain_1pct"]:+.3f}</td></tr>')
        return '<table>' + header + ''.join(lines) + '</table>'
    parts = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>NF-WHAMR条件別比較</title><style>body{font:16px/1.65 sans-serif;max-width:1120px;margin:36px auto;padding:0 20px;color:#172535}h1,h2{color:#12506a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #d9e1e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf4f7}img{width:100%}.note{background:#edf4f7;padding:16px}</style>',
             '<h1>Baseline S vs BiMamba推定CTF・FiLM+ALL S：条件別評価</h1>',
             '<p>NF-WHAMR!・8 kHz・1ch・test 3,000混合音／6,000音源。モデル・seedは各1条件。指標はESPnetのSI_SNR（平均除去SI-SDRと同じ定義）。値は高いほど良い。</p>',
             '<p class="note"><b>主な発見：</b>低残響ではSI-SNRが悪化する一方、中～高残響では改善する。部屋体積が小さい群で改善が比較的大きい。DOA差・距離差では明確な連続的傾向は得られなかった。単一seedの探索的解析であり、因果的な説明・推定器精度の証明ではない。</p>',
             '<h2>全体</h2>', table(summaries[:1]),
             '<h2>条件による改善量</h2><img src="condition_gains.png" alt="条件別のpaired改善量と95%区間">']
    for feature, (label, _) in FEATURES.items():
        parts.extend([f'<h2>{html.escape(label)}</h2>', table([r for r in summaries if r['feature'] == feature])])
    parts.extend(['<h2>RT60と部屋体積を同時に分けた確認</h2><img src="rt60_volume_gains.png" alt="RT60と部屋体積の交差集計">',
                  '<p>高RT60群と低RT60群の改善量の差：' + f'{contrast["gain_difference_db"]:+.3f} dB、95% CI [{contrast["ci_low"]:+.3f}, {contrast["ci_high"]:+.3f}]。' + '各RT60群で上下1%の極端値を除いても、低残響で悪化・高残響で改善という符号は維持された。</p>',
                  '<h2>他条件を同時に考慮した探索的回帰</h2><p>目的変数は各混合音のSI-SNR差。RT60、log部屋体積、DOA差、平均距離、距離差、実測クリーン音量差、長さを標準化して同時投入。HC3頑健標準誤差、係数単位は1標準偏差当たりdB。7項目内のp値をBH補正。' + f'R²={r2:.3f}であり、個別utteranceの差の大部分はこのメタデータだけでは説明できない。</p>',
                  '<table><tr><th>条件</th><th>係数</th><th>95% CI</th><th>BH q</th></tr>'])
    for r in reg:
        parts.append(f'<tr><td>{html.escape(r["feature"])}</td><td>{r["coefficient_db_per_sd"]:+.3f}</td><td>[{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]</td><td>{r["q_bh"]:.3g}</td></tr>')
    parts.extend(['</table><h2>定義・限界</h2><ul>',
                  '<li>RT60はCSVの生成指定T60。実測された各音源RIRのRT60ではない。</li>',
                  '<li>部屋体積はroom_x×room_y×room_z。距離は左micから音源への3次元距離。DOA差は左mic基準の水平音源角度差（0–180°）。monoモデルのため、角度別結果は空間配列手掛かり利用の証拠ではない。</li>',
                  f'<li>2話者を各utterance内で平均してモデル差を計算。{bootstrap:,}回、seed{seed}のpaired mixture bootstrap。区間は各群のpointwise 95%で、多重比較全体の同時区間ではない。学習seed間の不確実性は含まない。</li>',
                  '<li>区間は左閉右開、最終区間のみ右閉。thresholdは物理量の区切りであり、test性能を最大化する探索はしていない。全8条件を省略せず報告。</li>',
                  '<li>参照話者に整列済みの保存スコアを使用。キー集合・重複・有限値・メタデータ一意対応を全件検証。元の片側mic・clean参照から音量差と長さを算出。</li>',
                  '<li>RT60・体積を同時に考慮しても傾向は残るが、CTF推定誤差・反射構造が原因かは未検証。次の確認は別seedとOracle/Random条件、同一音声でRT60のみ変える制御実験。</li>',
                  '</ul><p><a href="condition_summary.csv">全条件集計CSV</a> / <a href="paired_utterances.csv">utterance別CSV</a> / <a href="adjusted_regression.csv">回帰CSV</a> / <a href="provenance.json">出所と検証情報</a></p></html>'])
    (output / 'report.html').write_text('\n'.join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "bimamba", "metadata", "audio-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--expected-count", type=int, default=3000)
    analyze(parser.parse_args())
