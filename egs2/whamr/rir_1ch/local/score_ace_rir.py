"""Score peak-aligned RIRs on a fixed two-second window at 16 kHz.

This is a WHAMR-ACE protocol, not a reproduction of the SimACE scorer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from score_rir import c50_db, drr_db, pearson, read_2col, rt60_schroeder


def aligned(x, sr=16000):
    """Align before cropping: retain 2.5 ms before the absolute peak."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.any(x):
        raise ValueError("Expected a finite nonzero mono RIR")
    peak = int(np.argmax(np.abs(x)))
    pre = round(sr * 0.0025)
    start = peak - pre
    result = np.zeros(sr * 2)
    source_start = max(0, start)
    dest_start = max(0, -start)
    count = min(len(x) - source_start, len(result) - dest_start)
    result[dest_start:dest_start + count] = x[source_start:source_start + count]
    return result / np.max(np.abs(result))


def metrics(pred, ref, sr=16000):
    pred, ref = aligned(pred, sr), aligned(ref, sr)
    pre = round(sr * 0.0025)
    early = slice(pre, pre + round(sr * 0.05))
    result = {
        "rmse_peak_to_50ms": float(np.sqrt(np.mean((pred[early] - ref[early]) ** 2))),
        "correlation_peak_to_50ms": pearson(pred[early], ref[early]),
    }
    for name, fn in (
        ("rt60_schroeder_s", lambda x: rt60_schroeder(x[pre:], sr)),
        ("drr_db", lambda x: drr_db(x, sr, 2.5)),
        ("c50_db", lambda x: c50_db(x, sr)),
    ):
        result[name + "_pred"] = fn(pred)
        result[name + "_ref"] = fn(ref)
        result[name + "_error"] = result[name + "_pred"] - result[name + "_ref"]
    return result


def summarize(rows):
    result = {"utterances": len(rows)}
    keys = [k for k in rows[0] if k.endswith("_error") or k.startswith(("rmse_", "correlation_"))]
    for key in keys:
        values = np.asarray([r[key] for r in rows])
        values = values[np.isfinite(values)]
        result[key] = {
            "valid": len(values), "invalid": len(rows) - len(values),
            "mean_absolute" if key.endswith("_error") else "mean":
                float(np.mean(np.abs(values) if key.endswith("_error") else values)) if len(values) else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-scp", type=Path, required=True)
    parser.add_argument("--ref-scp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()
    preds, refs = read_2col(args.pred_scp), read_2col(args.ref_scp)
    if not preds or preds.keys() != refs.keys():
        raise ValueError("Prediction and reference IDs must match exactly and be nonempty")
    rows = []
    for uid in sorted(preds):
        pred, psr = sf.read(preds[uid])
        ref, rsr = sf.read(refs[uid])
        if args.channel < 0:
            raise ValueError('channel must be nonnegative')
        if pred.ndim == 2:
            pred = pred[:, args.channel]
        if ref.ndim == 2:
            ref = ref[:, args.channel]
        if psr != 16000 or rsr != 16000:
            raise ValueError(f"{uid}: expected 16 kHz")
        rows.append({"uid": uid, "rir": Path(refs[uid]).stem, **metrics(pred, ref)})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # JSON null explicitly records undefined metrics, rather than nonstandard NaN.
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
              for k, v in row.items()} for row in rows]
    (args.output_dir / "utterances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in clean))
    per_rir = {name: summarize([r for r in rows if r["rir"] == name]) for name in sorted({r["rir"] for r in rows})}
    summary = {"protocol": {
        "sample_rate": 16000, "window_samples": 32000, "pre_peak_samples": 40,
        "alignment": "absolute peak before cropping; zero-pad unavailable samples",
        "amplitude": "independent absolute peak normalization; polarity preserved",
        "rt60": "Schroeder regression -5 to -35 dB; fallback -5 to -25 dB; extrapolate to -60 dB",
        "drr": "energy inside absolute peak +/-2.5 ms versus remaining window",
        "c50": "energy from peak to +50 ms versus remaining tail",
        "reference_tail": "cropped to the same aligned 2-second window as prediction",
        "scope": "WHAMR-ACE evaluation, not official SimACE scores",
    }, "micro": summarize(rows), "per_rir": per_rir}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary["micro"], indent=2))


if __name__ == "__main__":
    main()
