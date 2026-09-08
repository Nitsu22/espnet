#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score direct RT60 predictions against WHAMR generation values"
    )
    parser.add_argument("--prediction_scp", required=True)
    parser.add_argument("--room_param_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def read_scp(path: Path) -> Dict[str, str]:
    entries = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<uid> <value>'")
            uid, value = parts
            if uid in entries:
                raise ValueError(f"{path}:{line_number}: duplicate uid {uid}")
            entries[uid] = value
    if not entries:
        raise ValueError(f"No entries found in {path}")
    return entries


def main(cmd=None):
    args = get_parser().parse_args(cmd)
    prediction_scp = Path(args.prediction_scp)
    room_param_scp = Path(args.room_param_scp)
    output_dir = Path(args.output_dir)

    prediction_entries = read_scp(prediction_scp)
    room_param_entries = read_scp(room_param_scp)
    prediction_keys = set(prediction_entries)
    reference_keys = set(room_param_entries)
    if prediction_keys != reference_keys:
        missing_predictions = sorted(reference_keys - prediction_keys)
        missing_references = sorted(prediction_keys - reference_keys)
        raise ValueError(
            "Prediction/reference key mismatch: "
            f"missing_predictions={missing_predictions[:5]} "
            f"({len(missing_predictions)} total), "
            f"missing_references={missing_references[:5]} "
            f"({len(missing_references)} total)"
        )

    uids = sorted(prediction_entries)
    predictions = []
    references = []
    for uid in uids:
        prediction = float(prediction_entries[uid])
        room_param_path = room_param_entries[uid]
        with np.load(room_param_path, allow_pickle=False) as room_param_npz:
            if "T60" not in room_param_npz:
                raise KeyError(f"{uid}: T60 is missing from {room_param_path}")
            reference = float(np.asarray(room_param_npz["T60"]).item())
        if not math.isfinite(prediction) or prediction <= 0.0:
            raise ValueError(
                f"{uid}: predicted RT60 must be finite and positive, got {prediction}"
            )
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError(
                f"{uid}: reference RT60 must be finite and positive, got {reference}"
            )
        predictions.append(prediction)
        references.append(reference)

    prediction_array = np.asarray(predictions, dtype=np.float64)
    reference_array = np.asarray(references, dtype=np.float64)
    errors = prediction_array - reference_array
    mse = float(np.mean(errors**2))
    if len(uids) >= 2 and prediction_array.std() > 0 and reference_array.std() > 0:
        pearson = float(np.corrcoef(prediction_array, reference_array)[0, 1])
    else:
        pearson = None
    metrics = {
        "count": len(uids),
        "bias_sec": float(np.mean(errors)),
        "mae_sec": float(np.mean(np.abs(errors))),
        "mse_sec2": mse,
        "rmse_sec": math.sqrt(mse),
        "pearson": pearson,
        "prediction_mean_sec": float(np.mean(prediction_array)),
        "reference_mean_sec": float(np.mean(reference_array)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            value_text = "nan" if value is None else str(value)
            f.write(f"{key} {value_text}\n")
    with (output_dir / "per_utterance.tsv").open("w", encoding="utf-8") as f:
        f.write("uid\tprediction_sec\treference_sec\terror_sec\tabs_error_sec\n")
        for uid, prediction, reference, error in zip(
            uids, prediction_array, reference_array, errors
        ):
            f.write(
                f"{uid}\t{prediction:.10g}\t{reference:.10g}\t"
                f"{error:.10g}\t{abs(error):.10g}\n"
            )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
