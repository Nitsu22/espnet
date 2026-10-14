#!/usr/bin/env python3
"""Score existing ordinary-PIM predictions against a shared LS RIR reference."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import soundfile as sf
from prepare_ls_rir_reference import digest, validate_dump


def prediction_index(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "rir.scp"
    entries = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[0] in entries:
            raise ValueError(f"Invalid/duplicate prediction row: {path}")
        entries[parts[0]] = tuple(Path(p).resolve() for p in parts[1:])
    return path, entries


def evaluate(reference_dir, predictions, output, allow_limited=False,
             write_comparison_csv=True):
    reference_dir, output = Path(reference_dir).resolve(), Path(output).resolve()
    reference = validate_dump(reference_dir, allow_limited=allow_limited)
    config = reference["config"]
    ids = {
        line.split()[0]
        for line in (reference_dir / "reference.scp").read_text().splitlines()
    }
    if (
        output.exists()
        or not predictions
        or len({name for name, _ in predictions}) != len(predictions)
    ):
        raise ValueError(
            "Use a new result directory and distinct, nonempty model labels"
        )
    prepared = []
    for name, source in predictions:
        path, entries = prediction_index(source)
        if set(entries) != ids and not (allow_limited and ids <= set(entries)):
            raise ValueError(f"Prediction/reference IDs differ for {name}")
        for uid in ids:
            for audio in entries[uid]:
                info = sf.info(audio)
                if (
                    info.samplerate != config["sample_rate"]
                    or info.frames != config["rir_length"]
                    or info.channels != 1
                    or info.subtype != "FLOAT"
                ):
                    raise ValueError(
                        f"Expected matching mono FLOAT ordinary-PIM RIR: {audio}"
                    )
        metadata_path = path.parent.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        )
        if metadata is not None:
            if (
                not metadata.get("inference", "").startswith("ordinary sweep PIM")
                or metadata.get("sample_rate") != config["sample_rate"]
                or metadata.get("rir_length") != config["rir_length"]
            ):
                raise ValueError(f"Prediction measurement metadata differs for {name}")
        prepared.append((name, path, metadata))
    output.mkdir(parents=True)
    report = dict(
        reference_type="ls_direct_reverb",
        reference_dir=str(reference_dir),
        reference_preparation_sha256=digest(reference_dir / "preparation.json"),
        reference_config=config,
        count=reference["count"],
        source_pairs=2 * reference["count"],
        full_dataset=reference["full_dataset"],
        inference="ordinary sweep PIM",
        align="peak",
        scale_mode="peak",
        pit_metric="rmse",
        polarity_align=False,
        interpretation=(
            "relative-transfer waveform shape; absolute gain/delay are not scored"
        ),
        acoustic_metrics=(
            "relative-transfer diagnostics; not physical clean-reverb metrics"
        ),
        models={},
    )
    scorer = Path(__file__).with_name("score_rir_pit.py")
    for number, (name, path, metadata) in enumerate(prepared):
        folder = output / "models" / f"{number:02d}"
        print(f"LS direct-reverb scoring: {name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(scorer),
                "--ref_rir_scp",
                str(reference_dir / "reference.scp"),
                "--pred_rir_scp",
                str(path),
                "--out_dir",
                str(folder),
                "--sample_rate",
                str(config["sample_rate"]),
                "--rir_length",
                str(config["rir_length"]),
                "--align",
                "peak",
                "--scale_mode",
                "peak",
                "--pit_metric",
                "rmse",
            ],
            check=True,
        )
        summary = json.loads((folder / "summary.json").read_text())
        if (
            summary["num_utts"] != reference["count"]
            or summary["num_source_pairs"] != 2 * reference["count"]
        ):
            raise ValueError("Incomplete LS evaluation")
        with (folder / "per_source.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        valid = {
            key: sum(math.isfinite(float(row[key])) for row in rows)
            for key in ("rmse", "corr", "rt60_err", "drr_err", "c50_err")
        }
        if valid["rmse"] != len(rows) or valid["corr"] != len(rows):
            raise ValueError("Nonfinite primary waveform metrics")
        model = dict(
            summary=str(folder / "summary.json"),
            metrics=summary,
            valid_counts=valid,
            prediction_scp=str(path),
            prediction_index_sha256=digest(path),
            prediction_metadata=metadata,
        )
        (folder / "ls_reference_metadata.json").write_text(
            json.dumps(
                dict(
                    reference_type=report["reference_type"],
                    reference_preparation_sha256=report["reference_preparation_sha256"],
                    valid_counts=valid,
                ),
                indent=2,
            )
            + "\n"
        )
        report["models"][name] = model
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = ("rmse_mean", "corr_mean", "rt60_mae", "drr_mae", "c50_mae")
    if write_comparison_csv:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["model"] + list(keys))
            writer.writeheader()
            writer.writerows(
                dict(model=name, **{key: model["metrics"][key] for key in keys})
                for name, model in report["models"].items()
            )
    (output / "evaluation_complete.json").write_text(
        json.dumps(
            dict(
                count=reference["count"],
                models=list(report["models"]),
                comparison_sha256=digest(output / "comparison.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        nargs=2,
        action="append",
        required=True,
        metavar=("MODEL_LABEL", "PREDICTION_SCP_OR_DIR"),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--no_comparison_csv", action="store_true",
                        help="Keep JSON reports and per-source diagnostics only")
    parser.add_argument(
        "--allow_limited",
        action="store_true",
        help="Explicit smoke-test/subset scoring",
    )
    args = parser.parse_args()
    report = evaluate(
        args.reference_dir, args.prediction, args.out_dir, args.allow_limited,
        write_comparison_csv=not args.no_comparison_csv
    )
    print(
        json.dumps(
            {
                name: {key: model["metrics"][key] for key in ("rmse_mean", "corr_mean")}
                for name, model in report["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
