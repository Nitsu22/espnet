#!/usr/bin/env python3
"""Build a separate, immutable LS direct-reverb evaluation reference dump."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import soundfile as sf

from espnet2.rir.ls_rir import LSRIRConfig, LSRIRFitter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scp(path):
    entries = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] in entries:
            raise ValueError(f"Invalid/duplicate SCP entry in {path}: {line}")
        uid, value = parts
        if Path(uid).name != uid or uid in (".", ".."):
            raise ValueError(f"Unsafe utterance ID: {uid}")
        entries[uid] = str(Path(value).resolve())
    if not entries:
        raise ValueError(f"Empty SCP: {path}")
    return entries


def load_physical(path, sample_rate, channel):
    payload = Path(path).read_bytes()
    audio, actual_rate = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    if actual_rate != sample_rate or not 0 <= channel < audio.shape[1]:
        raise ValueError(f"Sample rate/channel mismatch: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite physical RIR: {path}")
    return audio[:, channel], dict(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        frames=len(audio),
        channels=audio.shape[1],
    )


def ordered_parallel(function, items, workers):
    """Bound pending work, including when a fit fails."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is not None:
                    pending.append(executor.submit(function, item))
            while pending:
                yield pending.popleft().result()
                item = next(iterator, None)
                if item is not None:
                    pending.append(executor.submit(function, item))
        finally:
            for future in pending:
                future.cancel()


def validate_dump(output, allow_limited=False):
    output = Path(output).resolve()
    metadata = json.loads((output / "preparation.json").read_text())
    if metadata["schema"] != 1 or metadata["reference_type"] != "ls_direct_reverb":
        raise ValueError("Expected an LS direct-reverb reference dump")
    if digest(output / "ls_rir_source.py") != metadata["implementation_sha256"]:
        raise ValueError("Reference-fitting source snapshot checksum differs")
    if not metadata["full_dataset"] and not allow_limited:
        raise ValueError("Limited/smoke reference dump is not a full evaluation set")
    if digest(output / "records.jsonl") != metadata["records_sha256"]:
        raise ValueError("Reference record checksum differs")
    for name, checksum in metadata["index_sha256"].items():
        if digest(output / name) != checksum:
            raise ValueError(f"Reference index checksum differs: {name}")
    refs = [read_scp(output / f"rir_ref_ls{s}.scp") for s in (1, 2)]
    filters = [read_scp(output / f"g_ls{s}.scp") for s in (1, 2)]
    records = [
        json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()
    ]
    ids = [record["uid"] for record in records]
    if len(set(ids)) != len(ids) or len(ids) != metadata["count"]:
        raise ValueError("Invalid LS reference count/IDs")
    if any(set(entries) != set(ids) for entries in refs + filters):
        raise ValueError("LS reference ID sets differ")
    reference_rows = [
        line.split() for line in (output / "reference.scp").read_text().splitlines()
    ]
    if reference_rows != [[uid, refs[0][uid], refs[1][uid]] for uid in ids]:
        raise ValueError("Combined reference SCP differs")
    for record in records:
        for speaker, pair in enumerate(record["speakers"]):
            for name, entries, length in (
                ("filter", filters, metadata["config"]["filter_length"]),
                ("reference", refs, metadata["config"]["rir_length"]),
            ):
                file_record = pair[name]
                path = output / file_record["relative_path"]
                if Path(entries[speaker][record["uid"]]).resolve() != path:
                    raise ValueError("Reference file/index path differs")
                if digest(path) != file_record["sha256"]:
                    raise ValueError(f"Reference WAV checksum differs: {path}")
                info = sf.info(path)
                if (
                    info.samplerate != metadata["config"]["sample_rate"]
                    or info.frames != length
                    or info.channels != 1
                    or info.subtype != "FLOAT"
                ):
                    raise ValueError(f"Reference WAV header differs: {path}")
    return metadata


def prepare(data, output, config=LSRIRConfig(), channel=0, limit=0, workers=2):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or output == data or data in output.parents:
        raise ValueError("Use a new evaluation-only directory outside the source dump")
    if workers < 1 or limit < 0 or channel < 0:
        raise ValueError("Invalid workers, limit or channel")
    names = [f"rir_{kind}{s}.scp" for kind in ("direct", "ref") for s in (1, 2)]
    indices = {name: read_scp(data / name) for name in names}
    ids = list(indices[names[0]])
    if any(set(value) != set(ids) for value in indices.values()):
        raise ValueError("Physical direct/reverb SCP ID sets differ")
    if (data / "wav.scp").exists():
        if set(read_scp(data / "wav.scp")) != set(ids):
            raise ValueError("Mixture/physical RIR SCP ID sets differ")
        names.append("wav.scp")
    source_hashes = {name: digest(data / name) for name in names}
    selected = ids[: limit or None]
    implementation_source = (
        Path(__file__).resolve().parents[4] / "espnet2/rir/ls_rir.py"
    ).read_bytes()
    fitter = LSRIRFitter(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent)
    )
    (partial / "ls_rir_source.py").write_bytes(implementation_source)
    for name in ("filters1", "filters2", "rir1", "rir2"):
        (partial / name).mkdir()

    def fit_one(uid):
        pairs = []
        for speaker in (1, 2):
            direct, direct_info = load_physical(
                indices[f"rir_direct{speaker}.scp"][uid], config.sample_rate, channel
            )
            reverb, reverb_info = load_physical(
                indices[f"rir_ref{speaker}.scp"][uid], config.sample_rate, channel
            )
            coefficients, reference, metrics = fitter.fit(direct, reverb)
            pair = dict(direct=direct_info, reverb=reverb_info, metrics=metrics)
            for name, folder, value in (
                ("filter", f"filters{speaker}", coefficients),
                ("reference", f"rir{speaker}", reference),
            ):
                relative = f"{folder}/{uid}.wav"
                sf.write(partial / relative, value, config.sample_rate, subtype="FLOAT")
                pair[name] = dict(
                    relative_path=relative, sha256=digest(partial / relative)
                )
            pairs.append(pair)
        return dict(uid=uid, speakers=pairs)

    try:
        records = []
        with (partial / "records.jsonl").open("w") as stream:
            for record in ordered_parallel(fit_one, selected, workers):
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                if len(records) % 100 == 0 or len(records) == len(selected):
                    print(
                        f"LS references: {len(records)}/{len(selected)} mixtures",
                        flush=True,
                    )
        if {name: digest(data / name) for name in names} != source_hashes:
            raise ValueError("Source indices changed during preparation")
        for speaker in (1, 2):
            for name, folder in (
                (f"g_ls{speaker}.scp", f"filters{speaker}"),
                (f"rir_ref_ls{speaker}.scp", f"rir{speaker}"),
            ):
                (partial / name).write_text(
                    "".join(
                        f"{uid} {output / folder / (uid + '.wav')}\n"
                        for uid in selected
                    )
                )
        (partial / "reference.scp").write_text(
            "".join(
                f"{uid} {output / 'rir1' / (uid+'.wav')} "
                f"{output / 'rir2' / (uid+'.wav')}\n"
                for uid in selected
            )
        )
        rows = [
            dict(uid=record["uid"], speaker=speaker, **pair["metrics"])
            for record in records
            for speaker, pair in enumerate(record["speakers"], 1)
        ]
        with (partial / "fit_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        diagnostics = {}
        for metric in (
            "iterations",
            "normal_relative_residual",
            "relative_response_mse",
            "relative_physical_pair_mse",
        ):
            values = [row[metric] for row in rows]
            diagnostics[metric] = dict(
                mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95)),
                maximum=float(np.max(values)),
            )
        index_names = [
            f"{prefix}{s}.scp" for prefix in ("g_ls", "rir_ref_ls") for s in (1, 2)
        ] + ["reference.scp", "fit_metrics.csv"]
        metadata = dict(
            schema=1,
            reference_type="ls_direct_reverb",
            config=config.as_dict(),
            source_data=str(data),
            source_index_sha256=source_hashes,
            channel=channel,
            count=len(selected),
            source_count=len(ids),
            full_dataset=len(selected) == len(ids),
            objective=(
                "||full_conv(sweep*h_direct,g)-sweep*h_reverb||^2 "
                "+ alpha*||x||^2*||g||^2"
            ),
            support=(
                "causal FIR, lag zero through filter_length-1; full convolution tails"
            ),
            normalization=(
                "physical pair divided by SAME direct peak, "
                "retaining relative timing/gain"
            ),
            measurement=(
                "ordinary sweep*g, ORIGINAL PIM inverse, peak-minus-2.5-ms crop"
            ),
            evaluation=dict(
                align="peak", scale_mode="peak", pit_metric="rmse", polarity_align=False
            ),
            limitation=(
                "LS-dependent reference; finite CTF approximation is not removed"
            ),
            sweep_sha256=hashlib.sha256(fitter.sweep.tobytes()).hexdigest(),
            inverse_sha256=hashlib.sha256(fitter.inverse.tobytes()).hexdigest(),
            implementation_sha256=hashlib.sha256(implementation_source).hexdigest(),
            diagnostics=diagnostics,
            records_sha256=digest(partial / "records.jsonl"),
            index_sha256={name: digest(partial / name) for name in index_names},
        )
        (partial / "preparation.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        # All WAVs were written/checksummed before indexes are published. Rename
        # makes only a complete reference directory available to the scorer.
        partial.rename(output)
        try:
            validate_dump(output, allow_limited=True)
        except Exception:
            output.rename(partial)
            raise
        return metadata
    finally:
        if partial.exists():
            shutil.rmtree(partial)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("prepare", "validate"), default="prepare")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow_limited", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    for field in fields(LSRIRConfig):
        parser.add_argument(
            f"--{field.name}", type=type(field.default), default=field.default
        )
    args = parser.parse_args()
    if args.mode == "validate":
        report = validate_dump(args.output, allow_limited=args.allow_limited)
    else:
        if args.data is None:
            parser.error("--data is required for preparation")
        config = LSRIRConfig(
            **{field.name: getattr(args, field.name) for field in fields(LSRIRConfig)}
        )
        report = prepare(
            args.data, args.output, config, args.channel, args.limit, args.workers
        )
    print(
        json.dumps(
            dict(
                count=report["count"],
                full_dataset=report["full_dataset"],
                diagnostics=report["diagnostics"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
