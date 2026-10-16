#!/usr/bin/env python3
"""Compare Rec-RIR speech heads with the saved full DAMSEP speech evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

METRICS = ("SI_SNR", "SDR", "SIR", "SAR", "STOI")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index(path):
    rows = [line.split(maxsplit=1) for line in Path(path).read_text().splitlines()]
    if not rows or len({uid for uid, _ in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate index: {path}")
    return dict(rows)


def summary(folder, ids):
    result = {}
    for metric in METRICS:
        values = []
        for spk in (1, 2):
            entries = index(folder / f"{metric}_spk{spk}")
            if set(entries) != set(ids):
                raise ValueError(f"Incomplete scores: {folder} {metric} {spk}")
            numbers = [float(entries[uid]) for uid in ids]
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError(f"Nonfinite scores: {folder} {metric}")
            values.extend(numbers)
        result[metric] = sum(values) / len(values)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use between one and eight CPU scoring workers")
    workspace, output = args.workspace.resolve(), args.output.resolve()
    recipe = workspace / "egs2/whamr/damsep_clean"
    test = recipe / "dump_clean/raw/tt_mix_clean_reverb_min_8k"
    reverb = recipe / "dump_reverb/raw/tt_mix_both_reverb_min_8k"
    source = workspace / "egs2/whamr/rir/exp/rir_train_rec_rir_2spk_pit_clean_8192"
    if (source / "valid.loss.best.pth").resolve().name != "97epoch.pth":
        raise ValueError("Expected Rec-RIR best97")
    mixtures = index(test / "wav.scp")
    ids = list(mixtures)
    if len(ids) != 3000:
        raise ValueError("Expected all 3000 NF-WHAMR test mixtures")
    destination = output / "rec_rir"
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    for name in ("config.yaml", "valid.loss.best.pth"):
        shutil.copy2(source / name, destination / name)

    def write_index(path, entries, subset=ids):
        path.write_text("".join(f"{uid} {entries[uid]}\n" for uid in subset))

    def absolute(entries):
        return {
            uid: str(
                (Path(value) if Path(value).is_absolute() else recipe / value).resolve()
            )
            for uid, value in entries.items()
        }

    write_index(destination / "wav.scp", absolute(mixtures))
    refs = {}
    for branch in ("clean", "reverb"):
        refs[branch] = []
        for spk in (1, 2):
            path = (
                test / f"spk{spk}.scp"
                if branch == "clean"
                else reverb / f"spk{spk}_reverb.scp"
            )
            entries = absolute(index(path))
            if set(entries) != set(ids):
                raise ValueError(f"Reference/mixture IDs differ: {path}")
            refs[branch].append(entries)
            write_index(destination / f"ref_{branch}{spk}.scp", entries)
            write_index(destination / f"ref_{branch}{spk}.smoke.scp", entries, ids[:3])
    versions = {
        name: version(name)
        for name in ("numpy", "scipy", "torch", "soundfile", "mir_eval", "pystoi")
    }
    (output / "metadata.json").write_text(
        json.dumps(
            dict(
                checkpoint="Rec-RIR 97epoch.pth",
                checkpoint_sha256=digest(destination / "valid.loss.best.pth"),
                sample_rate=8000,
                channel=0,
                utterances=3000,
                source_pairs=6000,
                normalize_output_wav=True,
                peak=0.9,
                subtype="PCM_16",
                exact_input_length=True,
                permutation="ESPnet enh_scoring: mir_eval bss_eval_sources default SIR assignment",
                packages=versions,
                original_mixture_index_sha256=digest(test / "wav.scp"),
                scorer_sha256=digest(
                    Path(__file__).resolve().parents[4] / "espnet2/bin/enh_scoring.py"
                ),
                references={
                    branch: [
                        digest(destination / f"ref_{branch}{spk}.scp") for spk in (1, 2)
                    ]
                    for branch in refs
                },
            ),
            indent=2,
        )
        + "\n"
    )

    def score(branch, inference, folder, keys, smoke=False):
        folder.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "espnet2.bin.enh_scoring",
            "--output_dir",
            str(folder),
            "--key_file",
            str(keys),
            "--ref_channel",
            "0",
        ]
        for spk in (1, 2):
            reference = (
                destination / f"ref_{branch}{spk}{'.smoke' if smoke else ''}.scp"
            )
            command += [
                "--ref_scp",
                str(reference),
                "--inf_scp",
                str(inference / branch / f"spk{spk}.scp"),
            ]
        with Path(str(folder) + ".log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    # Reproduce the stored DAMSEP scores with the current references and scorer.
    # This checks cross-host audio/metric compatibility before all Rec-RIR work.
    smoke = output / "input/damsep_smoke"
    (destination / "smoke_keys.scp").write_text("".join(f"{uid}\n" for uid in ids[:3]))
    smoke_checks = {}
    for branch in refs:
        for spk in (1, 2):
            write_index(
                smoke / branch / f"spk{spk}.scp",
                {
                    uid: str(smoke / branch / f"wav_spk{spk}" / f"{uid}.wav")
                    for uid in ids[:3]
                },
                ids[:3],
            )
        folder = output / "validation" / f"damsep_{branch}"
        score(branch, smoke, folder, destination / "smoke_keys.scp", smoke=True)
        maximum_error = 0.0
        for metric in METRICS:
            for spk in (1, 2):
                previous = index(
                    output / "input/damsep" / branch / f"{metric}_spk{spk}"
                )
                current = index(folder / f"{metric}_spk{spk}")
                for uid in ids[:3]:
                    error = abs(float(previous[uid]) - float(current[uid]))
                    if error > 1e-4:
                        raise ValueError(
                            f"DAMSEP scoring not reproduced: {branch} {metric} {uid}: {error}"
                        )
                    maximum_error = max(maximum_error, error)
        smoke_checks[branch] = maximum_error
    (output / "validation/damsep_score_reproduction.json").write_text(
        json.dumps(smoke_checks, indent=2) + "\n"
    )
    print(f"DAMSEP score reproduction passed: {smoke_checks}", flush=True)

    command = [
        sys.executable,
        "-m",
        "espnet2.bin.rec_rir_speech_inference",
        "--train_config",
        str(destination / "config.yaml"),
        "--model_file",
        str(destination / "valid.loss.best.pth"),
        "--wav_scp",
        str(destination / "wav.scp"),
        "--device",
        "cuda:0",
    ]
    subprocess.run(
        command + ["--limit", "3", "--output_dir", str(destination / "smoke")],
        check=True,
    )
    print("Rec-RIR speech smoke passed; starting all 3000 mixtures", flush=True)
    subprocess.run(
        command + ["--output_dir", str(destination / "inference")], check=True
    )
    print("Rec-RIR inference complete; starting CPU scoring", flush=True)
    tasks = []
    for branch in refs:
        folder = destination / "inference" / branch / "scoring"
        folder.mkdir()
        for job in range(1, args.workers + 1):
            keys = folder / f"keys.{job}.scp"
            keys.write_text("".join(f"{uid}\n" for uid in ids[job - 1 :: args.workers]))
            tasks.append(
                (branch, destination / "inference", folder / f"output.{job}", keys)
            )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(score, *task) for task in tasks]
        for future in futures:
            future.result()
    report = {"count": 3000, "source_pairs": 6000, "metrics": {}}
    for branch in refs:
        folder = destination / "inference" / branch / "scoring"
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2]
                    / "damsep_clean/local/summarize_scores.py"
                ),
                "--scoring-dir",
                str(folder),
                "--jobs",
                str(args.workers),
            ],
            check=True,
        )
        report["metrics"][branch] = {
            "RecRIR_best97": summary(folder, ids),
            "DAMSEP_best29": summary(output / "input/damsep" / branch, ids),
        }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["branch", "model", *METRICS])
        writer.writeheader()
        for branch, models in report["metrics"].items():
            for name, values in models.items():
                writer.writerow(dict(branch=branch, model=name, **values))
    (output / "complete.json").write_text(
        json.dumps(
            dict(
                count=3000,
                source_pairs=6000,
                all_scores_finite=True,
                elapsed_seconds=time.monotonic() - started,
                git_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                    cwd=Path(__file__).resolve().parents[4],
                ).strip(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
