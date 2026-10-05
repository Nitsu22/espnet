#!/usr/bin/env python3
"""Extend the Small baseline dump with matched 8-kHz reverberant teachers.

Never regenerate mixtures or substitute noise-free data. Verify the source
mixture/direct teachers against the exact baseline audio before accepting its
reverberant teachers. One PCM16 quantization step is allowed because ESPnet's
formatted baseline WAVs can differ from the original FLOAT WAVs by that amount.
PCM16 clipping is also reproduced when checking source/baseline identity; the
actual baseline inputs and clean references are retained, never rewritten.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SIGNALS = {
    "wav.scp": "speech_mix",
    "spk1.scp": "speech_ref1",
    "spk2.scp": "speech_ref2",
    "spk1_reverb.scp": "speech_reverb1",
    "spk2_reverb.scp": "speech_reverb2",
}
SPLITS = {"tr": ("train", 20000), "cv": ("valid", 5000), "tt": ("test", 3000)}


def read_index(path):
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        if key in result:
            raise ValueError(f"Duplicate utterance {key}: {path}")
        result[key] = value
    return result


def audio_path(value, root):
    if value.endswith("|"):
        raise ValueError("Expected existing WAV paths, not command pipelines")
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_left(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 8000 or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite 8-kHz speech: {path}")
    return audio[:, 0]


def prepare(
    baseline_root,
    source_root,
    output,
    baseline_dump="dump/raw",
    baseline_stats="exp/enh_stats_8k",
    limit=None,
):
    baseline_root = Path(baseline_root).resolve()
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dump: {output}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    report = {
        "sample_rate": 8000,
        "microphone": "left",
        "input": "baseline mix_both_reverb",
        "baseline_root": str(baseline_root),
        "source_root": str(source_root),
        "baseline_dump": baseline_dump,
        "baseline_stats": baseline_stats,
        "audio_tolerance": 1.0 / 32768 + 1.0e-7,
        "splits": {},
    }
    try:
        for split, (shape_split, expected) in SPLITS.items():
            dataset = f"{split}_mix_both_reverb_min_8k"
            baseline = baseline_root / baseline_dump / dataset
            source = source_root / dataset
            missing = [
                str(source / name) for name in SIGNALS if not (source / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Matched original WHAMR indexes are missing. Restore the "
                    "baseline's original audio/indexes or provide --source-data-root; "
                    "do not use regenerated/noise-free mixtures. Missing: "
                    + ", ".join(missing)
                )
            old = {name: read_index(baseline / name) for name in list(SIGNALS)[:3]}
            original = {name: read_index(source / name) for name in SIGNALS}
            keys = list(old["wav.scp"])
            for index in [*old.values(), *original.values()]:
                if set(index) != set(keys):
                    raise ValueError(f"Utterance IDs differ: {dataset}")
            if limit is None and len(keys) != expected:
                raise ValueError(f"Expected {expected} baseline utterances: {dataset}")
            keys = keys[:limit] if limit is not None else keys
            target = temporary / "raw" / dataset
            target.mkdir(parents=True)
            lines = {name: [] for name in SIGNALS}
            lengths = {}
            worst_error = 0.0
            clipped_samples = 0
            for i, key in enumerate(keys):
                n = None
                source_audio = {}
                for name in SIGNALS:
                    candidate = audio_path(original[name][key], source_root.parent)
                    data = read_left(candidate)
                    source_audio[name] = data
                    if n is None:
                        n = len(data)
                    if len(data) != n:
                        raise ValueError(f"Unaligned source audio: {key} {name}")
                    if name in old:
                        chosen = audio_path(old[name][key], baseline_root)
                        reference = read_left(chosen)
                        if len(reference) != n:
                            raise ValueError(f"Baseline/source lengths differ: {key}")
                        expected_audio = data
                        if sf.info(chosen).subtype == "PCM_16":
                            clipped_samples += int(np.count_nonzero(np.abs(data) > 1))
                            expected_audio = np.clip(data, -1, 1)
                        error = float(np.max(np.abs(reference - expected_audio)))
                        worst_error = max(worst_error, error)
                        if error > report["audio_tolerance"]:
                            raise ValueError(
                                f"Source is not the baseline dataset: {key} {name}, "
                                f"max error {error}. No alternate data is substituted."
                            )
                    else:
                        chosen = candidate
                    lines[name].append(f"{key} {chosen}\n")
                # The residual is the existing additive noise, not a new input.
                # Source-clean equality above ties both reverb speakers to the
                # baseline's same speaker slots and acoustic realization.
                noise_index = source / "noise1.scp"
                if not noise_index.is_file():
                    raise FileNotFoundError(
                        f"Noise pairing index missing: {noise_index}"
                    )
                # Loaded once below for efficiency, on the first utterance.
                if i == 0:
                    noise = read_index(noise_index)
                    if set(noise) != set(old["wav.scp"]):
                        raise ValueError(f"Noise IDs differ: {dataset}")
                noise_audio = read_left(audio_path(noise[key], source_root.parent))
                if len(noise_audio) != n:
                    raise ValueError(f"Noise length differs: {key}")
                residual = (
                    source_audio["wav.scp"]
                    - source_audio["spk1_reverb.scp"]
                    - source_audio["spk2_reverb.scp"]
                    - noise_audio
                )
                if np.max(np.abs(residual)) > 4.0 / 32768:
                    raise ValueError(
                        f"Reverb speakers do not reconstruct mixture: {key}"
                    )
                lengths[key] = n
                if (i + 1) % 500 == 0:
                    print(f"Verified {dataset}: {i + 1}/{len(keys)}", flush=True)
            for name, content in lines.items():
                (target / name).write_text("".join(content))
            (target / "feats_type").write_text("raw\n")
            (target / "utt2num_samples").write_text(
                "".join(f"{key} {lengths[key]}\n" for key in keys)
            )
            for name in ("utt2spk", "spk2utt"):
                if (baseline / name).exists() and limit is None:
                    shutil.copyfile(baseline / name, target / name)
            # Only the baseline's original three shape files control batching.
            # Adding CTF supervision must not change folded batch membership.
            if shape_split != "test":
                stats = temporary / "stats" / shape_split
                stats.mkdir(parents=True)
                for name in list(SIGNALS.values())[:3]:
                    shape_file = (
                        baseline_root / baseline_stats / shape_split / (name + "_shape")
                    )
                    shapes = read_index(shape_file)
                    if set(shapes) != set(old["wav.scp"]):
                        raise ValueError(f"Baseline shape IDs differ: {shape_file}")
                    for key in keys:
                        if int(shapes[key].split(",")[0]) != lengths[key]:
                            raise ValueError(f"Baseline shape length differs: {key}")
                    (stats / (name + "_shape")).write_text(
                        "".join(f"{key} {shapes[key]}\n" for key in keys)
                    )
            report["splits"][split] = {
                "count": len(keys),
                "max_audio_error_after_pcm16_clipping": worst_error,
                "source_samples_clipped_in_baseline": clipped_samples,
            }
        report["index_sha256"] = {
            str(p.relative_to(temporary)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(temporary.glob("raw/*/*.scp"))
        }
        (temporary / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"Prepared matched joint-CTF dump: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default="../enh1")
    parser.add_argument("--source-data-root", default="../enh1/data")
    parser.add_argument("--baseline-dump", default="dump/raw")
    parser.add_argument("--baseline-stats", default="exp/enh_stats_8k")
    parser.add_argument("--output", default="dump_ctf_joint")
    parser.add_argument("--limit", type=int, help="Isolated small validation dump only")
    args = parser.parse_args()
    prepare(
        args.baseline_root,
        args.source_data_root,
        args.output,
        args.baseline_dump,
        args.baseline_stats,
        args.limit,
    )


if __name__ == "__main__":
    main()
