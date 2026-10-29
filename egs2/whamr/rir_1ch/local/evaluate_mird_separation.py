#!/usr/bin/env python3
"""Score a two-speaker separation checkpoint on validated MIRD mixtures.

Report zero-mean SI-SDR and SI-SDR improvement with the best speaker permutation.
Reverberant source targets measure separation; signed peak-tap direct targets
also penalize remaining reverberation. No time alignment is fitted at scoring.
"""

import argparse
import itertools
import json
import math
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import create_whamr_but_dump as common


def si_sdr(reference, estimate):
    """Scale/polarity-invariant, zero-mean SI-SDR; clamp ratios at +/-120 dB."""
    reference, estimate = [np.asarray(x, dtype=np.float64) for x in
                           (reference, estimate)]
    if (reference.ndim != 1 or reference.shape != estimate.shape
            or len(reference) < 2 or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(estimate))):
        raise ValueError("Expected finite matching mono reference and estimate")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    ref_norm, est_norm = np.linalg.norm(reference), np.linalg.norm(estimate)
    if ref_norm == 0:
        raise ValueError("SI-SDR is undefined for a silent/constant reference")
    if est_norm == 0:
        return -120.0
    reference, estimate = reference / ref_norm, estimate / est_norm
    projection = float(np.dot(reference, estimate)) * reference
    target = float(np.dot(projection, projection))
    residual = estimate - projection
    error = float(np.dot(residual, residual))
    return 10 * math.log10(max(target, 1e-12) / max(error, 1e-12))


def pit_scores(references, estimates, mixture):
    if len(references) != 2 or len(estimates) != 2:
        raise ValueError("Expected exactly two references and separated streams")
    pairwise = [[si_sdr(ref, est) for est in estimates] for ref in references]
    permutation = max(itertools.permutations(range(2)),
                      key=lambda p: sum(pairwise[i][p[i]] for i in range(2)))
    separated = [pairwise[i][permutation[i]] for i in range(2)]
    baseline = [si_sdr(ref, mixture) for ref in references]
    return dict(permutation=list(permutation), si_sdr_per_speaker=separated,
                input_si_sdr_per_speaker=baseline,
                si_sdr_db=float(np.mean(separated)),
                input_si_sdr_db=float(np.mean(baseline)),
                si_sdri_db=float(np.mean(np.subtract(separated, baseline))))


def read_scp(path):
    entries = [line.split(maxsplit=1) for line in Path(path).read_text().splitlines()]
    if any(len(x) != 2 for x in entries):
        raise ValueError("Malformed SCP: " + str(path))
    result = dict(entries)
    if len(result) != len(entries):
        raise ValueError("Duplicate SCP IDs: " + str(path))
    return result


def read_wave(path, sample_rate, samples=None):
    wave, actual_rate = sf.read(path, dtype="float64")
    if (actual_rate != sample_rate or wave.ndim != 1 or not len(wave)
            or not np.all(np.isfinite(wave))
            or (samples is not None and len(wave) != samples)):
        raise ValueError("Mismatched mono evaluation audio: " + str(path))
    return wave


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def evaluate(args):
    # Import the model stack only for actual inference, not metric-only tests.
    import torch
    from espnet2.bin.enh_inference import SeparateSpeech

    dataset, out = args.dataset.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new separation result directory")
    validation_path = dataset / "validation.json"
    validation = json.loads(validation_path.read_text())
    if not validation["state"].startswith("validated_"):
        raise ValueError("Dataset has not passed audio validation")
    config_path = args.model_config.resolve()
    checkpoint = args.checkpoint.resolve()
    model_config = yaml.safe_load(config_path.read_text())
    if (model_config.get("sample_rate") != args.sample_rate
            or model_config.get("num_spk") != 2
            or not model_config.get("force_single_channel", False)):
        raise ValueError("Use a matching-rate, single-channel, two-speaker checkpoint")
    selected = json.loads((dataset / "selected_records.json").read_text())
    by_id = {row["utterance_id"]: row for row in selected}
    if len(by_id) != len(selected) or len(selected) != validation["utterances_per_set"]:
        raise ValueError("Validation/selection mixture counts differ")
    ids = list(by_id)
    if args.limit is not None:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No mixtures selected")
    indices = {}
    for condition in args.conditions:
        folder = dataset / "raw" / (
            f"tt_mird_2spk_{condition}_reverb_min_{args.sample_rate // 1000}k"
        )
        names = ["wav.scp"] + [f"speech_{kind}{side}.scp"
                                 for kind in ("reverb", "direct") for side in (1, 2)]
        indices[condition] = {name: read_scp(folder / name) for name in names}
        if any(set(index) != set(by_id) for index in indices[condition].values()):
            raise ValueError(
                "Mixture/reference SCP IDs differ from validated selection"
            )
    out.mkdir(parents=True)
    snapshot = out / "model"
    snapshot.mkdir()
    shutil.copyfile(config_path, snapshot / "config.yaml")
    shutil.copyfile(checkpoint, snapshot / "model.pth")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    metadata = dict(
        dataset=str(dataset), dataset_validation=validation,
        dataset_validation_sha256=common.digest(validation_path),
        selected_records_sha256=common.digest(dataset / "selected_records.json"),
        checkpoint=str(checkpoint),
        checkpoint_sha256=common.digest(snapshot / "model.pth"),
        model_config=str(config_path),
        model_config_sha256=common.digest(snapshot / "config.yaml"),
        evaluator_sha256=common.digest(Path(__file__)), host=socket.gethostname(),
        device=args.device, sample_rate=args.sample_rate, conditions=args.conditions,
        selected_count=len(ids), limit=args.limit, seed=0,
        metric="zero-mean SI-SDR, best of two speaker permutations; +/-120 dB clamp",
        averaging="equal utterance weights; average of both speakers per utterance",
        targets=dict(reverb="individual reverberant sources: separation",
                     direct=("signed peak-tap direct sources: "
                             "separation and dereverberation")),
        fitted_time_alignment=False, torch_version=torch.__version__,
    )
    save_json(out / "evaluation_config.json", metadata)
    separator = SeparateSpeech(
        train_config=snapshot / "config.yaml", model_file=snapshot / "model.pth",
        device=args.device, dtype="float32", normalize_output_wav=False,
    )
    if separator.num_spk != 2:
        raise ValueError("Loaded model does not output two speech streams")
    scores = []
    start = time.monotonic()
    with (out / "per_utterance.jsonl").open("w") as stream:
        for condition in args.conditions:
            index = indices[condition]
            for count, uid in enumerate(ids, 1):
                mix = read_wave(index["wav.scp"][uid], args.sample_rate)
                with torch.inference_mode():
                    predictions = separator(
                        mix.astype(np.float32)[None], args.sample_rate
                    )
                if len(predictions) != 2:
                    raise ValueError("Expected exactly two separated streams")
                waves = []
                for wave in predictions:
                    if isinstance(wave, torch.Tensor):
                        wave = wave.detach().cpu().numpy()
                    wave = np.asarray(wave, dtype=np.float64)
                    if wave.shape != (1, len(mix)) or not np.all(np.isfinite(wave)):
                        raise ValueError("Separated speech has invalid shape or values")
                    waves.append(wave[0])
                selection = by_id[uid]
                record = dict(utterance_id=uid, condition=condition,
                              sample_rate=args.sample_rate, samples=len(mix),
                              rt60_ms=selection["rt60_ms"],
                              spacing=selection["spacing"],
                              microphone=selection["microphone"], targets={})
                for target in ("reverb", "direct"):
                    refs = [read_wave(index[f"speech_{target}{side}.scp"][uid],
                                      args.sample_rate, len(mix)) for side in (1, 2)]
                    record["targets"][target] = pit_scores(refs, waves, mix)
                if args.save_audio:
                    for side, wave in enumerate(waves, 1):
                        common.write_audio(out / "audio" / condition / f"speaker{side}"
                                           / f"{uid}.wav", wave, args.sample_rate)
                scores.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if count % 50 == 0 or count == len(ids):
                    print(f"Scored {condition}: {count}/{len(ids)}", flush=True)
    grouped = defaultdict(list)
    for row in scores:
        for target, values in row["targets"].items():
            grouped[row["condition"], target].append(values)
    metrics = {}
    for (condition, target), values in grouped.items():
        metrics[f"{condition}/{target}"] = dict(
            count=len(values), **{key: float(np.mean([row[key] for row in values]))
                                 for key in ("si_sdr_db", "input_si_sdr_db",
                                             "si_sdri_db")})
    expected_count = len(ids) * len(args.conditions)
    if len(scores) != expected_count:
        raise ValueError("Incomplete separation evaluation")
    summary = dict(
        state="separation_evaluation_complete", sample_rate=args.sample_rate,
        utterances_per_condition=len(ids), conditions=args.conditions,
        all_selected_dataset_ids_evaluated=len(ids) == len(selected),
        is_original_nine_condition_production=validation["is_production_dataset"],
        dataset_mode=validation["mode"], inference_seconds=time.monotonic() - start,
        metrics=metrics,
    )
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--conditions", nargs="+", choices=("clean", "noisy"),
                        default=["clean", "noisy"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()
    if (args.threads < 1 or (args.limit is not None and args.limit < 1)
            or len(set(args.conditions)) != len(args.conditions)):
        parser.error("Use positive threads/limit and distinct conditions")
    evaluate(args)


if __name__ == "__main__":
    main()
