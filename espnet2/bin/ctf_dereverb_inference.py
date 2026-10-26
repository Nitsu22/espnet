#!/usr/bin/env python3
"""Mixture-only CTF inference, named branch export and common-PIT scoring.

Audio paths in SCP files are resolved relative to their SCP file, or relative to
``--audio_root`` when explicitly supplied.  Shell pipelines and archive specs
are not supported.  Output WAVs retain the model's speaker slots and amplitude;
references are used only after inference to score a single common assignment.
"""

import argparse
import json
import math
import re
import warnings
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.enh import EnhancementTask
from espnet2.utils.types import str2bool


CLEAN_BRANCHES = ("clean_prior", "clean_deconv", "clean_final")
SAMPLE_RATE = 8000


def read_wav_scp(filename, audio_root=None):
    """Read distinct safe IDs and plain audio paths without executing commands."""
    filename = Path(filename).resolve()
    root = filename.parent if audio_root is None else Path(audio_root).resolve()
    entries = {}
    with filename.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Malformed SCP line {filename}:{number}")
            uid, value = fields
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", uid):
                raise ValueError(f"Unsafe utterance ID: {uid!r}")
            if uid in entries:
                raise ValueError(f"Duplicate utterance ID: {uid}")
            if value.endswith("|") or value.startswith(("ark:", "scp:")):
                raise ValueError("Use plain audio file paths, not pipes or archives")
            path = Path(value)
            entries[uid] = (
                path.resolve() if path.is_absolute() else (root / path).resolve()
            )
    if not entries:
        raise ValueError(f"SCP contains no utterances: {filename}")
    return entries


def _read_mono(path):
    """Read the first (left) microphone, matching training preprocessing."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"Expected actual {SAMPLE_RATE} Hz audio: {path} is {rate} Hz")
    if audio.shape[1] not in (1, 2) or audio.shape[0] < 2:
        raise ValueError(f"Expected nonempty mono or stereo audio: {path}")
    if not np.isfinite(audio[:, 0]).all():
        raise ValueError(f"Nonfinite audio samples: {path}")
    return audio[:, 0]


def _si_sdr(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    epsilon = 1e-12
    projection = reference * (
        np.dot(estimate, reference) / max(np.dot(reference, reference), epsilon)
    )
    target = max(np.dot(projection, projection), epsilon)
    error = estimate - projection
    return float(10 * np.log10(target / max(np.dot(error, error), epsilon)))


def _snr(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = estimate - reference
    return float(10 * np.log10(
        max(np.dot(reference, reference), 1e-12)
        / max(np.dot(error, error), 1e-12)
    ))


def score_branches(branches, references, mixture, stoi_function=None):
    """Select final-clean SI-SDR PIT once, then score every saved branch.

    ``branches`` maps names to real ``[2, samples]`` arrays.  References never
    enter the model.  Returned assignment maps predicted slots to zero-based
    reference indices; export files keep the original predicted slot order.
    """
    references = np.asarray(references)
    mixture = np.asarray(mixture)
    if mixture.ndim != 1 or references.shape != (2, mixture.shape[0]):
        raise ValueError("Expected two references matching the mono mixture length")
    if "clean_final" not in branches:
        raise ValueError("The final clean branch is required to choose the common PIT")
    for name, branch in branches.items():
        if np.asarray(branch).shape != references.shape:
            raise ValueError(f"Invalid branch waveform shape: {name}")
        if not np.isfinite(branch).all():
            raise ValueError(f"Nonfinite branch waveform: {name}")
    if not np.isfinite(references).all() or not np.isfinite(mixture).all():
        raise ValueError("References and mixture must be finite")
    candidates = list(permutations(range(2)))
    costs = [
        sum(_si_sdr(branches["clean_final"][slot], references[ref])
            for slot, ref in enumerate(assignment))
        for assignment in candidates
    ]
    assignment = candidates[int(np.argmax(costs))]
    metrics = {}
    for name, branch in branches.items():
        source_metrics = []
        for slot, reference_index in enumerate(assignment):
            reference = references[reference_index]
            si_sdr = _si_sdr(branch[slot], reference)
            values = dict(
                si_sdr_db=si_sdr,
                si_sdri_db=si_sdr - _si_sdr(mixture, reference),
                snr_db=_snr(branch[slot], reference),
                stoi=None,
            )
            if stoi_function is not None:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", RuntimeWarning)
                        value = float(stoi_function(
                            reference, branch[slot], SAMPLE_RATE, extended=False
                        ))
                    insufficient = [
                        str(item.message) for item in caught
                        if "Not enough STFT frames" in str(item.message)
                    ]
                    if insufficient:
                        values["stoi_unavailable_reason"] = insufficient[0]
                    elif math.isfinite(value):
                        values["stoi"] = value
                    else:
                        values["stoi_unavailable_reason"] = (
                            "STOI returned a nonfinite value"
                        )
                except (ValueError, RuntimeError) as error:
                    values["stoi_unavailable_reason"] = str(error)
            source_metrics.append(values)
        average = {
            key: float(np.mean([values[key] for values in source_metrics]))
            for key in ("si_sdr_db", "si_sdri_db", "snr_db")
        }
        stoi_values = [values["stoi"] for values in source_metrics]
        average["stoi"] = (
            float(np.mean(stoi_values)) if all(v is not None for v in stoi_values)
            else None
        )
        metrics[name] = dict(average, per_source=source_metrics)
    return assignment, metrics


def _override_deconvolution(model, enabled, gamma, iterations):
    if enabled is None and gamma is None and iterations is None:
        return
    separator = model.separator
    if not hasattr(separator, "use_deconvolution"):
        raise ValueError("This model has no CTF deconvolution path to override")
    if enabled is not None:
        separator.use_deconvolution = enabled
    if gamma is not None:
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        separator.cg_gamma = gamma
    if iterations is not None:
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        separator.cg_iterations = iterations


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inference(
    train_config,
    model_file,
    wav_scp,
    output_dir,
    device="cpu",
    ref1_scp=None,
    ref2_scp=None,
    enable_deconvolution: Optional[bool] = None,
    gamma=None,
    iterations=None,
    audio_root=None,
    write_reverb=False,
):
    """Load one checkpoint, infer mixture-only and export every clean branch."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    mixtures = read_wav_scp(wav_scp, audio_root)
    if (ref1_scp is None) != (ref2_scp is None):
        raise ValueError("Supply both reference SCPs, or neither")
    references = None
    if ref1_scp is not None:
        references = [read_wav_scp(path, audio_root) for path in (ref1_scp, ref2_scp)]
        if any(set(ref) != set(mixtures) for ref in references):
            raise ValueError(
                "Reference SCPs must have exactly the mixture utterance IDs"
            )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    model, _ = EnhancementTask.build_model_from_file(
        str(train_config), str(model_file), device
    )
    if not hasattr(model, "predict_all"):
        raise ValueError("Use a checkpoint exposing CTF predict_all inference")
    model.eval()
    _override_deconvolution(model, enable_deconvolution, gamma, iterations)
    try:
        from pystoi import stoi
        stoi_status = dict(available=True)
    except ImportError:
        stoi = None
        stoi_status = dict(available=False, reason="pystoi is not installed")
    branches = CLEAN_BRANCHES + (("reverb",) if write_reverb else ())
    output_dir.mkdir(parents=True, exist_ok=True)
    scp_entries = {name: [[], []] for name in branches}
    ctf_entries = []
    utterance_results = {}
    with torch.no_grad():
        for uid, path in mixtures.items():
            mixture = _read_mono(path)
            tensor = torch.from_numpy(mixture).unsqueeze(0).to(device)
            lengths = torch.tensor([len(mixture)], device=device)
            # These are the ONLY model inputs. No reference enters prediction.
            result = model.predict_all(tensor, lengths)
            actual_length = int(result["sample_lengths"][0])
            if actual_length != len(mixture):
                raise ValueError("Model sample lengths do not match the input waveform")
            waveforms = {}
            for name in branches:
                if name not in result["waveforms"]:
                    raise ValueError(f"Model is missing the requested branch: {name}")
                waveform = result["waveforms"][name]
                if (
                    waveform.ndim != 3 or waveform.shape[:2] != (1, 2)
                    or waveform.shape[-1] < actual_length
                    or torch.is_complex(waveform)
                ):
                    raise ValueError(f"Expected branch [1,2,samples]: {name}")
                value = waveform[0, :, :actual_length].cpu().numpy().astype(np.float32)
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite output waveform: {name}")
                waveforms[name] = value
                for speaker in range(2):
                    directory = output_dir / name / f"spk{speaker + 1}"
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / f"{uid}.wav"
                    with destination.open("xb") as stream:
                        sf.write(stream, value[speaker], SAMPLE_RATE,
                                 format="WAV", subtype="FLOAT")
                    scp_entries[name][speaker].append(f"{uid} {destination}\n")
            if "ctf" in result:
                ctf = result["ctf"]
                if (
                    ctf.ndim != 4 or ctf.shape[:2] != (1, 2)
                    or not torch.is_complex(ctf)
                ):
                    raise ValueError("Expected complex CTF [1,2,F,L]")
                if not torch.isfinite(ctf).all():
                    raise ValueError("Nonfinite estimated CTF")
                directory = output_dir / "ctf"
                directory.mkdir(exist_ok=True)
                destination = directory / f"{uid}.npy"
                with destination.open("xb") as stream:
                    np.save(stream, ctf[0].cpu().numpy(), allow_pickle=False)
                ctf_entries.append(f"{uid} {destination}\n")
            metadata = dict(
                samples=actual_length,
                frames=int(result["frame_lengths"][0]),
                normalization_scale=float(result["normalization_scale"].reshape(-1)[0]),
            )
            if references is not None:
                paired = np.stack([_read_mono(ref[uid]) for ref in references])
                if paired.shape != (2, actual_length):
                    raise ValueError(f"Reference lengths differ from mixture: {uid}")
                assignment, metrics = score_branches(waveforms, paired, mixture, stoi)
                metadata["reference_assignment"] = [index + 1 for index in assignment]
                metadata["metrics"] = metrics
            utterance_results[uid] = metadata
    for name, sources in scp_entries.items():
        for speaker, entries in enumerate(sources, 1):
            with (output_dir / name / f"spk{speaker}.scp").open("x") as stream:
                stream.writelines(entries)
    if ctf_entries:
        with (output_dir / "ctf.scp").open("x") as stream:
            stream.writelines(ctf_entries)
    means = {}
    if references is not None:
        for name in branches:
            entries = [item["metrics"][name] for item in utterance_results.values()]
            means[name] = {
                key: float(np.mean([entry[key] for entry in entries]))
                for key in ("si_sdr_db", "si_sdri_db", "snr_db")
            }
            values = [entry["stoi"] for entry in entries]
            means[name]["stoi"] = (
                float(np.mean(values)) if all(value is not None for value in values)
                else None
            )
    separator = model.separator
    summary = dict(
        train_config=str(Path(train_config).resolve()),
        model_file=str(Path(model_file).resolve()),
        sample_rate=SAMPLE_RATE, num_utterances=len(mixtures),
        channel_policy="first (left) microphone, matching training",
        scored=references is not None, branches=list(branches),
        ctf_available=bool(ctf_entries), stoi=stoi_status,
        metric_aggregation="equal utterance mean of the two source means",
        assignment=(
            "predicted slots to one-based reference indices, chosen on final SI-SDR"
        ),
        deconvolution=dict(
            enabled=getattr(separator, "use_deconvolution", False),
            gamma=getattr(separator, "cg_gamma", None),
            iterations=getattr(separator, "cg_iterations", None),
        ),
        metrics=means, utterances=utterance_results,
    )
    _write_json(output_dir / "summary.json", summary)
    return summary


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_config", "model_file", "wav_scp", "output_dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ref1_scp")
    parser.add_argument("--ref2_scp")
    parser.add_argument("--audio_root")
    parser.add_argument("--enable_deconvolution", type=str2bool, default=None)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--write_reverb", type=str2bool, default=False)
    return parser


def main(cmd=None):
    inference(**vars(get_parser().parse_args(cmd)))


if __name__ == "__main__":
    main()
