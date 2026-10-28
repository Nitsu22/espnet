"""Scoped inference-only waveform replacement for the frozen CTF estimator."""

import hashlib

import numpy as np
import torch


def ctf_slot_override(separate, mixture, slots=(0, 0)):
    """Route estimated speaker slots without changing estimator/encoder audio.

    The predictor's native output is [B, time, speaker, frequency, complex].
    Select the entire speaker CTF, not its real/imaginary or frequency axis.
    (0, 1) is the identity control; (0, 0) duplicates the first native slot.
    No PIT or score-based slot selection is performed.
    """
    model = separate.enh_model
    if separate.segmenting or separate.num_spk != 2 or not hasattr(model, "frozen_ctf"):
        raise ValueError("Requires full-waveform two-speaker frozen-BiMamba inference")
    if (mixture.ndim != 2 or mixture.shape[0] != 1
            or not np.isfinite(mixture).all()):
        raise ValueError("Requires a finite [1, samples] mixture")
    if (len(slots) != 2 or any(type(k) is not int or k not in (0, 1) for k in slots)):
        raise ValueError("Requires two speaker slot indices in (0, 1)")
    records = []

    def checksum(value):
        return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    def replace(_module, _inputs, output):
        if tuple(output.shape) != (1, 60, 2, 129, 2) or not torch.isfinite(output).all():
            raise ValueError("Invalid estimated CTF")
        routed = output.index_select(2, torch.tensor(slots, device=output.device))
        if any(not torch.equal(routed[:, :, j], output[:, :, k]) for j, k in enumerate(slots)):
            raise ValueError("Speaker-slot copy was not exact")
        records.append(dict(slot_indices=list(slots), source_ctf_sha256=checksum(output),
                            ctf_sha256=checksum(routed),
                            source_slot_sha256=[checksum(output[:, :, k]) for k in (0, 1)],
                            used_slot_sha256=[checksum(routed[:, :, k]) for k in (0, 1)]))
        return routed

    handle = model.frozen_ctf.register_forward_hook(replace)
    try:
        with torch.no_grad():
            result = separate(mixture, fs=8000)
    finally:
        handle.remove()
    if len(records) != 1:
        raise ValueError("CTF slot intervention was not applied exactly once")
    return result, records[0]


def conditioning_override(separate, mixture, conditioning):
    """Keep standard SeparateSpeech inference; replace ONLY frozen_ctf input.

    Full-waveform, mono, batch1 only. Hooks are removed even on exceptions.
    No weights, training behavior or shared inference/model APIs are changed.
    """
    model = separate.enh_model
    if separate.segmenting or separate.num_spk != 2 or not hasattr(model, "frozen_ctf"):
        raise ValueError("Requires full-waveform two-speaker frozen-BiMamba inference")
    if mixture.shape != conditioning.shape or mixture.ndim != 2 or mixture.shape[0] != 1:
        raise ValueError("Mixture/conditioning must have identical [1, samples] shape")
    if not np.isfinite(mixture).all() or not np.isfinite(conditioning).all():
        raise ValueError("Nonfinite intervention input")
    wave = torch.as_tensor(conditioning, device=separate.device,
                           dtype=getattr(torch, separate.dtype))
    # Mirror SeparateSpeech's mono input normalization, including torch.std's
    # default correction. Its encoder/decoder still receive the ORIGINAL mix.
    if (getattr(model, "normalize_variance", False)
            or getattr(model, "normalize_variance_per_ch", False)):
        std = torch.std(wave, dim=1, keepdim=True)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Silent/invalid conditioning waveform")
        wave = wave / std
    calls, hashes = [], []

    def replace(_module, inputs):
        if (len(inputs) != 2 or inputs[0].shape != wave.shape
                or int(inputs[1][0]) != wave.shape[1]):
            raise ValueError("Unexpected frozen predictor call")
        calls.append(1)
        return wave, inputs[1]

    def inspect(_module, _inputs, output):
        if tuple(output.shape) != (1, 60, 2, 129, 2) or not torch.isfinite(output).all():
            raise ValueError("Invalid estimated CTF")
        data = output.detach().float().cpu().contiguous().numpy()
        hashes.append(hashlib.sha256(data.tobytes()).hexdigest())

    before = model.frozen_ctf.register_forward_pre_hook(replace)
    after = model.frozen_ctf.register_forward_hook(inspect)
    try:
        with torch.no_grad():
            result = separate(mixture, fs=8000)
    finally:
        before.remove()
        after.remove()
    if len(calls) != 1 or len(hashes) != 1:
        raise ValueError("Conditioning intervention was not applied exactly once")
    return result, hashes[0]
