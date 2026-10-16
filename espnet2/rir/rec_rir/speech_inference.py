"""Decode the existing Rec-RIR clean and reverberant speech heads."""

import torch


@torch.no_grad()
def estimate_speech(model, mixture):
    """Return two [speaker, sample] waveforms, preserving input length/gain.

    Analysis and head selection follow ESPnetRecRIRPITModel.forward(). The
    explicit ISTFT length retains the final partial hop of odd-length audio.
    No target audio, RIR, PIT assignment or waveform alignment is used here.
    """
    if mixture.ndim != 1 or mixture.numel() <= model.transforms.n_fft // 2:
        raise ValueError("Expected a mono utterance longer than half an FFT")
    if not torch.isfinite(mixture).all():
        raise ValueError("Nonfinite mixture")
    scale = (
        mixture.abs().max().clamp_min(1e-8)
        if model.normalize_by_mix
        else mixture.new_tensor(1.0)
    )
    spectrum = model.transforms.stft((mixture / scale)[None, None])
    clean, _, reverb = model.rec_rir(model.transforms.preprocess(spectrum))
    result = {}
    for name, values in (("clean", clean), ("reverb", reverb)):
        spectrum = model._postprocess_multi(values)[0]
        waveform = (
            torch.istft(
                spectrum,
                n_fft=model.transforms.n_fft,
                hop_length=model.transforms.hop_len,
                win_length=model.transforms.win_len,
                window=model.transforms.swin.to(spectrum.device),
                onesided=True,
                length=mixture.numel(),
            )
            * scale
        )
        if waveform.shape != (2, mixture.numel()) or not torch.isfinite(waveform).all():
            raise ValueError(f"Invalid {name} speech estimates")
        result[name] = waveform
    return result
