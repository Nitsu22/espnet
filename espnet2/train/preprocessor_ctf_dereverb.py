"""Aligned target crops with real past/future context for CTF convolution."""

import numpy as np

from espnet2.train.preprocessor_enh_ctf import EnhCTFPreprocessor


class CTFDereverbPreprocessor(EnhCTFPreprocessor):
    def __init__(
        self, train, ctf_taps=128, ctf_n_fft=256, ctf_hop_length=64,
        speech_segment=32000, **kwargs,
    ):
        if min(ctf_taps, ctf_n_fft, ctf_hop_length) <= 0:
            raise ValueError("CTF/STFT dimensions must be positive")
        if ctf_n_fft % ctf_hop_length:
            raise ValueError("Context must align with the STFT hop")
        if speech_segment is not None and speech_segment <= 1:
            raise ValueError("The target crop must contain at least two samples")
        super().__init__(train=train, speech_segment=None, **kwargs)
        if self.speech_volume_normalize is not None:
            raise ValueError("Use common mixture-derived model normalization")
        self.target_samples = speech_segment
        self.hop = ctf_hop_length
        self.past = (ctf_taps - 1) * ctf_hop_length + ctf_n_fft
        self.future = ctf_n_fft

    def __call__(self, uid, data):
        # Parent applies channel selection/common peak scaling to ALL branches;
        # its own random cropping is disabled so physical context is retained.
        data = super().__call__(uid, data)
        data.pop("reverb_crop_start", None)
        total = data[self.speech_name].shape[0]
        start, end = 0, total
        if self.train and self.target_samples and total > self.target_samples:
            max_start = (total - self.target_samples) // self.hop
            refs = [data[f"speech_ref{s+1}"] for s in range(self.num_spk)]
            for _ in range(10):
                start = int(np.random.randint(max_start + 1)) * self.hop
                end = start + self.target_samples
                if not self.avoid_allzero_segment or all(
                    np.any(x[start:end]) for x in refs
                ):
                    break
        left = max(0, start - self.past)
        right = min(total, end + self.future)
        self._apply_to_all_signals(data, lambda x: x[left:right], self.num_spk)
        for name, value in (
            ("ctf_loss_start", start - left), ("ctf_loss_end", end - left),
            ("ctf_crop_start", left), ("ctf_crop_end", right),
            ("ctf_original_length", total),
        ):
            data[name] = np.array([value], dtype=np.int64)
        return data
