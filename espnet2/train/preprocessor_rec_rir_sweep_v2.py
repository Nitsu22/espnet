"""Paired physical RIRs for direct-to-reverberant sweep supervision."""

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor


class RecRIRSweepV2Preprocessor(AbsPreprocessor):
    def __init__(self, train, speech_segment=64000, rir_length=32000,
                 force_single_channel=True, **kwargs):
        self.train = train
        self.speech_segment = speech_segment
        self.rir_length = int(rir_length)
        if not force_single_channel or self.rir_length <= 0:
            raise ValueError("Expected left channel and a positive RIR length")

    def __call__(self, uid, data):
        def mono(value):
            value = np.asarray(value, dtype=np.float64)
            if value.ndim == 2:
                value = value[:, 0]
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError(f"{uid}: invalid waveform")
            return value

        speech = mono(data["speech_mix"])
        if self.train and self.speech_segment and len(speech) > self.speech_segment:
            start = np.random.randint(len(speech) - self.speech_segment + 1)
            for _ in range(10):
                if np.any(speech[start:start + self.speech_segment] != 0):
                    break
                start = np.random.randint(len(speech) - self.speech_segment + 1)
            speech = speech[start:start + self.speech_segment]
        direct, reverb = mono(data["rir_direct"]), mono(data["rir_ref"])
        scale = np.max(np.abs(direct))
        if scale == 0 or not np.any(reverb):
            raise ValueError(f"{uid}: zero RIR")
        result = {"speech_mix": speech}
        for name, rir in (("rir_direct", direct), ("rir_ref", reverb)):
            # Never silently remove a measured/simulated tail. No peak shift;
            # a shared gain preserves the direct-to-reverberant relation.
            if np.any(rir[self.rir_length:] != 0):
                raise ValueError(f"{uid}: {name} exceeds rir_length; increase it")
            target = np.zeros(self.rir_length, dtype=np.float64)
            count = min(len(rir), self.rir_length)
            target[:count] = rir[:count] / scale
            result[name] = target
        return result
