"""Single-speaker input/RIR preprocessing without speech targets."""
import numpy as np
from espnet2.train.preprocessor import AbsPreprocessor


class RecRIRSweepPreprocessor(AbsPreprocessor):
    def __init__(self, train, speech_segment=64000, rir_length=32000,
                 rir_peak_index=40, force_single_channel=True, **kwargs):
        self.train = train
        self.speech_segment = speech_segment
        self.rir_length = int(rir_length)
        self.rir_peak_index = int(rir_peak_index)
        if not force_single_channel or not 0 <= self.rir_peak_index < self.rir_length:
            raise ValueError('Expected left-channel input and a valid RIR peak index')

    def __call__(self, uid, data):
        def mono(x):
            x = np.asarray(x)
            if x.ndim == 2:
                x = x[:, 0]
            if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
                raise ValueError(f'{uid}: invalid waveform')
            return x

        speech = mono(data['speech_mix'])
        if self.train and self.speech_segment and len(speech) > self.speech_segment:
            start = np.random.randint(len(speech) - self.speech_segment + 1)
            for _ in range(10):
                if np.any(speech[start:start + self.speech_segment] != 0):
                    break
                start = np.random.randint(len(speech) - self.speech_segment + 1)
            speech = speech[start:start + self.speech_segment]
        rir = mono(data['rir_ref'])
        peak = int(np.argmax(np.abs(rir)))
        amplitude = abs(rir[peak])
        if amplitude == 0:
            raise ValueError(f'{uid}: zero RIR')
        # Align before cropping, retaining the absolute peak even with long delay.
        start = peak - self.rir_peak_index
        target = np.zeros(self.rir_length, dtype=np.float64)
        source_start, dest_start = max(0, start), max(0, -start)
        size = min(len(rir) - source_start, len(target) - dest_start)
        target[dest_start:dest_start + size] = rir[source_start:source_start + size] / amplitude
        return {'speech_mix': speech.astype(np.float64), 'rir_ref': target}
