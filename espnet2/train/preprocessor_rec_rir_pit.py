from typing import Dict, List, Optional

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor


class RecRIRPITPreprocessor(AbsPreprocessor):
    """Preprocessor for two-source Rec-RIR PIT speech tuples."""

    def __init__(
        self,
        train: bool,
        speech_name: str = "speech_mix",
        num_spk: int = 2,
        speech_direct_prefix: str = "speech_direct",
        speech_reverb_prefix: str = "speech_reverb",
        speech_sample_rate: int = 8000,
        force_single_channel: bool = True,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        **kwargs,
    ):
        if int(num_spk) != 2:
            raise ValueError("RecRIRPITPreprocessor currently supports num_spk=2")
        self.train = train
        self.speech_name = speech_name
        self.num_spk = int(num_spk)
        self.speech_direct_names = [
            f"{speech_direct_prefix}{idx + 1}" for idx in range(self.num_spk)
        ]
        self.speech_reverb_names = [
            f"{speech_reverb_prefix}{idx + 1}" for idx in range(self.num_spk)
        ]
        self.speech_sample_rate = int(speech_sample_rate)
        self.force_single_channel = force_single_channel
        self.speech_segment = speech_segment
        self.avoid_allzero_segment = avoid_allzero_segment

    def __call__(self, uid: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        names = [self.speech_name] + self.speech_direct_names + self.speech_reverb_names
        signals = [self._process_speech(np.asarray(data[name])) for name in names]

        common_length = min(signal.shape[0] for signal in signals)
        signals = [signal[:common_length] for signal in signals]

        if self.train and self.speech_segment is not None:
            signals = self._crop_speech_list(signals, int(self.speech_segment))

        for name, signal in zip(names, signals):
            data[name] = signal.astype(np.float64, copy=False)
        data.pop("room_param_path", None)
        return data

    def _process_speech(self, speech: np.ndarray) -> np.ndarray:
        if speech.ndim == 2 and self.force_single_channel:
            speech = speech[:, 0]
        if speech.ndim > 2:
            raise ValueError(f"Unsupported speech shape: {speech.shape}")
        return speech

    def _crop_speech_list(
        self,
        signals: List[np.ndarray],
        speech_segment: int,
    ) -> List[np.ndarray]:
        if signals[0].shape[0] <= speech_segment:
            return signals
        last_start = signals[0].shape[0] - speech_segment
        start = np.random.randint(0, last_start + 1)
        if self.avoid_allzero_segment:
            for _ in range(10):
                segment = signals[0][start : start + speech_segment]
                if np.any(segment != 0):
                    break
                start = np.random.randint(0, last_start + 1)
        end = start + speech_segment
        return [signal[start:end] for signal in signals]
