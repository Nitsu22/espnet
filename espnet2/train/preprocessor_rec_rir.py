from typing import Dict, Optional

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor


class RecRIRPreprocessor(AbsPreprocessor):
    """Preprocessor for single-source Rec-RIR speech triples."""

    def __init__(
        self,
        train: bool,
        speech_name: str = "speech_mix",
        speech_direct_name: str = "speech_direct",
        speech_reverb_name: str = "speech_reverb",
        speech_sample_rate: int = 8000,
        force_single_channel: bool = True,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        **kwargs,
    ):
        self.train = train
        self.speech_name = speech_name
        self.speech_direct_name = speech_direct_name
        self.speech_reverb_name = speech_reverb_name
        self.speech_sample_rate = int(speech_sample_rate)
        self.force_single_channel = force_single_channel
        self.speech_segment = speech_segment
        self.avoid_allzero_segment = avoid_allzero_segment

    def __call__(self, uid: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mix = self._process_speech(np.asarray(data[self.speech_name]))
        direct = self._process_speech(np.asarray(data[self.speech_direct_name]))
        reverb = self._process_speech(np.asarray(data[self.speech_reverb_name]))

        common_length = min(mix.shape[0], direct.shape[0], reverb.shape[0])
        mix = mix[:common_length]
        direct = direct[:common_length]
        reverb = reverb[:common_length]

        if self.train and self.speech_segment is not None:
            mix, direct, reverb = self._crop_speech_triple(
                mix, direct, reverb, int(self.speech_segment)
            )

        data[self.speech_name] = mix.astype(np.float64, copy=False)
        data[self.speech_direct_name] = direct.astype(np.float64, copy=False)
        data[self.speech_reverb_name] = reverb.astype(np.float64, copy=False)
        data.pop("room_param_path", None)
        return data

    def _process_speech(self, speech: np.ndarray) -> np.ndarray:
        if speech.ndim == 2 and self.force_single_channel:
            speech = speech[:, 0]
        if speech.ndim > 2:
            raise ValueError(f"Unsupported speech shape: {speech.shape}")
        return speech

    def _crop_speech_triple(
        self,
        mix: np.ndarray,
        direct: np.ndarray,
        reverb: np.ndarray,
        speech_segment: int,
    ):
        if mix.shape[0] <= speech_segment:
            return mix, direct, reverb
        last_start = mix.shape[0] - speech_segment
        start = np.random.randint(0, last_start + 1)
        if self.avoid_allzero_segment:
            for _ in range(10):
                segment = mix[start : start + speech_segment]
                if np.any(segment != 0):
                    break
                start = np.random.randint(0, last_start + 1)
        end = start + speech_segment
        return mix[start:end], direct[start:end], reverb[start:end]

