import math
from typing import Dict, Optional

import numpy as np
from scipy.signal import resample_poly

from espnet2.train.preprocessor import AbsPreprocessor


def _as_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


class RIRPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        speech_name: str = "speech_mix",
        rir_path_name: str = "rir_path",
        room_param_path_name: str = "room_param_path",
        rir_ref_name: str = "rir_ref",
        speech_sample_rate: int = 8000,
        rir_sample_rate: int = 8000,
        rir_length: Optional[int] = None,
        rir_target: str = "reverberant",
        rir_source_index: int = 0,
        rir_mic_index: int = 0,
        predict_all_sources: bool = False,
        predict_all_mics: bool = False,
        force_single_channel: bool = True,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        rir_normalize: str = "none",
        rir_crop_mode: str = "right",
        rir_pad_mode: str = "right",
    ):
        if rir_length is None:
            raise ValueError("rir_length must be set in preprocessor_conf")
        if rir_length <= 0:
            raise ValueError("rir_length must be a positive integer")
        if rir_target not in ("reverberant", "anechoic"):
            raise ValueError(f"Unsupported rir_target: {rir_target}")
        if rir_normalize not in ("none", "peak", "rms"):
            raise ValueError(f"Unsupported rir_normalize: {rir_normalize}")
        if rir_crop_mode not in ("right", "center", "random"):
            raise ValueError(f"Unsupported rir_crop_mode: {rir_crop_mode}")
        if rir_pad_mode not in ("right", "center"):
            raise ValueError(f"Unsupported rir_pad_mode: {rir_pad_mode}")

        self.train = train
        self.speech_name = speech_name
        self.rir_path_name = rir_path_name
        self.room_param_path_name = room_param_path_name
        self.rir_ref_name = rir_ref_name
        self.speech_sample_rate = int(speech_sample_rate)
        self.rir_sample_rate = int(rir_sample_rate)
        self.rir_length = int(rir_length)
        self.rir_target = rir_target
        self.rir_source_index = int(rir_source_index)
        self.rir_mic_index = int(rir_mic_index)
        self.predict_all_sources = predict_all_sources
        self.predict_all_mics = predict_all_mics
        self.force_single_channel = force_single_channel
        self.speech_segment = speech_segment
        self.avoid_allzero_segment = avoid_allzero_segment
        self.rir_normalize = rir_normalize
        self.rir_crop_mode = rir_crop_mode
        self.rir_pad_mode = rir_pad_mode

    def __call__(self, uid: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        speech = np.asarray(data[self.speech_name])
        speech = self._process_speech(speech)
        data[self.speech_name] = speech

        rir_path = _as_str(data.pop(self.rir_path_name))
        data.pop(self.room_param_path_name, None)
        rir = self._load_rir(rir_path)
        data[self.rir_ref_name] = rir
        return data

    def _process_speech(self, speech: np.ndarray) -> np.ndarray:
        if speech.ndim == 2 and self.force_single_channel:
            speech = speech[:, 0]
        if speech.ndim > 2:
            raise ValueError(f"Unsupported speech shape: {speech.shape}")
        if self.train and self.speech_segment is not None:
            speech = self._crop_speech(speech, int(self.speech_segment))
        return speech.astype(np.float64, copy=False)

    def _crop_speech(self, speech: np.ndarray, speech_segment: int) -> np.ndarray:
        if speech.shape[0] <= speech_segment:
            return speech
        last_start = speech.shape[0] - speech_segment
        start = np.random.randint(0, last_start + 1)
        if self.avoid_allzero_segment:
            for _ in range(10):
                segment = speech[start : start + speech_segment]
                if np.any(segment != 0):
                    return segment
                start = np.random.randint(0, last_start + 1)
        return speech[start : start + speech_segment]

    def _load_rir(self, rir_path: str) -> np.ndarray:
        with np.load(rir_path, allow_pickle=True) as npz:
            key = f"rir_{self.rir_target}"
            if key not in npz:
                raise KeyError(f"{key} is not found in {rir_path}")
            rir = np.asarray(npz[key], dtype=np.float64)
            rir_fs = int(np.asarray(npz["fs"]).item()) if "fs" in npz else 16000

        if rir.ndim != 3:
            raise ValueError(f"RIR must have shape [source, mic, time]: {rir.shape}")

        rir = self._select_rir(rir)
        rir = np.moveaxis(rir, -1, 0)
        if rir_fs != self.rir_sample_rate:
            rir = self._resample_rir(rir, rir_fs)
        rir = self._fix_length(rir)
        rir = self._normalize_rir(rir)

        if rir.shape[1:] == (1, 1):
            rir = rir[:, 0, 0]
        elif rir.shape[2] == 1:
            rir = rir[:, :, 0]
        elif rir.shape[1] == 1:
            rir = rir[:, 0, :]

        return rir.astype(np.float64, copy=False)

    def _select_rir(self, rir: np.ndarray) -> np.ndarray:
        num_sources, num_mics, _ = rir.shape
        if self.predict_all_sources:
            source_slice = slice(None)
        else:
            if self.rir_source_index >= num_sources:
                raise IndexError(
                    f"rir_source_index={self.rir_source_index} but "
                    f"num_sources={num_sources}"
                )
            source_slice = slice(self.rir_source_index, self.rir_source_index + 1)

        if self.predict_all_mics:
            mic_slice = slice(None)
        else:
            if self.rir_mic_index >= num_mics:
                raise IndexError(
                    f"rir_mic_index={self.rir_mic_index} but num_mics={num_mics}"
                )
            mic_slice = slice(self.rir_mic_index, self.rir_mic_index + 1)

        return rir[source_slice, mic_slice, :]

    def _resample_rir(self, rir: np.ndarray, rir_fs: int) -> np.ndarray:
        gcd = math.gcd(rir_fs, self.rir_sample_rate)
        up = self.rir_sample_rate // gcd
        down = rir_fs // gcd
        return resample_poly(rir, up, down, axis=0)

    def _fix_length(self, rir: np.ndarray) -> np.ndarray:
        length = rir.shape[0]
        if length > self.rir_length:
            if self.rir_crop_mode == "right":
                start = 0
            elif self.rir_crop_mode == "center":
                start = (length - self.rir_length) // 2
            else:
                start = np.random.randint(0, length - self.rir_length + 1)
            return rir[start : start + self.rir_length]

        if length < self.rir_length:
            pad = self.rir_length - length
            if self.rir_pad_mode == "right":
                before, after = 0, pad
            else:
                before = pad // 2
                after = pad - before
            return np.pad(rir, [(before, after), (0, 0), (0, 0)])

        return rir

    def _normalize_rir(self, rir: np.ndarray) -> np.ndarray:
        if self.rir_normalize == "none":
            return rir
        if self.rir_normalize == "peak":
            denom = np.max(np.abs(rir))
        else:
            denom = np.sqrt(np.mean(rir**2))
        if denom > 0:
            rir = rir / denom
        return rir
