import math
from typing import Dict, Optional

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor


def _as_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


class DARASPreprocessor(AbsPreprocessor):
    """Preprocessor for single-channel DARAS RIR estimation."""

    def __init__(
        self,
        train: bool,
        speech_name: str = "speech_mix",
        rir_ref_name: str = "rir_ref",
        room_param_path_name: str = "room_param_path",
        room_param_name: str = "room_param",
        speech_sample_rate: int = 8000,
        rir_sample_rate: int = 8000,
        rir_length: int = 8192,
        force_single_channel: bool = True,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        rir_normalize: str = "peak",
        bp_window_ms: float = 20.0,
        bp_threshold: float = 0.95,
        bp_min_ms: float = 2.0,
        bp_max_ms: float = 250.0,
        **kwargs,
    ):
        if rir_length <= 0:
            raise ValueError("rir_length must be a positive integer")
        if rir_normalize not in ("none", "peak", "rms"):
            raise ValueError(f"Unsupported rir_normalize: {rir_normalize}")
        self.train = train
        self.speech_name = speech_name
        self.rir_ref_name = rir_ref_name
        self.room_param_path_name = room_param_path_name
        self.room_param_name = room_param_name
        self.speech_sample_rate = int(speech_sample_rate)
        self.rir_sample_rate = int(rir_sample_rate)
        self.rir_length = int(rir_length)
        self.force_single_channel = force_single_channel
        self.speech_segment = speech_segment
        self.avoid_allzero_segment = avoid_allzero_segment
        self.rir_normalize = rir_normalize
        self.bp_window_ms = float(bp_window_ms)
        self.bp_threshold = float(bp_threshold)
        self.bp_min_ms = float(bp_min_ms)
        self.bp_max_ms = float(bp_max_ms)

    def __call__(self, uid: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        speech = self._process_speech(np.asarray(data[self.speech_name]))
        if self.train and self.speech_segment is not None:
            speech = self._crop_speech(speech, int(self.speech_segment))
        data[self.speech_name] = speech.astype(np.float64, copy=False)

        rir = self._process_rir(np.asarray(data[self.rir_ref_name]))
        data[self.rir_ref_name] = rir.astype(np.float64, copy=False)

        room_param_path = _as_str(data.pop(self.room_param_path_name))
        volume, t60 = self._load_room_params(room_param_path)
        bp = self._estimate_boundary_point(rir)
        data[self.room_param_name] = np.asarray(
            [
                math.log10(max(volume, 1.0e-8)),
                math.log10(max(t60, 1.0e-8)),
                math.log10(max(bp, 1.0)),
            ],
            dtype=np.float64,
        )
        return data

    def _process_speech(self, speech: np.ndarray) -> np.ndarray:
        if speech.ndim == 2 and self.force_single_channel:
            speech = speech[:, 0]
        if speech.ndim > 2:
            raise ValueError(f"Unsupported speech shape: {speech.shape}")
        return speech

    def _crop_speech(self, speech: np.ndarray, speech_segment: int) -> np.ndarray:
        if speech.shape[0] <= speech_segment:
            return speech
        last_start = speech.shape[0] - speech_segment
        start = np.random.randint(0, last_start + 1)
        if self.avoid_allzero_segment:
            for _ in range(10):
                segment = speech[start : start + speech_segment]
                if np.any(segment != 0):
                    break
                start = np.random.randint(0, last_start + 1)
        return speech[start : start + speech_segment]

    def _process_rir(self, rir: np.ndarray) -> np.ndarray:
        if rir.ndim == 2 and self.force_single_channel:
            rir = rir[:, 0]
        if rir.ndim > 2:
            raise ValueError(f"Unsupported RIR shape: {rir.shape}")
        rir = np.asarray(rir, dtype=np.float64).reshape(-1)
        if rir.shape[0] >= self.rir_length:
            rir = rir[: self.rir_length]
        else:
            rir = np.pad(rir, (0, self.rir_length - rir.shape[0]))
        return self._normalize_rir(rir)

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

    @staticmethod
    def _load_room_params(path: str):
        with np.load(path, allow_pickle=True) as z:
            room_dim = np.asarray(z["room_dim"], dtype=np.float64).reshape(-1)
            if room_dim.size < 3:
                raise ValueError(f"room_dim must have 3 elements: {path}")
            t60 = float(np.asarray(z["T60"]).item())
        volume = float(np.prod(room_dim[:3]))
        return volume, t60

    def _estimate_boundary_point(self, rir: np.ndarray) -> float:
        x = np.asarray(rir, dtype=np.float64).reshape(-1)
        if not np.any(np.abs(x) > 0):
            return max(1.0, self.rir_sample_rate * 0.05)

        win = max(3, int(round(self.rir_sample_rate * self.bp_window_ms / 1000.0)))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float64)
        power = np.convolve(x**2, kernel / win, mode="same")
        std = np.sqrt(np.maximum(power, 1.0e-12))
        indicator = (np.abs(x) > std).astype(np.float64)
        denom = math.erfc(1.0 / math.sqrt(2.0))
        ned = np.convolve(indicator, kernel / win, mode="same") / max(denom, 1.0e-12)
        ned = np.clip(ned, 0.0, 1.0)

        peak = int(np.argmax(np.abs(x)))
        lo = max(peak + 1, int(round(self.rir_sample_rate * self.bp_min_ms / 1000.0)))
        hi = min(x.shape[0] - 1, int(round(self.rir_sample_rate * self.bp_max_ms / 1000.0)))
        if hi <= lo:
            return float(min(max(lo, 1), x.shape[0] - 1))
        idxs = np.flatnonzero(ned[lo:hi] >= self.bp_threshold)
        if idxs.size:
            return float(lo + int(idxs[0]))
        return float(min(max(peak + int(round(0.05 * self.rir_sample_rate)), 1), hi))
