from typing import Dict, List, Optional

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor


def _as_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


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
        use_sweep_target: bool = False,
        rir_ref_prefix: str = "rir_ref",
        rir_length: Optional[int] = None,
        canonicalize_rir_by_peak: bool = False,
        rir_peak_index: int = 20,
        load_t60: bool = False,
        room_param_path_name: str = "room_param_path",
        t60_name: str = "t60",
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
        self.use_sweep_target = bool(use_sweep_target)
        self.rir_ref_names = [
            f"{rir_ref_prefix}{idx + 1}" for idx in range(self.num_spk)
        ]
        self.rir_length = None if rir_length is None else int(rir_length)
        if self.use_sweep_target and (
            self.rir_length is None or self.rir_length <= 0
        ):
            raise ValueError(
                "rir_length must be a positive integer when "
                "use_sweep_target=true"
            )
        self.canonicalize_rir_by_peak = bool(canonicalize_rir_by_peak)
        self.rir_peak_index = int(rir_peak_index)
        if self.canonicalize_rir_by_peak:
            if not self.use_sweep_target:
                raise ValueError(
                    "canonicalize_rir_by_peak requires use_sweep_target=true"
                )
            assert self.rir_length is not None
            if not 0 <= self.rir_peak_index < self.rir_length:
                raise ValueError(
                    "rir_peak_index must be within the fixed RIR length: "
                    f"rir_peak_index={self.rir_peak_index}, "
                    f"rir_length={self.rir_length}"
                )
        self.load_t60 = bool(load_t60)
        self.room_param_path_name = room_param_path_name
        self.t60_name = t60_name

    def __call__(self, uid: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.use_sweep_target:
            self._prepare_sweep_example(data)
        else:
            self._prepare_speech_example(data)

        if self.load_t60:
            if self.room_param_path_name not in data:
                raise ValueError(
                    f"{uid}: {self.room_param_path_name} is required when "
                    "load_t60=true"
                )
            room_param_path = _as_str(data.pop(self.room_param_path_name))
            with np.load(room_param_path, allow_pickle=False) as room_param_npz:
                if "T60" not in room_param_npz:
                    raise KeyError(f"{uid}: T60 is missing from {room_param_path}")
                t60 = float(np.asarray(room_param_npz["T60"]).item())
            if not np.isfinite(t60) or t60 <= 0.0:
                raise ValueError(
                    f"{uid}: T60 must be finite and positive, got {t60}"
                )
            data[self.t60_name] = np.asarray([t60], dtype=np.float32)
        else:
            data.pop(self.room_param_path_name, None)
        return data

    def _prepare_speech_example(self, data: Dict[str, np.ndarray]) -> None:
        names = [self.speech_name] + self.speech_direct_names + self.speech_reverb_names
        signals = [self._process_speech(np.asarray(data[name])) for name in names]

        common_length = min(signal.shape[0] for signal in signals)
        signals = [signal[:common_length] for signal in signals]

        if self.train and self.speech_segment is not None:
            signals = self._crop_speech_list(signals, int(self.speech_segment))

        for name, signal in zip(names, signals):
            data[name] = signal.astype(np.float64, copy=False)

    def _prepare_sweep_example(self, data: Dict[str, np.ndarray]) -> None:
        speech = self._process_speech(np.asarray(data[self.speech_name]))
        if self.train and self.speech_segment is not None:
            speech = self._crop_speech_list(
                [speech], int(self.speech_segment)
            )[0]
        data[self.speech_name] = speech.astype(np.float64, copy=False)

        for name in self.rir_ref_names:
            rir = self._process_speech(np.asarray(data[name]))
            rir = self._fix_rir_length(rir)
            if self.canonicalize_rir_by_peak:
                rir = self._canonicalize_rir_by_peak(rir)
            data[name] = rir.astype(np.float64, copy=False)

    def _canonicalize_rir_by_peak(self, rir: np.ndarray) -> np.ndarray:
        if not np.isfinite(rir).all():
            raise ValueError("RIR must contain only finite values")

        peak_source_index = int(np.argmax(np.abs(rir)))
        peak_amplitude = float(np.abs(rir[peak_source_index]))
        if peak_amplitude == 0.0:
            raise ValueError("RIR must not be all-zero")

        shift = self.rir_peak_index - peak_source_index
        aligned = np.zeros_like(rir)
        if shift > 0:
            aligned[shift:] = rir[:-shift]
        elif shift < 0:
            aligned[:shift] = rir[-shift:]
        else:
            aligned = rir.copy()
        return aligned / peak_amplitude

    def _fix_rir_length(self, rir: np.ndarray) -> np.ndarray:
        assert self.rir_length is not None
        if rir.shape[0] > self.rir_length:
            return rir[: self.rir_length]
        if rir.shape[0] < self.rir_length:
            pad_width = [(0, self.rir_length - rir.shape[0])]
            pad_width.extend((0, 0) for _ in range(rir.ndim - 1))
            return np.pad(rir, pad_width)
        return rir

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
