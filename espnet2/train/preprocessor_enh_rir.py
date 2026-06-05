import math
from typing import Dict, Optional, Union

import numpy as np
from scipy.signal import resample_poly
from typeguard import typechecked

from espnet2.train.preprocessor import EnhPreprocessor


LEFT_CH_IND = 0


def _as_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


class EnhRIRPreprocessor(EnhPreprocessor):
    """Enhancement preprocessor that adds oracle WHAMR RIR input."""

    def __init__(
        self,
        train: bool,
        rir_path_name: str = "rir_path",
        rir_ref_name: str = "rir_ref",
        rir_target: str = "reverberant",
        rir_mic_index: int = LEFT_CH_IND,
        **kwargs,
    ):
        super().__init__(train=train, **kwargs)

        if rir_target not in ("reverberant", "anechoic"):
            raise ValueError(f"Unsupported rir_target: {rir_target}")
        if int(rir_mic_index) != LEFT_CH_IND:
            raise ValueError("enh_rir currently supports LEFT_CH_IND = 0 only")

        self.rir_path_name = rir_path_name
        self.rir_ref_name = rir_ref_name
        self.rir_target = rir_target
        self.rir_mic_index = int(rir_mic_index)

    @typechecked
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.rir_path_name not in data:
            raise KeyError(f"{self.rir_path_name} is required for enh_rir: {uid}")

        rir_path = _as_str(data.pop(self.rir_path_name))
        data = super().__call__(uid, data)
        data[self.rir_ref_name] = self._load_rir(rir_path, uid)
        return data

    def _load_rir(self, rir_path: str, uid: str) -> np.ndarray:
        with np.load(rir_path, allow_pickle=True) as npz:
            key = f"rir_{self.rir_target}"
            if key not in npz:
                raise KeyError(f"{key} is not found in {rir_path}")
            rir = np.asarray(npz[key], dtype=np.float64)
            rir_fs = int(np.asarray(npz["fs"]).item()) if "fs" in npz else 16000

        if rir.ndim != 3:
            raise ValueError(f"RIR must have shape [source, mic, time]: {rir.shape}")
        if rir.shape[0] < self.num_spk:
            raise ValueError(
                f"RIR source count {rir.shape[0]} is smaller than num_spk "
                f"{self.num_spk}: {uid}"
            )
        if self.rir_mic_index >= rir.shape[1]:
            raise ValueError(
                f"rir_mic_index={self.rir_mic_index} but RIR has "
                f"{rir.shape[1]} mics: {uid}"
            )

        rir = rir[: self.num_spk, self.rir_mic_index, :]
        if rir_fs != self.sample_rate:
            rir = self._resample_rir(rir, rir_fs)

        # Collate pads the first axis, so keep time first and speakers second.
        return np.moveaxis(rir, -1, 0).astype(np.float64, copy=False)

    def _resample_rir(self, rir: np.ndarray, rir_fs: int) -> np.ndarray:
        gcd = math.gcd(int(rir_fs), int(self.sample_rate))
        up = int(self.sample_rate) // gcd
        down = int(rir_fs) // gcd
        return resample_poly(rir, up, down, axis=-1)
