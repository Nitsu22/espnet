from typing import Dict, Union

import numpy as np
from typeguard import typechecked

from espnet2.train.preprocessor import EnhPreprocessor


LEFT_CH_IND = 0


class EnhRIRPreprocessor(EnhPreprocessor):
    """Enhancement preprocessor that adds oracle WHAMR RIR input."""

    def __init__(
        self,
        train: bool,
        rir_ref_name: str = "rir_ref",
        rir_mic_index: int = LEFT_CH_IND,
        **kwargs,
    ):
        super().__init__(train=train, **kwargs)

        if int(rir_mic_index) != LEFT_CH_IND:
            raise ValueError("enh_rir currently supports LEFT_CH_IND = 0 only")

        self.rir_ref_name = rir_ref_name
        self.rir_mic_index = int(rir_mic_index)

    @typechecked
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        rir_ref = data.pop(self.rir_ref_name, None)
        if rir_ref is None:
            raise KeyError(f"{self.rir_ref_name} is required for enh_rir: {uid}")

        data = super().__call__(uid, data)
        data[self.rir_ref_name] = self._format_rir_ref(rir_ref, uid)
        return data

    def _format_rir_ref(self, rir_ref: np.ndarray, uid: str) -> np.ndarray:
        rir = np.asarray(rir_ref, dtype=np.float64)

        if rir.ndim == 3:
            # variable_columns_sound over two 2ch RIR wavs gives
            # [num_spk, time, num_mic]. Select one microphone per speaker
            # to match the current RIR-CMHA separator contract.
            if rir.shape[0] < self.num_spk:
                raise ValueError(
                    f"RIR source count {rir.shape[0]} is smaller than num_spk "
                    f"{self.num_spk}: {uid}"
                )
            if self.rir_mic_index >= rir.shape[2]:
                raise ValueError(
                    f"rir_mic_index={self.rir_mic_index} but RIR has "
                    f"{rir.shape[2]} mics: {uid}"
                )
            rir = rir[: self.num_spk, :, self.rir_mic_index]
            rir = np.moveaxis(rir, 0, -1)
        elif rir.ndim == 2:
            if rir.shape[0] == self.num_spk and rir.shape[1] != self.num_spk:
                rir = rir.T
            elif rir.shape[1] >= self.num_spk:
                rir = rir[:, : self.num_spk]
            else:
                raise ValueError(
                    f"RIR must include num_spk={self.num_spk} channels: "
                    f"{rir.shape}, uid={uid}"
                )
        else:
            raise ValueError(
                f"RIR must be [num_spk, time, mic], [num_spk, time], "
                f"or [time, num_spk], but got {rir.shape}: {uid}"
            )

        return np.nan_to_num(rir, copy=False).astype(np.float64, copy=False)
