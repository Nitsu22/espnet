"""Keep clean/reverberant teachers aligned with the baseline's input crop."""

import numpy as np

from espnet2.train.preprocessor import EnhPreprocessor


class EnhCTFPreprocessor(EnhPreprocessor):
    def __init__(self, train, **kwargs):
        super().__init__(train=train, **kwargs)
        if self.rirs is not None or self.noises is not None:
            raise ValueError("Joint CTF training uses existing paired WHAMR audio")
        if self.channel_reordering or self.use_reverberant_ref:
            raise ValueError("Joint CTF training requires fixed direct-speech teachers")
        self._crop_start = 0

    def _apply_to_all_signals(self, data_dict, func, num_spk):
        super()._apply_to_all_signals(data_dict, func, num_spk)
        for s in range(1, num_spk + 1):
            name = f"speech_reverb{s}"
            if name in data_dict:
                data_dict[name] = func(data_dict[name])

    def _random_crop_range(self, *args, **kwargs):
        start, end = super()._random_crop_range(*args, **kwargs)
        self._crop_start = start
        return start, end

    def __call__(self, uid, data):
        self._crop_start = 0
        for s in range(1, self.num_spk + 1):
            name = f"speech_reverb{s}"
            if self.train and name not in data:
                raise ValueError(f"{name} is required for {uid}")
            if name in data and data[name].shape[0] != data[self.speech_name].shape[0]:
                raise ValueError(f"Unaligned {name} for {uid}")
        data = super().__call__(uid, data)
        if any(f"speech_reverb{s}" in data for s in range(1, self.num_spk + 1)):
            data["reverb_crop_start"] = np.array([self._crop_start], dtype=np.int64)
        return data
