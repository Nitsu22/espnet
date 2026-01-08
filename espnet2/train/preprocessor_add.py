"""Additional preprocessors for ESPnet2."""

import logging
import re
from typing import Dict, List, Optional, Union

import numpy as np
from typeguard import typechecked

from espnet2.train.preprocessor import CommonPreprocessor


class SePreprocessor(CommonPreprocessor):
    """Preprocessor for Spatial Encoder (SE) task."""

    def __init__(
        self,
        train: bool,
        speech_volume_normalize: Optional[str] = None,
        speech_name: str = "speech_mix",
        sample_rate: int = 8000,
        force_single_channel: bool = True,
        categories: Optional[List] = None,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
    ):
        super().__init__(
            train=train,
            token_type=None,
            token_list=None,
            bpemodel=None,
            text_cleaner=None,
            g2p_type=None,
            unk_symbol="<unk>",
            space_symbol="<space>",
            non_linguistic_symbols=None,
            delimiter=None,
            rir_scp=None,
            rir_apply_prob=0.0,
            noise_scp=None,
            noise_apply_prob=0.0,
            noise_db_range="3_10",
            short_noise_thres=0.5,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            fs=sample_rate,
            data_aug_effects=None,
            data_aug_num=[1, 1],
            data_aug_prob=0.0,
        )
        self.sample_rate = sample_rate
        # Whether to always convert the signals to single-channel
        self.force_single_channel = force_single_channel

        # If specified, the audios will be chomped to the specified length
        self.speech_segment = speech_segment
        # Only used when `speech_segment` is specified.
        # If True, make sure all chomped segments are not all-zero.
        self.avoid_allzero_segment = avoid_allzero_segment

        # Map each category into a unique integer
        self.categories = {}
        if categories:
            count = 0
            for c in categories:
                if c not in self.categories:
                    self.categories[c] = count
                    count += 1

        if self.speech_volume_normalize is not None:
            sps = speech_volume_normalize.split("_")
            if len(sps) == 1:
                self.volume_low, self.volume_high = float(sps[0])
            elif len(sps) == 2:
                self.volume_low, self.volume_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error for --speech_volume_normalize: "
                    f"'{speech_volume_normalize}'"
                )

    def __basic_str__(self):
        msg = ""
        if self.force_single_channel:
            msg += f", force_single_channel={self.force_single_channel}"
        if self.speech_volume_normalize:
            msg += f", speech_volume_normalize={self.speech_volume_normalize}"
        if self.speech_segment:
            msg += f", speech_segment={self.speech_segment}"
            msg += f", avoid_allzero_segment={self.avoid_allzero_segment}"
        if self.categories:
            if len(self.categories) <= 10:
                msg += f", categories={self.categories}"
            else:
                msg += f", num_category={len(self.categories)}"
        return msg

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        msg = f"{name}(train={self.train}"
        msg += self.__basic_str__()
        return msg + ")"

    def _ensure_2d(self, signal):
        if isinstance(signal, tuple):
            return tuple(self._ensure_2d(sig) for sig in signal)
        elif isinstance(signal, list):
            return [self._ensure_2d(sig) for sig in signal]
        else:
            # (Nmic, Time)
            return signal[None, :] if signal.ndim == 1 else signal.T

    def _apply_to_speech_mix(self, data_dict, func):
        """Apply function to speech_mix only."""
        data_dict[self.speech_name] = func(data_dict[self.speech_name])

    def _random_crop_range(
        self, data_dict, tgt_length, uid=None, max_trials=10
    ):
        """Randomly crop the signals to the length `tgt_length`."""
        assert tgt_length > 0, tgt_length
        speech_mix = data_dict[self.speech_name]
        length = speech_mix.shape[0]
        
        if length <= tgt_length:
            if length < tgt_length:
                logging.warning(
                    f"The sample ({uid}) is not cropped due to its short length "
                    f"({length} < {tgt_length})."
                )
            return 0, length

        start = np.random.randint(0, length - tgt_length)
        count = 1
        if self.avoid_allzero_segment:
            # try to find a segment region that ensures the signal is non-allzero
            while True:
                segment = speech_mix[start : start + tgt_length]
                is_allzero = np.allclose(segment, 0.0)
                
                if not is_allzero:
                    break
                
                count += 1
                if count > max_trials:
                    logging.warning(
                        f"Can't find non-allzero segments for {uid}."
                    )
                    break
                if start > 0:
                    start = np.random.randint(0, start)
                else:
                    break
        return start, start + tgt_length

    @typechecked
    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:

        if self.speech_name not in data:
            return data

        # Check required data for training
        if self.train:
            if "speech_mix_mc" not in data:
                raise ValueError(
                    f"speech_mix_mc is required for training but not found in data for {uid}"
                )
            if "speech_mix_reverse_mc" not in data:
                raise ValueError(
                    f"speech_mix_reverse_mc is required for training but not found in data for {uid}"
                )

        # Add the category information (an integer) to `data`
        if not self.categories and "category" in data:
            raise ValueError(
                "categories must be set in the config file when utt2category files "
                "exist in the data directory (e.g., dump/raw/*/utt2category)"
            )

        # Add the sampling rate information (an integer) to `data`
        if "fs" in data:
            fs = int(data.pop("fs"))
            data["utt2fs"] = np.array([fs])
        else:
            fs = self.sample_rate

        if self.train:
            # Random cropping if speech_segment is specified
            # Same as EnhPreprocessor: crop in [Time] or [Time, Nmic] format
            if self.speech_segment is not None:
                speech_segment = self.speech_segment // self.sample_rate * fs
                start, end = self._random_crop_range(
                    data, speech_segment, uid=uid
                )
                # Apply same cropping range to all signals (same as EnhPreprocessor)
                self._apply_to_speech_mix(data, lambda x: x[start:end])
                data["speech_mix_mc"] = data["speech_mix_mc"][start:end]
                data["speech_mix_reverse_mc"] = data["speech_mix_reverse_mc"][start:end]

            # Process speech_mix (same as EnhPreprocessor with force_single_channel=True)
            # 1. _ensure_2d converts to [Nmic, Time] format
            speech_mix = self._ensure_2d(data[self.speech_name])  # [Nmic, Time]
            # Note: RIR, Noise, and data_aug processing are removed for SE task

            # 2. Convert to [Time, Nmic] format (same as EnhPreprocessor line 1403)
            data[self.speech_name] = speech_mix.T  # [Time, Nmic]
            # 3. Normalize if max abs > 1.0 (same as EnhPreprocessor lines 1404-1406)
            ma = np.max(np.abs(data[self.speech_name]))
            if ma > 1.0:
                self._apply_to_speech_mix(data, lambda x: x / ma)

            # 4. squeeze: only for speech_mix (same as EnhPreprocessor line 1408)
            self._apply_to_speech_mix(data, lambda x: x.squeeze())

            # Process speech_mix_mc (same as EnhPreprocessor with force_single_channel=False)
            # Note: squeeze() is applied regardless of force_single_channel in EnhPreprocessor (line 1408)
            # 1. _ensure_2d converts to [Nmic, Time] format
            speech_mix_mc = self._ensure_2d(data["speech_mix_mc"])  # [Nmic, Time]
            # 2. Convert to [Time, Nmic] format
            data["speech_mix_mc"] = speech_mix_mc.T  # [Time, Nmic]
            # 3. Normalize if max abs > 1.0
            ma_mc = np.max(np.abs(data["speech_mix_mc"]))
            if ma_mc > 1.0:
                data["speech_mix_mc"] = data["speech_mix_mc"] / ma_mc
            # 4. squeeze: same as EnhPreprocessor (applied regardless of force_single_channel)
            # For multi-channel [Time, Nmic] where Nmic > 1, squeeze has no effect
            data["speech_mix_mc"] = data["speech_mix_mc"].squeeze()

            # Process speech_mix_reverse_mc (same as speech_mix_mc)
            # 1. _ensure_2d converts to [Nmic, Time] format
            speech_mix_reverse_mc = self._ensure_2d(data["speech_mix_reverse_mc"])  # [Nmic, Time]
            # 2. Convert to [Time, Nmic] format
            data["speech_mix_reverse_mc"] = speech_mix_reverse_mc.T  # [Time, Nmic]
            # 3. Normalize if max abs > 1.0
            ma_rev = np.max(np.abs(data["speech_mix_reverse_mc"]))
            if ma_rev > 1.0:
                data["speech_mix_reverse_mc"] = data["speech_mix_reverse_mc"] / ma_rev
            # 4. squeeze: same as EnhPreprocessor (applied regardless of force_single_channel)
            # For multi-channel [Time, Nmic] where Nmic > 1, squeeze has no effect
            data["speech_mix_reverse_mc"] = data["speech_mix_reverse_mc"].squeeze()

        # Apply force_single_channel only to speech_mix (same as EnhPreprocessor lines 1410-1413)
        # Note: speech_mix_mc and speech_mix_reverse_mc are NOT affected by force_single_channel
        if self.force_single_channel:
            self._apply_to_speech_mix(
                data, lambda x: x if x.ndim == 1 else x[:, 0]
            )

        if self.speech_volume_normalize is not None:
            if self.train:
                volume_scale = np.random.uniform(self.volume_low, self.volume_high)
            else:
                # use a fixed scale to make it deterministic
                volume_scale = self.volume_low
            
            # Apply volume normalization individually to each signal
            ma = np.max(np.abs(data[self.speech_name]))
            self._apply_to_speech_mix(data, lambda x: x * volume_scale / ma)
            
            if self.train:
                ma_mc = np.max(np.abs(data["speech_mix_mc"]))
                data["speech_mix_mc"] = data["speech_mix_mc"] * volume_scale / ma_mc
                
                ma_rev = np.max(np.abs(data["speech_mix_reverse_mc"]))
                data["speech_mix_reverse_mc"] = data["speech_mix_reverse_mc"] * volume_scale / ma_rev

        if self.categories and "category" in data:
            category = data.pop("category")
            if not re.fullmatch(r"\d+ch.*", category):
                speech_mix = data[self.speech_name]
                nch = 1 if speech_mix.ndim == 1 else speech_mix.shape[-1]
                category = f"{nch}ch_" + category
            assert category in self.categories, category
            data["utt2category"] = np.array([self.categories[category]])

        return data

    @typechecked
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:

        data = self._speech_process(uid, data)
        data = self._text_process(data)
        return data
