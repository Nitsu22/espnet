"""Additional preprocessors for ESPnet2."""

import logging
import re
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.signal
from typeguard import typechecked

from espnet2.train.preprocessor import CommonPreprocessor, detect_non_silence, any_allzero


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


class EnhPreprocessorPlus(CommonPreprocessor):
    """Preprocessor for Speech Enhancement (Enh) task."""

    def __init__(
        self,
        train: bool,
        rir_scp: Optional[str] = None,
        rir_apply_prob: float = 1.0,
        noise_scp: Optional[str] = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 1,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = True,
        channel_reordering: bool = False,
        categories: Optional[List] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        flexible_numspk: bool = False,
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
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            fs=sample_rate,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
        )
        self.speech_ref_name_prefix = speech_ref_name_prefix
        self.noise_ref_name_prefix = noise_ref_name_prefix
        self.dereverb_ref_name_prefix = dereverb_ref_name_prefix
        self.use_reverberant_ref = use_reverberant_ref
        self.num_spk = num_spk
        self.num_noise_type = num_noise_type
        self.sample_rate = sample_rate
        self.rir_scp = rir_scp
        self.noise_scp = noise_scp
        self.noise_db_range = noise_db_range
        # Whether to always convert the signals to single-channel
        self.force_single_channel = force_single_channel
        # If True, randomly reorder the channels of the multi-channel signals
        self.channel_reordering = channel_reordering

        # If specified, the audios will be chomped to the specified length
        self.speech_segment = speech_segment
        # Only used when `speech_segment` is specified.
        # If True, make sure all chomped segments are not all-zero.
        self.avoid_allzero_segment = avoid_allzero_segment

        # If True, load variable numbers of speakers in each sample, and
        # self.num_spk is regarded as the maximum possible number of speakers
        self.flexible_numspk = flexible_numspk

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

        if (self.rirs is not None and self.rir_apply_prob > 0) or (
            self.noises is not None and self.noise_apply_prob > 0
        ):
            logging.warning(
                "Note: Please ensure the sampling rates of all data, including audios "
                f"and RIRs, are all equal to {self.sample_rate} Hz when applying "
                "dynamic mixing."
            )

    def __basic_str__(self):
        msg = f", num_spk={self.num_spk}"
        for key in (
            "force_single_channel",
            "channel_reordering",
            "speech_volume_normalize",
        ):
            if getattr(self, key):
                msg += f", {key}={getattr(self, key)}"
        if self.rirs is not None and self.rir_apply_prob > 0:
            msg += f", sample_rate={self.sample_rate}"
            msg += f", rir_scp={self.rir_scp}, rir_apply_prob={self.rir_apply_prob}"
            if self.use_reverberant_ref:
                msg += f", use_reverberant_ref={self.use_reverberant_ref}"
        if self.noises is not None and self.noise_apply_prob > 0:
            msg += f", noise_scp={self.noise_scp}"
            msg += f", noise_apply_prob={self.noise_apply_prob}"
            msg += f", noise_db_range={self.noise_db_range}"
        if self.data_aug and self.data_aug_prob > 0:
            msg += f", data_aug={self.data_aug}, data_aug_prob={self.data_aug_prob}"
        if self.speech_segment:
            msg += f", speech_segment={self.speech_segment}"
            msg += f", avoid_allzero_segment={self.avoid_allzero_segment}"
        if self.flexible_numspk:
            msg += f", flexible_numspk={self.flexible_numspk}"
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

    def _get_early_signal(self, speech, rir, power):
        predelay = 50  # milliseconds
        dt = np.argmax(rir, axis=1).min()
        et = dt + (predelay * self.sample_rate) // 1000
        rir_early = rir[:, :et]
        speech2 = scipy.signal.convolve(speech, rir_early, mode="full")[
            :, : speech.shape[1]
        ]
        # Reverse mean power to the original power
        power2 = (speech2[detect_non_silence(speech2)] ** 2).mean()
        speech2 = np.sqrt(power / max(power2, 1e-10)) * speech2
        return speech2

    def _apply_to_all_signals(self, data_dict, func, num_spk):
        data_dict[self.speech_name] = func(data_dict[self.speech_name])

        for n in range(self.num_noise_type):
            noise_name = self.noise_ref_name_prefix + str(n + 1)
            if noise_name in data_dict:
                data_dict[noise_name] = func(data_dict[noise_name])

        for spk in range(num_spk):
            speech_ref_name = self.speech_ref_name_prefix + str(spk + 1)
            if self.train or speech_ref_name in data_dict:
                data_dict[speech_ref_name] = func(data_dict[speech_ref_name])

            dereverb_ref_name = self.dereverb_ref_name_prefix + str(spk + 1)
            if dereverb_ref_name in data_dict:
                data_dict[dereverb_ref_name] = func(data_dict[dereverb_ref_name])

    def _random_crop_range(
        self, data_dict, num_spk, tgt_length, uid=None, max_trials=10
    ):
        # Randomly crop the signals to the length `tgt_length`
        assert tgt_length > 0, tgt_length
        speech_refs = [
            data_dict[self.speech_ref_name_prefix + str(spk + 1)]
            for spk in range(num_spk)
        ]
        length = speech_refs[0].shape[0]
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
            # try to find a segment region that ensures all references are non-allzero
            while any_allzero([sf[start : start + tgt_length] for sf in speech_refs]):
                count += 1
                if count > max_trials:
                    logging.warning(
                        f"Can't find non-allzero segments for all references in {uid}."
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

        speech_mix = data[self.speech_name]
        if speech_mix.ndim == 1:
            raise ValueError(
                f"speech_mix must be multi-channel (2D) but got 1D for {uid}"
            )
        if speech_mix.ndim == 2:
            if min(speech_mix.shape) <= 1:
                raise ValueError(
                    f"speech_mix must have >1 channel for {uid}; "
                    f"got shape {speech_mix.shape}"
                )
        else:
            raise ValueError(
                f"speech_mix must be 1D or 2D array; got shape {speech_mix.shape} for {uid}"
            )

        num_spk = self.num_spk

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

        sref_name = self.speech_ref_name_prefix + "1"
        if self.flexible_numspk and sref_name in data:
            # The number of speaker varies in each sample.
            # Different speaker signals are stacked in the first dimension.
            dref_name = self.dereverb_ref_name_prefix + "1"
            num_spk = len(data[sref_name])
            for i in range(2, self.num_spk + 1):
                data.pop(self.speech_ref_name_prefix + str(i), None)
                data.pop(self.dereverb_ref_name_prefix + str(i), None)
            # Divide the stacked signals into single speaker signals for consistency
            for i in range(num_spk - 1, -1, -1):
                idx = str(i + 1)
                # make sure no np.nan paddings are in the data
                assert not np.isnan(np.sum(data[sref_name][i])), uid
                data[self.speech_ref_name_prefix + idx] = data[sref_name][i]
                if dref_name in data:
                    # make sure no np.nan paddings are in the data
                    assert not np.isnan(np.sum(data[dref_name][i])), uid
                    data[self.dereverb_ref_name_prefix + idx] = data[dref_name][i]

        if self.train:
            if self.speech_segment is not None:
                speech_segment = self.speech_segment // self.sample_rate * fs
                start, end = self._random_crop_range(
                    data, num_spk, speech_segment, uid=uid
                )
                self._apply_to_all_signals(data, lambda x: x[start:end], num_spk)
            # clean speech signal (Nmic, Time)
            speech_ref = [
                self._ensure_2d(data[self.speech_ref_name_prefix + str(i + 1)])
                for i in range(num_spk)
            ]

            # dereverberated (noisy) signal (Nmic, Time)
            if self.dereverb_ref_name_prefix + "1" in data:
                dereverb_speech_ref = [
                    self._ensure_2d(data[self.dereverb_ref_name_prefix + str(i + 1)])
                    for i in range(num_spk)
                    if self.dereverb_ref_name_prefix + str(i + 1) in data
                ]
                assert len(dereverb_speech_ref) in (1, num_spk), len(
                    dereverb_speech_ref
                )
            else:
                dereverb_speech_ref = None

            # Calc power on non silence region
            power_ref = [
                (sref[detect_non_silence(sref)] ** 2).mean() for sref in speech_ref
            ]

            speech_mix = self._ensure_2d(data[self.speech_name])
            # 1. Convolve RIR
            if self.rirs is not None and self.rir_apply_prob >= np.random.random():
                speech_ref, rir_ref = zip(
                    *[
                        self._convolve_rir(
                            sp,
                            power,
                            self.rirs,
                            tgt_fs=fs,
                            single_channel=self.force_single_channel,
                        )
                        for sp, power in zip(speech_ref, power_ref)
                    ]
                )
                if self.force_single_channel:
                    speech_ref = list(map(lambda x: x[:1], speech_ref))
                    rir_ref = list(map(lambda x: x[:1], rir_ref))

                if self.use_reverberant_ref:
                    for spk in range(num_spk):
                        suffix = str(spk + 1)
                        speech_ref_name = self.speech_ref_name_prefix + suffix
                        # (Time, Nmic)
                        data[speech_ref_name] = speech_ref[spk].T

                        if dereverb_speech_ref is not None:
                            if spk == 0 or len(dereverb_speech_ref) > 1:
                                dereverb_name = self.dereverb_ref_name_prefix + suffix
                                data[dereverb_name] = self._get_early_signal(
                                    speech_ref[spk], rir_ref[spk], power_ref[spk]
                                ).T
                else:
                    for spk in range(num_spk):
                        suffix = str(spk + 1)
                        speech_ref_name = self.speech_ref_name_prefix + suffix
                        # clean speech with early reflections (Time, Nmic)
                        data[speech_ref_name] = self._get_early_signal(
                            speech_ref[spk], rir_ref[spk], power_ref[spk]
                        ).T

                        if dereverb_speech_ref is not None:
                            if spk == 0 or len(dereverb_speech_ref) > 1:
                                dereverb_name = self.dereverb_ref_name_prefix + suffix
                                data[dereverb_name] = data[speech_ref_name]

                if self.noise_ref_name_prefix + "1" in data:
                    noise = data[self.noise_ref_name_prefix + "1"]
                    speech_mix = sum(speech_ref) + noise
                else:
                    speech_mix = sum(speech_ref)

                # Add category information for dynamic mixing
                # "_reverb" means dereverberation is required
                # "_both" means both reverberant and dereverberated signals are required
                if "category" in data:
                    if self.use_reverberant_ref:
                        if dereverb_speech_ref is None:
                            if data["category"].endswith("_reverb"):
                                data["category"] = data["category"][:-7]
                            if data["category"].endswith("_both"):
                                data["category"] = data["category"][:-5]
                        else:
                            if not data["category"].endswith("_both"):
                                data["category"] = data["category"] + "_both"
                    elif not data["category"].endswith("_reverb"):
                        data["category"] = data["category"] + "_reverb"

            # 2. Add Noise
            if self.noises is not None and self.noise_apply_prob >= np.random.random():
                speech_mix = sum(speech_ref)
                if self.force_single_channel and speech_mix.shape[0] > 1:
                    speech_mix = speech_mix[:1]

                power_mix = (speech_mix[detect_non_silence(speech_mix)] ** 2).mean()
                speech_mix, noise = self._add_noise(
                    speech_mix,
                    power_mix,
                    self.noises,
                    self.noise_db_low,
                    self.noise_db_high,
                    tgt_fs=fs,
                    single_channel=self.force_single_channel,
                )

                name = self.noise_ref_name_prefix + "1"
                if name in data:
                    data[name] = noise.T
                for n in range(1, self.num_noise_type):
                    name = self.noise_ref_name_prefix + str(n + 1)
                    data.pop(name, None)

            if self.data_aug:
                if self.data_aug_prob > 0 and self.data_aug_prob >= np.random.random():
                    # Currently, we only apply data augmentation to the mixture.
                    # So, some effects should not be used for Enh, such as pitch_shift,
                    # speed_perturb, time_stretch, polarity_inverse, reverse, etc.
                    speech_mix = self.data_aug(
                        speech_mix.T if speech_mix.shape[0] > 1 else speech_mix[0],
                        self.sample_rate,
                    )

            data[self.speech_name] = speech_mix.T
            ma = np.max(np.abs(data[self.speech_name]))
            if ma > 1.0:
                self._apply_to_all_signals(data, lambda x: x / ma, num_spk)

            self._apply_to_all_signals(data, lambda x: x.squeeze(), num_spk)
        
        # Save multi-channel version before force_single_channel processing
        speech_mix_mc = data[self.speech_name].copy()

        if self.force_single_channel:
            self._apply_to_all_signals(
                data, lambda x: x if x.ndim == 1 else x[:, 0], num_spk
            )

        data["speech_mix_mc"] = speech_mix_mc

        if self.speech_volume_normalize is not None:
            if self.train:
                volume_scale = np.random.uniform(self.volume_low, self.volume_high)
            else:
                # use a fixed scale to make it deterministic
                volume_scale = self.volume_low
            ma = np.max(np.abs(data[self.speech_name]))
            self._apply_to_all_signals(data, lambda x: x * volume_scale / ma, num_spk)
            
            # Apply volume normalization to multi-channel version
            if "speech_mix_mc" in data:
                ma_mc = np.max(np.abs(data["speech_mix_mc"]))
                data["speech_mix_mc"] = data["speech_mix_mc"] * volume_scale / ma_mc

        if self.categories and "category" in data:
            category = data.pop("category")
            if not re.fullmatch(r"\d+ch.*", category):
                speech_mix = data[self.speech_name]
                nch = 1 if speech_mix.ndim == 1 else speech_mix.shape[-1]
                category = f"{nch}ch_" + category
            assert category in self.categories, category
            data["utt2category"] = np.array([self.categories[category]])

        speech_mix = data[self.speech_name]
        speech_mix_mc = data.get("speech_mix_mc")
        # Reorder channels of the multi-channel signals
        if self.channel_reordering and self.train:
            if speech_mix.ndim > 1:
                num_ch = speech_mix.shape[-1]
                # chs = np.random.choice(range(num_ch), size=num_ch, replace=False).tolist()
                chs = np.random.permutation(num_ch).tolist()
                data[self.speech_name] = speech_mix[..., chs]
                for i in range(num_spk):
                    k = self.speech_ref_name_prefix + str(i + 1)
                    if self.train:
                        assert k in data, (data.keys(), k)
                    if k in data and data[k].ndim > 1:
                        assert data[k].shape == speech_mix.shape
                        data[k] = data[k][..., chs]

                # Apply same channel reordering to speech_mix_mc if it exists
                if speech_mix_mc is not None and speech_mix_mc.ndim > 1:
                    assert speech_mix_mc.shape == speech_mix.shape
                    data["speech_mix_mc"] = speech_mix_mc[..., chs]
            elif speech_mix_mc is not None and speech_mix_mc.ndim > 1:
                num_ch = speech_mix_mc.shape[-1]
                chs = np.random.permutation(num_ch).tolist()
                data["speech_mix_mc"] = speech_mix_mc[..., chs]

        return data

    @typechecked
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:

        data = self._speech_process(uid, data)
        data = self._text_process(data)
        return data
