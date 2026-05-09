"""NPZ preprocessor for WHAMR-style data."""

import hashlib
import os
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.signal
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.train.preprocessor import EnhPreprocessor

_WHAMROOM_CLS = None


def _get_whamroom_cls():
    global _WHAMROOM_CLS
    if _WHAMROOM_CLS is not None:
        return _WHAMROOM_CLS
    try:
        import pyroomacoustics as pra
        from pyroomacoustics.parameters import constants
    except Exception as exc:
        raise ImportError(
            "pyroomacoustics is required for NpzPreprocessor."
        ) from exc

    class WhamRoom(pra.room.ShoeBox):
        def __init__(self, p, mics, s1, s2, T60, fs=16000, t0=0.0, sigma2_awgn=None):
            self.T60 = T60
            self.max_rir_len = int(np.ceil(T60 * fs))

            volume = p[0] * p[1] * p[2]
            surface_area = 2 * (p[0] * p[1] + p[0] * p[2] + p[1] * p[2])
            absorption = (
                24 * volume * np.log(10.0) / (constants.get("c") * surface_area * T60)
            )

            max_order = int(np.ceil(T60 * constants.get("c") / min(p)))

            super().__init__(
                p,
                fs=fs,
                t0=t0,
                absorption=absorption,
                max_order=max_order,
                sigma2_awgn=sigma2_awgn,
                sources=None,
                mics=None,
            )

            self.add_source(s1)
            self.add_source(s2)
            self.add_microphone_array(pra.MicrophoneArray(np.array(mics).T, fs))

        def add_audio(self, s1, s2):
            self.sources[0].add_signal(s1)
            self.sources[1].add_signal(s2)

        def generate_audio(self, anechoic=False, fs=16000):
            if not self.rir:
                self.generate_rirs()
            if anechoic:
                self.rir = self.rir_anechoic
            else:
                self.rir = self.rir_reverberant
            audio_array = self.simulate(return_premix=True, recompute_rir=False)

            if type(fs) is not list:
                fs_array = [fs]
            else:
                fs_array = fs
            audio_out = []
            for elem in fs_array:
                if type(elem) is str:
                    elem = int(elem.replace("k", "000"))
                if elem != self.fs:
                    assert self.fs % elem == 0
                    audio_out.append(
                        scipy.signal.resample_poly(audio_array, elem, self.fs, axis=2)
                    )
                else:
                    audio_out.append(audio_array)
            if type(fs) is not list:
                return audio_out[0]
            else:
                return audio_out

    _WHAMROOM_CLS = WhamRoom
    return _WHAMROOM_CLS


def _np_item(value):
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def _as_str(value):
    return str(_np_item(value))


def _append_zeros(samples, desired_length):
    samples_to_add = desired_length - len(samples)
    if len(samples.shape) == 1:
        new_zeros = np.zeros(samples_to_add)
    elif len(samples.shape) == 2:
        new_zeros = np.zeros((samples_to_add, 2))
    else:
        raise ValueError("Unsupported sample shape in _append_zeros")
    return np.append(samples, new_zeros, axis=0)


def _fix_length(s1, s2, min_or_max="max"):
    if min_or_max == "min":
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:
        utt_len = np.maximum(len(s1), len(s2))
        s1 = _append_zeros(s1, utt_len)
        s2 = _append_zeros(s2, utt_len)
    return s1, s2


def _append_or_truncate(
    s1_samples, s2_samples, noise_samples, min_or_max="max", start_samp_16k=0, downsample=False
):
    if downsample:
        speech_start_sample = start_samp_16k // 2
    else:
        speech_start_sample = start_samp_16k

    if min_or_max == "min":
        speech_end_sample = speech_start_sample + len(s1_samples)
        noise_samples = noise_samples[speech_start_sample:speech_end_sample]
    else:
        speech_end_sample = len(s1_samples) - speech_start_sample
        s1_append = np.zeros_like(noise_samples)
        s2_append = np.zeros_like(noise_samples)
        s1_append[speech_start_sample:len(s1_samples)] = s1_samples[0:speech_end_sample]
        s2_append[speech_start_sample:len(s1_samples)] = s2_samples[0:speech_end_sample]
        s1_samples = s1_append
        s2_samples = s2_append

    return s1_samples, s2_samples, noise_samples


def _quantize_pcm16(samples: np.ndarray) -> np.ndarray:
    int_samples = np.int16(np.round((2 ** 15) * samples))
    return np.float64(int_samples) / (2 ** 15)


class NpzPreprocessor(EnhPreprocessor):
    """Preprocessor for WHAMR NPZ data."""

    def __init__(
        self,
        train: bool,
        mix_type: str = "both",
        ref_condition: str = "reverb",
        npz_path_name: str = "npz_path",
        s1_base_name: str = "s1_base",
        s2_base_name: str = "s2_base",
        s1_temp_name: str = "s1_temp",
        s2_temp_name: str = "s2_temp",
        noise_base_name: str = "noise_base",
        rir_path_name: str = "rir_path",
        room_param_path_name: str = "room_param_path",
        # inherited from EnhPreprocessor
        rir_scp: Optional[str] = None,
        rir_apply_prob: float = 0.0,
        noise_scp: Optional[str] = None,
        noise_apply_prob: float = 0.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 2,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
        channel_reordering: bool = False,
        categories: Optional[List] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        flexible_numspk: bool = False,
        output_audio_subtype: Optional[str] = "PCM_16",
        anchor_single_channel: bool = False,
        contrastive_enable: bool = False,
        contrastive_num_neg: int = 1,
        contrastive_seed: int = 1234,
        contrastive_epoch: int = 0,
        contrastive_random_each_call: bool = False,
        contrastive_random_train_only: bool = False,
        contrastive_scale_min: float = 0.9,
        contrastive_scale_max: float = 1.1,
        contrastive_pool_rir_scp: Optional[str] = None,
        contrastive_pool_room_param_scp: Optional[str] = None,
    ):
        super().__init__(
            train=train,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            speech_ref_name_prefix=speech_ref_name_prefix,
            noise_ref_name_prefix=noise_ref_name_prefix,
            dereverb_ref_name_prefix=dereverb_ref_name_prefix,
            use_reverberant_ref=use_reverberant_ref,
            num_spk=num_spk,
            num_noise_type=num_noise_type,
            sample_rate=sample_rate,
            force_single_channel=force_single_channel,
            channel_reordering=channel_reordering,
            categories=categories,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
            speech_segment=speech_segment,
            avoid_allzero_segment=avoid_allzero_segment,
            flexible_numspk=flexible_numspk,
        )
        if mix_type not in ("both", "clean", "single"):
            raise ValueError(f"mix_type must be one of both/clean/single: {mix_type}")
        if ref_condition not in ("anechoic", "reverb"):
            raise ValueError(
                f"ref_condition must be one of anechoic/reverb: {ref_condition}"
            )
        if mix_type == "single" and num_spk != 1:
            raise ValueError("mix_type=single requires num_spk=1")
        if mix_type in ("both", "clean") and num_spk < 2:
            raise ValueError("mix_type=both/clean requires num_spk>=2")

        self.mix_type = mix_type
        self.ref_condition = ref_condition
        self.npz_path_name = npz_path_name
        self.s1_base_name = s1_base_name
        self.s2_base_name = s2_base_name
        self.s1_temp_name = s1_temp_name
        self.s2_temp_name = s2_temp_name
        self.noise_base_name = noise_base_name
        self.rir_path_name = rir_path_name
        self.room_param_path_name = room_param_path_name
        if output_audio_subtype is not None and output_audio_subtype != "PCM_16":
            raise ValueError("Only PCM_16 or None is supported for output_audio_subtype")
        self.output_audio_subtype = output_audio_subtype
        self.anchor_single_channel = bool(anchor_single_channel)

        self.contrastive_enable = contrastive_enable
        self.contrastive_num_neg = int(contrastive_num_neg)
        self.contrastive_seed = int(contrastive_seed)
        self.contrastive_epoch = int(contrastive_epoch)
        self.contrastive_random_each_call = bool(contrastive_random_each_call)
        self.contrastive_random_train_only = bool(contrastive_random_train_only)
        self.contrastive_scale_min = float(contrastive_scale_min)
        self.contrastive_scale_max = float(contrastive_scale_max)
        if self.contrastive_scale_min > self.contrastive_scale_max:
            raise ValueError("contrastive_scale_min must be <= contrastive_scale_max")

        self.contrastive_pool_rir = None
        self.contrastive_pool_room_param = None
        self._contrastive_pool_keys = None
        if self.contrastive_enable:
            if not contrastive_pool_rir_scp or not contrastive_pool_room_param_scp:
                raise ValueError(
                    "contrastive_pool_rir_scp and contrastive_pool_room_param_scp are required"
                )
            if isinstance(contrastive_pool_rir_scp, dict) or isinstance(
                contrastive_pool_room_param_scp, dict
            ):
                if not (
                    isinstance(contrastive_pool_rir_scp, dict)
                    and isinstance(contrastive_pool_room_param_scp, dict)
                ):
                    raise ValueError(
                        "contrastive_pool_rir_scp and contrastive_pool_room_param_scp must both be dicts or both be strings"
                    )
                self.contrastive_pool_rir = {}
                self.contrastive_pool_room_param = {}
                self._contrastive_pool_keys = {}
                if set(contrastive_pool_rir_scp.keys()) != set(
                    contrastive_pool_room_param_scp.keys()
                ):
                    raise ValueError("contrastive pool split keys mismatch")
                for split_key, scp_path in contrastive_pool_rir_scp.items():
                    pool_rir = read_2columns_text(scp_path)
                    pool_room = read_2columns_text(
                        contrastive_pool_room_param_scp[split_key]
                    )
                    pool_keys = set(pool_rir.keys())
                    if pool_keys != set(pool_room.keys()):
                        raise ValueError(
                            f"contrastive pool scp keys mismatch for split {split_key}"
                        )
                    if len(pool_keys) <= self.contrastive_num_neg:
                        raise ValueError(
                            f"contrastive pool size is too small for num_neg in split {split_key}"
                        )
                    self.contrastive_pool_rir[split_key] = pool_rir
                    self.contrastive_pool_room_param[split_key] = pool_room
                    self._contrastive_pool_keys[split_key] = sorted(pool_keys)
            else:
                self.contrastive_pool_rir = read_2columns_text(contrastive_pool_rir_scp)
                self.contrastive_pool_room_param = read_2columns_text(
                    contrastive_pool_room_param_scp
                )
                pool_keys = set(self.contrastive_pool_rir.keys())
                if pool_keys != set(self.contrastive_pool_room_param.keys()):
                    raise ValueError("contrastive pool scp keys mismatch")
                if len(pool_keys) <= self.contrastive_num_neg:
                    raise ValueError("contrastive pool size is too small for num_neg")
                self._contrastive_pool_keys = sorted(pool_keys)

    def _rir_to_list(self, rir_array: np.ndarray):
        n_mics, n_src, _ = rir_array.shape
        return [
            [rir_array[m, s] for s in range(n_src)]
            for m in range(n_mics)
        ]

    def _load_npz(self, path: str):
        return np.load(path, allow_pickle=True)

    def _rng_for_uid(self, uid: str) -> np.random.Generator:
        payload = f"{self.contrastive_seed}:{self.contrastive_epoch}:{uid}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        seed = int.from_bytes(digest, "little")
        return np.random.default_rng(seed)

    def _load_room_and_rir(self, rir_path: str, room_param_path: str):
        rir_npz = self._load_npz(rir_path)
        room_npz = self._load_npz(room_param_path)
        room_dim = np.asarray(room_npz["room_dim"], dtype=np.float64)
        mic_pos = np.asarray(room_npz["mic_pos"], dtype=np.float64)
        s1_pos = np.asarray(room_npz["s1_pos"], dtype=np.float64)
        s2_pos = np.asarray(room_npz["s2_pos"], dtype=np.float64)
        t60 = float(_np_item(room_npz["T60"]))
        room_fs = int(_np_item(room_npz["fs"]))
        rir_fs = int(_np_item(rir_npz["fs"]))
        if room_fs != rir_fs:
            raise ValueError(f"room fs and rir fs mismatch: {room_fs} vs {rir_fs}")
        return (
            rir_npz,
            room_dim,
            mic_pos,
            s1_pos,
            s2_pos,
            t60,
            room_fs,
        )

    def _synthesize_mix(
        self,
        sample_rate: int,
        data_len: str,
        start_samp_16k: int,
        wsjmix_scale: np.ndarray,
        wham_speech_scale: float,
        wham_noise_scale: float,
        mono: bool,
        s1_base: np.ndarray,
        s2_base: np.ndarray,
        s1_temp: np.ndarray,
        s2_temp: np.ndarray,
        noise_base: np.ndarray,
        rir_npz,
        room_dim: np.ndarray,
        mic_pos: np.ndarray,
        s1_pos: np.ndarray,
        s2_pos: np.ndarray,
        t60: float,
        room_fs: int,
        s1_scale_factor: float,
        s2_scale_factor: float,
    ):
        WhamRoom = _get_whamroom_cls()
        room = WhamRoom(room_dim, mic_pos, s1_pos, s2_pos, t60, fs=room_fs)
        room.rir_anechoic = self._rir_to_list(
            np.asarray(rir_npz["rir_anechoic"], dtype=np.float64)
        )
        room.rir_reverberant = self._rir_to_list(
            np.asarray(rir_npz["rir_reverberant"], dtype=np.float64)
        )
        room.rir = room.rir_anechoic
        room.add_audio(s1_temp, s2_temp)

        anechoic = room.generate_audio(anechoic=True, fs=sample_rate)
        reverberant = room.generate_audio(fs=sample_rate)

        left_ch = 0
        ch_ind = left_ch if mono else [0, 1]

        if wsjmix_scale.shape[0] != 2:
            raise ValueError(f"wsjmix_scale must have 2 elements: {wsjmix_scale}")
        s1 = s1_base * wham_speech_scale * s1_scale_factor
        s2 = s2_base * wham_speech_scale * s2_scale_factor

        s1_spatial_scaling = np.sqrt(
            np.sum(s1**2) / np.sum(anechoic[0, left_ch, :] ** 2)
        )
        s2_spatial_scaling = np.sqrt(
            np.sum(s2**2) / np.sum(anechoic[1, left_ch, :] ** 2)
        )

        noise_samples_full = noise_base * wham_noise_scale

        if data_len == "max":
            out_len = len(noise_samples_full)
        else:
            out_len = np.minimum(len(s1), len(s2))

        s1_anechoic, s2_anechoic = _fix_length(
            anechoic[0, ch_ind, :out_len].T * s1_spatial_scaling,
            anechoic[1, ch_ind, :out_len].T * s2_spatial_scaling,
            data_len,
        )
        s1_reverb, s2_reverb = _fix_length(
            reverberant[0, ch_ind, :out_len].T * s1_spatial_scaling,
            reverberant[1, ch_ind, :out_len].T * s2_spatial_scaling,
            data_len,
        )

        if self.ref_condition == "anechoic":
            s1_src, s2_src = s1_anechoic, s2_anechoic
        else:
            s1_src, s2_src = s1_reverb, s2_reverb

        s1_samples, s2_samples, noise_samples = _append_or_truncate(
            s1_src,
            s2_src,
            noise_samples_full,
            data_len,
            start_samp_16k,
            downsample=(sample_rate != room_fs),
        )

        if self.mix_type == "clean":
            speech_mix = s1_samples + s2_samples
        elif self.mix_type == "single":
            speech_mix = noise_samples + s1_samples
        else:
            speech_mix = noise_samples + s1_samples + s2_samples

        if self.output_audio_subtype == "PCM_16":
            speech_mix = _quantize_pcm16(speech_mix)
            s1_samples = _quantize_pcm16(s1_samples)
            s2_samples = _quantize_pcm16(s2_samples)

        return speech_mix, s1_samples, s2_samples

    def _apply_postprocess(self, uid: str, speech_mix, s1_samples, s2_samples, crop=None):
        if crop is not None:
            start, end = crop
            speech_mix = speech_mix[start:end]
            s1_samples = s1_samples[start:end]
            s2_samples = s2_samples[start:end]
        tmp = {
            self.speech_name: speech_mix,
            self.speech_ref_name_prefix + "1": s1_samples,
        }
        if self.num_spk > 1:
            tmp[self.speech_ref_name_prefix + "2"] = s2_samples
        if crop is not None and self.train and self.speech_segment is not None:
            original = self.speech_segment
            try:
                self.speech_segment = None
                tmp = super()._speech_process(uid, tmp)
            finally:
                self.speech_segment = original
        else:
            tmp = super()._speech_process(uid, tmp)
        return tmp[self.speech_name]

    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        if self.npz_path_name not in data:
            return data
        if self.contrastive_enable:
            return self._speech_process_contrastive(uid, data)

        npz_path = _as_str(data[self.npz_path_name])
        meta = self._load_npz(npz_path)

        sample_rate = int(_np_item(meta["sample_rate"]))
        data_len = _as_str(meta["data_len"])
        mono = bool(_np_item(meta["mono"]))
        start_samp_16k = int(_np_item(meta["start_samp_16k"]))
        split = _as_str(meta["split"])
        wsjmix_scale = np.asarray(meta["wsjmix_scale"], dtype=np.float64)
        wham_speech_scale = float(_np_item(meta["wham_speech_scale"]))
        wham_noise_scale = float(_np_item(meta["wham_noise_scale"]))

        s1_base = np.asarray(data[self.s1_base_name], dtype=np.float64)
        s2_base = np.asarray(data[self.s2_base_name], dtype=np.float64)
        s1_temp = np.asarray(data[self.s1_temp_name], dtype=np.float64)
        s2_temp = np.asarray(data[self.s2_temp_name], dtype=np.float64)
        noise_base = np.asarray(data[self.noise_base_name], dtype=np.float64)

        rir_path = _as_str(data[self.rir_path_name])
        room_param_path = _as_str(data[self.room_param_path_name])
        rir_npz = self._load_npz(rir_path)
        room_npz = self._load_npz(room_param_path)

        room_dim = np.asarray(room_npz["room_dim"], dtype=np.float64)
        mic_pos = np.asarray(room_npz["mic_pos"], dtype=np.float64)
        s1_pos = np.asarray(room_npz["s1_pos"], dtype=np.float64)
        s2_pos = np.asarray(room_npz["s2_pos"], dtype=np.float64)
        t60 = float(_np_item(room_npz["T60"]))
        room_fs = int(_np_item(room_npz["fs"]))
        rir_fs = int(_np_item(rir_npz["fs"]))

        if room_fs != rir_fs:
            raise ValueError(f"room fs and rir fs mismatch: {room_fs} vs {rir_fs}")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"metadata sample_rate {sample_rate} != preprocessor sample_rate {self.sample_rate}"
            )

        (
            speech_mix,
            s1_samples,
            s2_samples,
        ) = self._synthesize_mix(
            sample_rate=sample_rate,
            data_len=data_len,
            start_samp_16k=start_samp_16k,
            wsjmix_scale=wsjmix_scale,
            wham_speech_scale=wham_speech_scale,
            wham_noise_scale=wham_noise_scale,
            mono=mono,
            s1_base=s1_base,
            s2_base=s2_base,
            s1_temp=s1_temp,
            s2_temp=s2_temp,
            noise_base=noise_base,
            rir_npz=rir_npz,
            room_dim=room_dim,
            mic_pos=mic_pos,
            s1_pos=s1_pos,
            s2_pos=s2_pos,
            t60=t60,
            room_fs=room_fs,
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )

        data[self.speech_name] = self._apply_postprocess(
            uid, speech_mix, s1_samples, s2_samples
        )

        data.pop(self.npz_path_name, None)
        data.pop(self.s1_base_name, None)
        data.pop(self.s2_base_name, None)
        data.pop(self.s1_temp_name, None)
        data.pop(self.s2_temp_name, None)
        data.pop(self.noise_base_name, None)
        data.pop(self.rir_path_name, None)
        data.pop(self.room_param_path_name, None)

        return data

    def _speech_process_contrastive(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        npz_path = _as_str(data[self.npz_path_name])
        meta = self._load_npz(npz_path)

        sample_rate = int(_np_item(meta["sample_rate"]))
        data_len = _as_str(meta["data_len"])
        mono = bool(_np_item(meta["mono"]))
        start_samp_16k = int(_np_item(meta["start_samp_16k"]))
        split = _as_str(meta["split"])
        wsjmix_scale = np.asarray(meta["wsjmix_scale"], dtype=np.float64)
        wham_speech_scale = float(_np_item(meta["wham_speech_scale"]))
        wham_noise_scale = float(_np_item(meta["wham_noise_scale"]))

        s1_base = np.asarray(data[self.s1_base_name], dtype=np.float64)
        s2_base = np.asarray(data[self.s2_base_name], dtype=np.float64)
        s1_temp = np.asarray(data[self.s1_temp_name], dtype=np.float64)
        s2_temp = np.asarray(data[self.s2_temp_name], dtype=np.float64)
        noise_base = np.asarray(data[self.noise_base_name], dtype=np.float64)

        rir_path = _as_str(data[self.rir_path_name])
        room_param_path = _as_str(data[self.room_param_path_name])

        (
            rir_npz,
            room_dim,
            mic_pos,
            s1_pos,
            s2_pos,
            t60,
            room_fs,
        ) = self._load_room_and_rir(rir_path, room_param_path)

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"metadata sample_rate {sample_rate} != preprocessor sample_rate {self.sample_rate}"
            )

        if self.contrastive_random_each_call and (
            not self.contrastive_random_train_only or self.train
        ):
            rng = np.random.default_rng()
        else:
            rng = self._rng_for_uid(uid)

        speech_mix, s1_samples, s2_samples = self._synthesize_mix(
            sample_rate=sample_rate,
            data_len=data_len,
            start_samp_16k=start_samp_16k,
            wsjmix_scale=wsjmix_scale,
            wham_speech_scale=wham_speech_scale,
            wham_noise_scale=wham_noise_scale,
            mono=mono,
            s1_base=s1_base,
            s2_base=s2_base,
            s1_temp=s1_temp,
            s2_temp=s2_temp,
            noise_base=noise_base,
            rir_npz=rir_npz,
            room_dim=room_dim,
            mic_pos=mic_pos,
            s1_pos=s1_pos,
            s2_pos=s2_pos,
            t60=t60,
            room_fs=room_fs,
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )
        crop = None
        if self.train and self.speech_segment is not None:
            speech_segment = self.speech_segment // self.sample_rate * sample_rate
            crop = self._random_crop_range(
                {
                    self.speech_ref_name_prefix + "1": s1_samples,
                    self.speech_ref_name_prefix + "2": s2_samples,
                },
                self.num_spk,
                speech_segment,
                uid=uid,
            )
        speech_anchor = self._apply_postprocess(
            uid, speech_mix, s1_samples, s2_samples, crop=crop
        )
        if self.anchor_single_channel and speech_anchor.ndim > 1:
            # Match EnhPreprocessor single-channel behavior (take channel 0)
            speech_anchor = speech_anchor[:, 0]

        s1_factor = rng.uniform(self.contrastive_scale_min, self.contrastive_scale_max)
        s2_factor = rng.uniform(self.contrastive_scale_min, self.contrastive_scale_max)
        speech_mix, s1_samples, s2_samples = self._synthesize_mix(
            sample_rate=sample_rate,
            data_len=data_len,
            start_samp_16k=start_samp_16k,
            wsjmix_scale=wsjmix_scale,
            wham_speech_scale=wham_speech_scale,
            wham_noise_scale=wham_noise_scale,
            mono=mono,
            s1_base=s1_base,
            s2_base=s2_base,
            s1_temp=s1_temp,
            s2_temp=s2_temp,
            noise_base=noise_base,
            rir_npz=rir_npz,
            room_dim=room_dim,
            mic_pos=mic_pos,
            s1_pos=s1_pos,
            s2_pos=s2_pos,
            t60=t60,
            room_fs=room_fs,
            s1_scale_factor=s1_factor,
            s2_scale_factor=s2_factor,
        )
        speech_pos = self._apply_postprocess(
            uid, speech_mix, s1_samples, s2_samples, crop=crop
        )

        if isinstance(self._contrastive_pool_keys, dict):
            if split not in self._contrastive_pool_keys:
                raise ValueError(f"contrastive pool not found for split {split}")
            pool_keys = [
                k for k in self._contrastive_pool_keys[split] if k != uid
            ]
            pool_rir = self.contrastive_pool_rir[split]
            pool_room = self.contrastive_pool_room_param[split]
        else:
            pool_keys = [k for k in self._contrastive_pool_keys if k != uid]
            pool_rir = self.contrastive_pool_rir
            pool_room = self.contrastive_pool_room_param
        if len(pool_keys) < self.contrastive_num_neg:
            raise ValueError("contrastive pool is too small for requested negatives")
        neg_keys = rng.choice(
            pool_keys, size=self.contrastive_num_neg, replace=False
        ).tolist()

        speech_negs = []
        for nkey in neg_keys:
            neg_rir_path = pool_rir[nkey]
            neg_room_param_path = pool_room[nkey]
            (
                neg_rir_npz,
                neg_room_dim,
                neg_mic_pos,
                neg_s1_pos,
                neg_s2_pos,
                neg_t60,
                neg_room_fs,
            ) = self._load_room_and_rir(neg_rir_path, neg_room_param_path)

            s1_factor = rng.uniform(
                self.contrastive_scale_min, self.contrastive_scale_max
            )
            s2_factor = rng.uniform(
                self.contrastive_scale_min, self.contrastive_scale_max
            )
            speech_mix, s1_samples, s2_samples = self._synthesize_mix(
                sample_rate=sample_rate,
                data_len=data_len,
                start_samp_16k=start_samp_16k,
                wsjmix_scale=wsjmix_scale,
                wham_speech_scale=wham_speech_scale,
                wham_noise_scale=wham_noise_scale,
                mono=mono,
                s1_base=s1_base,
                s2_base=s2_base,
                s1_temp=s1_temp,
                s2_temp=s2_temp,
                noise_base=noise_base,
                rir_npz=neg_rir_npz,
                room_dim=neg_room_dim,
                mic_pos=neg_mic_pos,
                s1_pos=neg_s1_pos,
                s2_pos=neg_s2_pos,
                t60=neg_t60,
                room_fs=neg_room_fs,
                s1_scale_factor=s1_factor,
                s2_scale_factor=s2_factor,
            )
            speech_negs.append(
                self._apply_postprocess(uid, speech_mix, s1_samples, s2_samples, crop=crop)
            )

        preserved = {
            k: v
            for k, v in data.items()
            if k
            not in {
                self.npz_path_name,
                self.s1_base_name,
                self.s2_base_name,
                self.s1_temp_name,
                self.s2_temp_name,
                self.noise_base_name,
                self.rir_path_name,
                self.room_param_path_name,
            }
        }
        data = preserved
        data["speech_anchor"] = speech_anchor
        data["speech_pos"] = speech_pos
        for idx, neg in enumerate(speech_negs, start=1):
            data[f"speech_neg{idx}"] = neg
        return data

    @typechecked
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        data = self._speech_process(uid, data)
        data = self._text_process(data)
        return data


class NpzSwapRirPreprocessor(NpzPreprocessor):
    """NPZ preprocessor for contrastive learning with RIR swap.

    Anchor  : data1 (speech/noise/RIR)
    Positive: data2 speech/noise + data1 RIR
    Negative: data2 (speech/noise/RIR)
    """

    def __init__(
        self,
        train: bool,
        mix_type: str = "both",
        ref_condition: str = "reverb",
        npz_path_name: str = "npz_path",
        s1_base_name: str = "s1_base",
        s2_base_name: str = "s2_base",
        s1_temp_name: str = "s1_temp",
        s2_temp_name: str = "s2_temp",
        noise_base_name: str = "noise_base",
        rir_path_name: str = "rir_path",
        room_param_path_name: str = "room_param_path",
        rir_scp: Optional[str] = None,
        rir_apply_prob: float = 0.0,
        noise_scp: Optional[str] = None,
        noise_apply_prob: float = 0.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 2,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
        channel_reordering: bool = False,
        categories: Optional[List] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        flexible_numspk: bool = False,
        output_audio_subtype: Optional[str] = "PCM_16",
        anchor_single_channel: bool = False,
        contrastive_enable: bool = False,
        contrastive_num_neg: int = 1,
        contrastive_seed: int = 1234,
        contrastive_epoch: int = 0,
        contrastive_random_each_call: bool = False,
        contrastive_random_train_only: bool = False,
        contrastive_scale_min: float = 1.0,
        contrastive_scale_max: float = 1.0,
        contrastive_pool_rir_scp: Optional[str] = None,
        contrastive_pool_room_param_scp: Optional[str] = None,
        contrastive_pool_npz_scp: Optional[Union[str, Dict[str, str]]] = None,
    ):
        super().__init__(
            train=train,
            mix_type=mix_type,
            ref_condition=ref_condition,
            npz_path_name=npz_path_name,
            s1_base_name=s1_base_name,
            s2_base_name=s2_base_name,
            s1_temp_name=s1_temp_name,
            s2_temp_name=s2_temp_name,
            noise_base_name=noise_base_name,
            rir_path_name=rir_path_name,
            room_param_path_name=room_param_path_name,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            speech_ref_name_prefix=speech_ref_name_prefix,
            noise_ref_name_prefix=noise_ref_name_prefix,
            dereverb_ref_name_prefix=dereverb_ref_name_prefix,
            use_reverberant_ref=use_reverberant_ref,
            num_spk=num_spk,
            num_noise_type=num_noise_type,
            sample_rate=sample_rate,
            force_single_channel=force_single_channel,
            channel_reordering=channel_reordering,
            categories=categories,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
            speech_segment=speech_segment,
            avoid_allzero_segment=avoid_allzero_segment,
            flexible_numspk=flexible_numspk,
            output_audio_subtype=output_audio_subtype,
            anchor_single_channel=anchor_single_channel,
            contrastive_enable=contrastive_enable,
            contrastive_num_neg=contrastive_num_neg,
            contrastive_seed=contrastive_seed,
            contrastive_epoch=contrastive_epoch,
            contrastive_random_each_call=contrastive_random_each_call,
            contrastive_random_train_only=contrastive_random_train_only,
            contrastive_scale_min=contrastive_scale_min,
            contrastive_scale_max=contrastive_scale_max,
            contrastive_pool_rir_scp=contrastive_pool_rir_scp,
            contrastive_pool_room_param_scp=contrastive_pool_room_param_scp,
        )

        self.contrastive_pool_npz = None
        self._contrastive_pool_npz_keys = None
        if self.contrastive_enable:
            if contrastive_pool_npz_scp is None:
                raise ValueError("contrastive_pool_npz_scp is required")
            if isinstance(contrastive_pool_npz_scp, dict):
                self.contrastive_pool_npz = {}
                self._contrastive_pool_npz_keys = {}
                for split_key, scp_path in contrastive_pool_npz_scp.items():
                    pool_npz = read_2columns_text(scp_path)
                    if len(pool_npz) <= 1:
                        raise ValueError(
                            f"contrastive pool npz size is too small for split {split_key}"
                        )
                    self.contrastive_pool_npz[split_key] = pool_npz
                    self._contrastive_pool_npz_keys[split_key] = sorted(pool_npz.keys())
            else:
                pool_npz = read_2columns_text(contrastive_pool_npz_scp)
                if len(pool_npz) <= 1:
                    raise ValueError("contrastive pool npz size is too small")
                self.contrastive_pool_npz = pool_npz
                self._contrastive_pool_npz_keys = sorted(pool_npz.keys())

    def _resolve_npz_relpath(self, npz_path: str, relpath: str) -> str:
        if os.path.isabs(relpath):
            return relpath
        base_dir = os.path.dirname(os.path.dirname(npz_path))
        return os.path.join(base_dir, relpath)

    def _get_npz_pool(self, split: str):
        if isinstance(self._contrastive_pool_npz_keys, dict):
            if split not in self._contrastive_pool_npz_keys:
                raise ValueError(f"contrastive pool npz not found for split {split}")
            pool_keys = self._contrastive_pool_npz_keys[split]
            pool_npz = self.contrastive_pool_npz[split]
        else:
            pool_keys = self._contrastive_pool_npz_keys
            pool_npz = self.contrastive_pool_npz
        return pool_npz, pool_keys

    def _load_npz_bundle(self, npz_path: str):
        meta = self._load_npz(npz_path)
        sample_rate = int(_np_item(meta["sample_rate"]))
        data_len = _as_str(meta["data_len"])
        mono = bool(_np_item(meta["mono"]))
        start_samp_16k = int(_np_item(meta["start_samp_16k"]))
        split = _as_str(meta["split"])
        wsjmix_scale = np.asarray(meta["wsjmix_scale"], dtype=np.float64)
        wham_speech_scale = float(_np_item(meta["wham_speech_scale"]))
        wham_noise_scale = float(_np_item(meta["wham_noise_scale"]))

        s1_base = np.asarray(
            np.load(
                self._resolve_npz_relpath(npz_path, _as_str(meta["s1_base_relpath"])),
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        s2_base = np.asarray(
            np.load(
                self._resolve_npz_relpath(npz_path, _as_str(meta["s2_base_relpath"])),
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        s1_temp = np.asarray(
            np.load(
                self._resolve_npz_relpath(npz_path, _as_str(meta["s1_temp_relpath"])),
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        s2_temp = np.asarray(
            np.load(
                self._resolve_npz_relpath(npz_path, _as_str(meta["s2_temp_relpath"])),
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        noise_base = np.asarray(
            np.load(
                self._resolve_npz_relpath(npz_path, _as_str(meta["noise_base_relpath"])),
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        rir_path = self._resolve_npz_relpath(npz_path, _as_str(meta["rir_relpath"]))
        room_param_path = self._resolve_npz_relpath(
            npz_path, _as_str(meta["room_param_relpath"])
        )

        return {
            "sample_rate": sample_rate,
            "data_len": data_len,
            "mono": mono,
            "start_samp_16k": start_samp_16k,
            "split": split,
            "wsjmix_scale": wsjmix_scale,
            "wham_speech_scale": wham_speech_scale,
            "wham_noise_scale": wham_noise_scale,
            "s1_base": s1_base,
            "s2_base": s2_base,
            "s1_temp": s1_temp,
            "s2_temp": s2_temp,
            "noise_base": noise_base,
            "rir_path": rir_path,
            "room_param_path": room_param_path,
        }

    def _speech_process_contrastive(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        if self.contrastive_num_neg != 1:
            raise ValueError("NpzSwapRirPreprocessor supports contrastive_num_neg=1")

        npz_path = _as_str(data[self.npz_path_name])
        meta = self._load_npz(npz_path)

        sample_rate = int(_np_item(meta["sample_rate"]))
        data_len = _as_str(meta["data_len"])
        mono = bool(_np_item(meta["mono"]))
        start_samp_16k = int(_np_item(meta["start_samp_16k"]))
        split = _as_str(meta["split"])
        wsjmix_scale = np.asarray(meta["wsjmix_scale"], dtype=np.float64)
        wham_speech_scale = float(_np_item(meta["wham_speech_scale"]))
        wham_noise_scale = float(_np_item(meta["wham_noise_scale"]))

        s1_base = np.asarray(data[self.s1_base_name], dtype=np.float64)
        s2_base = np.asarray(data[self.s2_base_name], dtype=np.float64)
        s1_temp = np.asarray(data[self.s1_temp_name], dtype=np.float64)
        s2_temp = np.asarray(data[self.s2_temp_name], dtype=np.float64)
        noise_base = np.asarray(data[self.noise_base_name], dtype=np.float64)

        rir_path = _as_str(data[self.rir_path_name])
        room_param_path = _as_str(data[self.room_param_path_name])

        (
            rir_npz,
            room_dim,
            mic_pos,
            s1_pos,
            s2_pos,
            t60,
            room_fs,
        ) = self._load_room_and_rir(rir_path, room_param_path)

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"metadata sample_rate {sample_rate} != preprocessor sample_rate {self.sample_rate}"
            )

        if self.contrastive_random_each_call and (
            not self.contrastive_random_train_only or self.train
        ):
            rng = np.random.default_rng()
        else:
            rng = self._rng_for_uid(uid)

        speech_mix, s1_samples, s2_samples = self._synthesize_mix(
            sample_rate=sample_rate,
            data_len=data_len,
            start_samp_16k=start_samp_16k,
            wsjmix_scale=wsjmix_scale,
            wham_speech_scale=wham_speech_scale,
            wham_noise_scale=wham_noise_scale,
            mono=mono,
            s1_base=s1_base,
            s2_base=s2_base,
            s1_temp=s1_temp,
            s2_temp=s2_temp,
            noise_base=noise_base,
            rir_npz=rir_npz,
            room_dim=room_dim,
            mic_pos=mic_pos,
            s1_pos=s1_pos,
            s2_pos=s2_pos,
            t60=t60,
            room_fs=room_fs,
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )
        crop_anchor = None
        if self.train and self.speech_segment is not None:
            speech_segment = self.speech_segment // self.sample_rate * sample_rate
            crop_anchor = self._random_crop_range(
                {
                    self.speech_ref_name_prefix + "1": s1_samples,
                    self.speech_ref_name_prefix + "2": s2_samples,
                },
                self.num_spk,
                speech_segment,
                uid=uid,
            )
        speech_anchor = self._apply_postprocess(
            uid, speech_mix, s1_samples, s2_samples, crop=crop_anchor
        )
        if self.anchor_single_channel and speech_anchor.ndim > 1:
            speech_anchor = speech_anchor[:, 0]

        pool_npz, pool_keys = self._get_npz_pool(split)
        pool_keys = [k for k in pool_keys if k != uid]
        if len(pool_keys) == 0:
            raise ValueError("contrastive pool npz is too small for sampling data2")
        data2_uid = rng.choice(pool_keys)
        data2_npz_path = pool_npz[data2_uid]

        data2 = self._load_npz_bundle(data2_npz_path)
        sample_rate2 = data2["sample_rate"]
        if sample_rate2 != self.sample_rate:
            raise ValueError(
                f"metadata sample_rate {sample_rate2} != preprocessor sample_rate {self.sample_rate}"
            )

        (
            neg_rir_npz,
            neg_room_dim,
            neg_mic_pos,
            neg_s1_pos,
            neg_s2_pos,
            neg_t60,
            neg_room_fs,
        ) = self._load_room_and_rir(data2["rir_path"], data2["room_param_path"])

        speech_mix_pos, s1_pos_samples, s2_pos_samples = self._synthesize_mix(
            sample_rate=sample_rate2,
            data_len=data2["data_len"],
            start_samp_16k=data2["start_samp_16k"],
            wsjmix_scale=data2["wsjmix_scale"],
            wham_speech_scale=data2["wham_speech_scale"],
            wham_noise_scale=data2["wham_noise_scale"],
            mono=data2["mono"],
            s1_base=data2["s1_base"],
            s2_base=data2["s2_base"],
            s1_temp=data2["s1_temp"],
            s2_temp=data2["s2_temp"],
            noise_base=data2["noise_base"],
            rir_npz=rir_npz,
            room_dim=room_dim,
            mic_pos=mic_pos,
            s1_pos=s1_pos,
            s2_pos=s2_pos,
            t60=t60,
            room_fs=room_fs,
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )

        speech_mix_neg, s1_neg_samples, s2_neg_samples = self._synthesize_mix(
            sample_rate=sample_rate2,
            data_len=data2["data_len"],
            start_samp_16k=data2["start_samp_16k"],
            wsjmix_scale=data2["wsjmix_scale"],
            wham_speech_scale=data2["wham_speech_scale"],
            wham_noise_scale=data2["wham_noise_scale"],
            mono=data2["mono"],
            s1_base=data2["s1_base"],
            s2_base=data2["s2_base"],
            s1_temp=data2["s1_temp"],
            s2_temp=data2["s2_temp"],
            noise_base=data2["noise_base"],
            rir_npz=neg_rir_npz,
            room_dim=neg_room_dim,
            mic_pos=neg_mic_pos,
            s1_pos=neg_s1_pos,
            s2_pos=neg_s2_pos,
            t60=neg_t60,
            room_fs=neg_room_fs,
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )

        crop_posneg = None
        if self.train and self.speech_segment is not None:
            speech_segment = self.speech_segment // self.sample_rate * sample_rate2
            crop_posneg = self._random_crop_range(
                {
                    self.speech_ref_name_prefix + "1": s1_pos_samples,
                    self.speech_ref_name_prefix + "2": s2_pos_samples,
                },
                self.num_spk,
                speech_segment,
                uid=data2_uid,
            )
        speech_pos = self._apply_postprocess(
            uid, speech_mix_pos, s1_pos_samples, s2_pos_samples, crop=crop_posneg
        )
        speech_neg = self._apply_postprocess(
            uid, speech_mix_neg, s1_neg_samples, s2_neg_samples, crop=crop_posneg
        )

        preserved = {
            k: v
            for k, v in data.items()
            if k
            not in {
                self.npz_path_name,
                self.s1_base_name,
                self.s2_base_name,
                self.s1_temp_name,
                self.s2_temp_name,
                self.noise_base_name,
                self.rir_path_name,
                self.room_param_path_name,
            }
        }
        data = preserved
        data["speech_anchor"] = speech_anchor
        data["speech_pos"] = speech_pos
        data["speech_neg1"] = speech_neg
        return data

    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        if self.npz_path_name not in data:
            return data
        if self.contrastive_enable:
            return self._speech_process_contrastive(uid, data)
        return super()._speech_process(uid, data)


class NpzSpatialAblationPreprocessor(NpzSwapRirPreprocessor):
    """Inference-time NPZ preprocessor for spatial-embedding ablation.

    This utility synthesizes multi-channel mixtures only for the spatial branch.
    The separation input can remain the original observation.
    """

    def __init__(
        self,
        npz_scp: str,
        pool_npz_scp: Optional[str] = None,
        sample_rate: int = 8000,
        seed: int = 1234,
        epoch: int = 0,
        mix_type: str = "both",
        output_audio_subtype: Optional[str] = "PCM_16",
    ):
        super().__init__(
            train=False,
            mix_type=mix_type,
            sample_rate=sample_rate,
            force_single_channel=True,
            output_audio_subtype=output_audio_subtype,
            contrastive_enable=False,
            contrastive_seed=seed,
            contrastive_epoch=epoch,
        )
        self.npz_map = read_2columns_text(npz_scp)
        if len(self.npz_map) == 0:
            raise ValueError(f"npz.scp is empty: {npz_scp}")

        pool_npz_scp = pool_npz_scp or npz_scp
        self.pool_npz_map = read_2columns_text(pool_npz_scp)
        if len(self.pool_npz_map) <= 1:
            raise ValueError(
                f"pool npz.scp must contain at least 2 utterances: {pool_npz_scp}"
            )
        self.pool_keys = sorted(self.pool_npz_map.keys())

    def _select_pool_uid(self, uid: str) -> str:
        candidates = [k for k in self.pool_keys if k != uid]
        if len(candidates) == 0:
            raise ValueError(f"No pool candidates available for uid={uid}")
        rng = self._rng_for_uid(uid)
        return str(rng.choice(candidates))

    def _load_bundle_from_map(self, uid: str, mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        if uid not in mapping:
            raise KeyError(f"UID not found in NPZ map: {uid}")
        return self._load_npz_bundle(mapping[uid])

    def _synthesize_from_bundle(
        self,
        uid: str,
        bundle: Dict[str, np.ndarray],
        rir_path: str,
        room_param_path: str,
    ) -> np.ndarray:
        if bundle["sample_rate"] != self.sample_rate:
            raise ValueError(
                f"metadata sample_rate {bundle['sample_rate']} != preprocessor sample_rate {self.sample_rate}"
            )
        (
            rir_npz,
            room_dim,
            mic_pos,
            s1_pos,
            s2_pos,
            t60,
            room_fs,
        ) = self._load_room_and_rir(rir_path, room_param_path)

        use_mono = bool(bundle["mono"] or self.force_single_channel)

        def _to_mono_if_needed(x: np.ndarray) -> np.ndarray:
            if use_mono and isinstance(x, np.ndarray) and x.ndim > 1:
                return x[:, 0]
            return x

        speech_mix, s1_samples, s2_samples = self._synthesize_mix(
            sample_rate=bundle["sample_rate"],
            data_len=bundle["data_len"],
            start_samp_16k=bundle["start_samp_16k"],
            wsjmix_scale=bundle["wsjmix_scale"],
            wham_speech_scale=bundle["wham_speech_scale"],
            wham_noise_scale=bundle["wham_noise_scale"],
            mono=use_mono,
            s1_base=_to_mono_if_needed(bundle["s1_base"]),
            s2_base=_to_mono_if_needed(bundle["s2_base"]),
            s1_temp=_to_mono_if_needed(bundle["s1_temp"]),
            s2_temp=_to_mono_if_needed(bundle["s2_temp"]),
            noise_base=_to_mono_if_needed(bundle["noise_base"]),
            rir_npz=rir_npz,
            room_dim=room_dim,
            mic_pos=mic_pos,
            s1_pos=s1_pos,
            s2_pos=s2_pos,
            t60=t60,
            room_fs=room_fs,
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )
        out = self._apply_postprocess(uid, speech_mix, s1_samples, s2_samples)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 1:
            out = out[:, None]
        return out

    def build_spatial_mix(self, uid: str, mode: str = "swap_rir") -> np.ndarray:
        """Build multi-channel mixture for spatial ablation.

        Args:
            uid: Anchor utterance id.
            mode: oracle | swap_rir | swap_audio | rand_sample
        Returns:
            (T, C) float32 waveform.
        """
        if mode not in ("oracle", "swap_rir", "swap_audio", "rand_sample"):
            raise ValueError(f"Unsupported ablation mode: {mode}")

        anchor = self._load_bundle_from_map(uid, self.npz_map)
        if mode == "oracle":
            return self._synthesize_from_bundle(
                uid, anchor, anchor["rir_path"], anchor["room_param_path"]
            )

        uid2 = self._select_pool_uid(uid)
        data2 = self._load_bundle_from_map(uid2, self.pool_npz_map)

        if mode == "swap_rir":
            return self._synthesize_from_bundle(
                uid, anchor, data2["rir_path"], data2["room_param_path"]
            )
        if mode == "swap_audio":
            return self._synthesize_from_bundle(
                uid2, data2, anchor["rir_path"], anchor["room_param_path"]
            )
        return self._synthesize_from_bundle(
            uid2, data2, data2["rir_path"], data2["room_param_path"]
        )
