"""Additional NPZ preprocessors for speaker-wise contrastive sampling."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.train.preprocessor_npz import (
    NpzSwapRirPreprocessor,
    _append_or_truncate,
    _as_str,
    _fix_length,
    _get_whamroom_cls,
    _np_item,
    _quantize_pcm16,
)


class NpzSwapRirSamePairNewPreprocessor(NpzSwapRirPreprocessor):
    """RIR-swap preprocessor with speaker-wise random data2 construction.

    Anchor  : data1 (speech/noise/RIR)
    Positive: data2 speech + data1 noise + data1 RIR
    Negative: data2 speech + data1 noise + random RIR

    data2 is built speaker-wise:
      - spk1 speech is sampled from random (spk1, *) sample
      - spk2 speech is sampled from random (spk2, *) sample
    Both sampled utterances must differ from the anchor utterances.
    """

    @typechecked
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
        speech_volume_normalize: Optional[float] = None,
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
        data_aug_effects: Optional[List] = None,
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
        contrastive_pool_rir_scp: Optional[Union[str, Dict[str, str]]] = None,
        contrastive_pool_room_param_scp: Optional[Union[str, Dict[str, str]]] = None,
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
            contrastive_pool_npz_scp=contrastive_pool_npz_scp,
        )

        self._speaker_index_by_split = isinstance(self._contrastive_pool_npz_keys, dict)
        self._speaker_uid_index = None
        self._uid_to_speaker_utt = None
        if self.contrastive_enable:
            if self._speaker_index_by_split:
                self._speaker_uid_index = {}
                self._uid_to_speaker_utt = {}
                for split_key, pool_keys in self._contrastive_pool_npz_keys.items():
                    speaker_index, utt_index = self._build_speaker_indexes(pool_keys)
                    self._speaker_uid_index[split_key] = speaker_index
                    self._uid_to_speaker_utt[split_key] = utt_index
            else:
                speaker_index, utt_index = self._build_speaker_indexes(
                    self._contrastive_pool_npz_keys
                )
                self._speaker_uid_index = speaker_index
                self._uid_to_speaker_utt = utt_index

    @staticmethod
    def _parse_uid_fields(uid: str) -> Tuple[str, str, str, str]:
        parts = uid.split("_")
        if len(parts) < 7:
            raise ValueError(
                "Failed to parse uid for speaker-wise sampling. "
                f"Expected at least 7 '_' separated tokens, got {uid}"
            )
        return parts[0], parts[1], parts[2], parts[4]

    def _build_speaker_indexes(self, pool_keys: List[str]):
        speaker_index: Dict[str, List[str]] = {}
        utt_index: Dict[str, Dict[str, str]] = {}
        for key in pool_keys:
            spk1, spk2, utt1, utt2 = self._parse_uid_fields(key)
            spk_to_utt = {spk1: utt1, spk2: utt2}
            speakers = list(spk_to_utt.keys())
            if len(speakers) != 2:
                raise ValueError(
                    "Exactly 2 speakers are required for speaker-wise sampling: "
                    f"{key} -> {speakers}"
                )
            for spk in speakers:
                speaker_index.setdefault(spk, []).append(key)
            utt_index[key] = spk_to_utt
        return speaker_index, utt_index

    def _sample_uid_for_speaker(
        self,
        target_speaker: str,
        anchor_utt: str,
        split: str,
        rng: np.random.Generator,
        avoid_uid: Optional[str] = None,
    ) -> str:
        if self._speaker_index_by_split:
            if split not in self._speaker_uid_index:
                raise ValueError(f"speaker index not found for split {split}")
            speaker_index = self._speaker_uid_index[split]
            utt_index = self._uid_to_speaker_utt[split]
        else:
            speaker_index = self._speaker_uid_index
            utt_index = self._uid_to_speaker_utt
        if target_speaker not in speaker_index:
            raise ValueError(
                f"target speaker not found in speaker index: {target_speaker}, split={split}"
            )

        candidates = []
        for cand_uid in speaker_index[target_speaker]:
            if avoid_uid is not None and cand_uid == avoid_uid:
                continue
            cand_spk_to_utt = utt_index[cand_uid]
            if cand_spk_to_utt[target_speaker] != anchor_utt:
                candidates.append(cand_uid)

        if len(candidates) == 0:
            raise ValueError(
                "No valid speaker-conditioned candidate found with different utterance: "
                f"speaker={target_speaker}, anchor_utt={anchor_utt}, split={split}"
            )

        return str(rng.choice(candidates))

    def _extract_speaker_streams(
        self, bundle: Dict[str, Union[np.ndarray, str, int, float]], cand_uid: str, target_speaker: str
    ):
        spk1, spk2, _, _ = self._parse_uid_fields(cand_uid)
        if target_speaker == spk1:
            return bundle["s1_base"], bundle["s1_temp"]
        if target_speaker == spk2:
            return bundle["s2_base"], bundle["s2_temp"]
        raise ValueError(
            f"target speaker {target_speaker} is not contained in candidate uid {cand_uid}"
        )

    def _sample_random_rir_uid(
        self,
        split: str,
        rng: np.random.Generator,
        avoid_uid: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        if isinstance(self._contrastive_pool_keys, dict):
            if split not in self._contrastive_pool_keys:
                raise ValueError(f"RIR pool not found for split {split}")
            pool_keys = self._contrastive_pool_keys[split]
            pool_rir = self.contrastive_pool_rir[split]
            pool_room = self.contrastive_pool_room_param[split]
        else:
            pool_keys = self._contrastive_pool_keys
            pool_rir = self.contrastive_pool_rir
            pool_room = self.contrastive_pool_room_param
        candidates = [k for k in pool_keys if k != avoid_uid]
        if len(candidates) == 0:
            raise ValueError(f"No available RIR candidates for split={split}")
        rir_uid = str(rng.choice(candidates))
        return pool_rir[rir_uid], pool_room[rir_uid], rir_uid

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

        # Use strict truncation to the shortest stream to avoid shape mismatch.
        target_len = min(len(noise_samples), len(s1_samples), len(s2_samples))
        if len(noise_samples) != target_len:
            noise_samples = noise_samples[:target_len]
        if len(s1_samples) != target_len:
            s1_samples = s1_samples[:target_len]
        if len(s2_samples) != target_len:
            s2_samples = s2_samples[:target_len]

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

    def _speech_process_contrastive(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        if self.contrastive_num_neg != 1:
            raise ValueError(
                "NpzSwapRirSamePairNewPreprocessor supports contrastive_num_neg=1"
            )

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

        pool_npz, _ = self._get_npz_pool(split)
        anchor_spk1, anchor_spk2, anchor_utt1, anchor_utt2 = self._parse_uid_fields(uid)

        data2_uid_spk1 = self._sample_uid_for_speaker(
            target_speaker=anchor_spk1,
            anchor_utt=anchor_utt1,
            split=split,
            rng=rng,
            avoid_uid=uid,
        )
        data2_uid_spk2 = self._sample_uid_for_speaker(
            target_speaker=anchor_spk2,
            anchor_utt=anchor_utt2,
            split=split,
            rng=rng,
            avoid_uid=uid,
        )

        bundle_cache: Dict[str, Dict[str, Union[np.ndarray, str, int, float]]] = {}
        for sampled_uid in (data2_uid_spk1, data2_uid_spk2):
            if sampled_uid not in bundle_cache:
                bundle_cache[sampled_uid] = self._load_npz_bundle(pool_npz[sampled_uid])

        bundle_spk1 = bundle_cache[data2_uid_spk1]
        bundle_spk2 = bundle_cache[data2_uid_spk2]

        for sampled_uid, sampled_bundle in (
            (data2_uid_spk1, bundle_spk1),
            (data2_uid_spk2, bundle_spk2),
        ):
            sampled_sr = sampled_bundle["sample_rate"]
            if sampled_sr != self.sample_rate:
                raise ValueError(
                    f"metadata sample_rate {sampled_sr} != preprocessor sample_rate "
                    f"{self.sample_rate} for uid={sampled_uid}"
                )

        data2_s1_base, data2_s1_temp = self._extract_speaker_streams(
            bundle_spk1, data2_uid_spk1, anchor_spk1
        )
        data2_s2_base, data2_s2_temp = self._extract_speaker_streams(
            bundle_spk2, data2_uid_spk2, anchor_spk2
        )

        neg_rir_path, neg_room_path, neg_rir_uid = self._sample_random_rir_uid(
            split=split, rng=rng, avoid_uid=uid
        )
        (
            neg_rir_npz,
            neg_room_dim,
            neg_mic_pos,
            neg_s1_pos,
            neg_s2_pos,
            neg_t60,
            neg_room_fs,
        ) = self._load_room_and_rir(neg_rir_path, neg_room_path)

        speech_mix_pos, s1_pos_samples, s2_pos_samples = self._synthesize_mix(
            sample_rate=sample_rate,
            data_len=data_len,
            start_samp_16k=start_samp_16k,
            wsjmix_scale=wsjmix_scale,
            wham_speech_scale=wham_speech_scale,
            wham_noise_scale=wham_noise_scale,
            mono=mono,
            s1_base=np.asarray(data2_s1_base, dtype=np.float64),
            s2_base=np.asarray(data2_s2_base, dtype=np.float64),
            s1_temp=np.asarray(data2_s1_temp, dtype=np.float64),
            s2_temp=np.asarray(data2_s2_temp, dtype=np.float64),
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

        speech_mix_neg, s1_neg_samples, s2_neg_samples = self._synthesize_mix(
            sample_rate=sample_rate,
            data_len=data_len,
            start_samp_16k=start_samp_16k,
            wsjmix_scale=wsjmix_scale,
            wham_speech_scale=wham_speech_scale,
            wham_noise_scale=wham_noise_scale,
            mono=mono,
            s1_base=np.asarray(data2_s1_base, dtype=np.float64),
            s2_base=np.asarray(data2_s2_base, dtype=np.float64),
            s1_temp=np.asarray(data2_s1_temp, dtype=np.float64),
            s2_temp=np.asarray(data2_s2_temp, dtype=np.float64),
            noise_base=noise_base,
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
            speech_segment = self.speech_segment // self.sample_rate * sample_rate
            crop_posneg = self._random_crop_range(
                {
                    self.speech_ref_name_prefix + "1": s1_pos_samples,
                    self.speech_ref_name_prefix + "2": s2_pos_samples,
                },
                self.num_spk,
                speech_segment,
                uid=f"{data2_uid_spk1}__{data2_uid_spk2}__{neg_rir_uid}",
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


class NpzSwapRirAblationSameUidPreprocessor(NpzSwapRirSamePairNewPreprocessor):
    """RIR-swap ablation preprocessor with same-uid anchor/positive.

    Anchor  : data1 (speech/noise/RIR)
    Positive: data1 (speech/noise/RIR)  # identical to anchor
    Negative: data1 speech/noise + random RIR
    """

    def _speech_process_contrastive(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        if self.contrastive_num_neg != 1:
            raise ValueError(
                "NpzSwapRirAblationSameUidPreprocessor supports contrastive_num_neg=1"
            )

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
        speech_anchor_full = self._apply_postprocess(
            uid, speech_mix, s1_samples, s2_samples, crop=crop
        )
        speech_pos = np.copy(speech_anchor_full)
        speech_anchor = speech_anchor_full
        if self.anchor_single_channel and speech_anchor.ndim > 1:
            speech_anchor = speech_anchor[:, 0]

        neg_rir_path, neg_room_path, _ = self._sample_random_rir_uid(
            split=split, rng=rng, avoid_uid=uid
        )
        (
            neg_rir_npz,
            neg_room_dim,
            neg_mic_pos,
            neg_s1_pos,
            neg_s2_pos,
            neg_t60,
            neg_room_fs,
        ) = self._load_room_and_rir(neg_rir_path, neg_room_path)

        speech_mix_neg, s1_neg_samples, s2_neg_samples = self._synthesize_mix(
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
            s1_scale_factor=1.0,
            s2_scale_factor=1.0,
        )
        speech_neg = self._apply_postprocess(
            uid, speech_mix_neg, s1_neg_samples, s2_neg_samples, crop=crop
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
