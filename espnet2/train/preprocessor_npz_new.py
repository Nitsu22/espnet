"""Additional NPZ preprocessors for stricter same-pair contrastive sampling."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.train.preprocessor_npz import NpzSwapRirPreprocessor, _as_str, _np_item


class NpzSwapRirSamePairNewPreprocessor(NpzSwapRirPreprocessor):
    """RIR-swap preprocessor with same-speaker-pair data2 sampling.

    Anchor  : data1 (speech/noise/RIR)
    Positive: data2 speech + data1 noise + data1 RIR
    Negative: data2 speech + data1 noise + data2 RIR

    data2 is sampled from the same unordered speaker pair as data1, with
    both speaker utterances different from data1.
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

        self._same_pair_by_split = isinstance(self._contrastive_pool_npz_keys, dict)
        self._same_pair_uid_index = None
        self._uid_to_speaker_utt = None
        if self.contrastive_enable:
            if self._same_pair_by_split:
                self._same_pair_uid_index = {}
                self._uid_to_speaker_utt = {}
                for split_key, pool_keys in self._contrastive_pool_npz_keys.items():
                    pair_index, utt_index = self._build_pair_indexes(pool_keys)
                    self._same_pair_uid_index[split_key] = pair_index
                    self._uid_to_speaker_utt[split_key] = utt_index
            else:
                pair_index, utt_index = self._build_pair_indexes(
                    self._contrastive_pool_npz_keys
                )
                self._same_pair_uid_index = pair_index
                self._uid_to_speaker_utt = utt_index

    @staticmethod
    def _speaker_pair_key(spk1: str, spk2: str) -> Tuple[str, str]:
        if spk1 <= spk2:
            return (spk1, spk2)
        return (spk2, spk1)

    @staticmethod
    def _parse_uid_speaker_utt(uid: str) -> Dict[str, str]:
        parts = uid.split("_")
        if len(parts) < 7:
            raise ValueError(
                "Failed to parse uid for same-pair sampling. "
                f"Expected at least 7 '_' separated tokens, got {uid}"
            )
        spk1 = parts[0]
        spk2 = parts[1]
        utt1 = parts[2]
        utt2 = parts[4]
        return {spk1: utt1, spk2: utt2}

    def _build_pair_indexes(self, pool_keys: List[str]):
        pair_index: Dict[Tuple[str, str], List[str]] = {}
        utt_index: Dict[str, Dict[str, str]] = {}
        for key in pool_keys:
            spk_to_utt = self._parse_uid_speaker_utt(key)
            speakers = sorted(spk_to_utt.keys())
            if len(speakers) != 2:
                raise ValueError(
                    "Exactly 2 speakers are required for same-pair sampling: "
                    f"{key} -> {speakers}"
                )
            pair_key = self._speaker_pair_key(speakers[0], speakers[1])
            pair_index.setdefault(pair_key, []).append(key)
            utt_index[key] = spk_to_utt
        return pair_index, utt_index

    def _resolve_pair_context(self, split: str):
        if self._same_pair_by_split:
            if split not in self._same_pair_uid_index:
                raise ValueError(f"same-pair index not found for split {split}")
            return self._same_pair_uid_index[split], self._uid_to_speaker_utt[split]
        return self._same_pair_uid_index, self._uid_to_speaker_utt

    def _sample_same_pair_uid(
        self,
        uid: str,
        split: str,
        rng: np.random.Generator,
    ) -> str:
        pair_index, utt_index = self._resolve_pair_context(split)
        if uid not in utt_index:
            raise ValueError(f"uid not found in same-pair index: {uid}")

        anchor_spk_to_utt = utt_index[uid]
        speakers = sorted(anchor_spk_to_utt.keys())
        pair_key = self._speaker_pair_key(speakers[0], speakers[1])
        if pair_key not in pair_index:
            raise ValueError(f"speaker pair not found in same-pair index: {pair_key}")

        candidates = []
        for cand_uid in pair_index[pair_key]:
            if cand_uid == uid:
                continue
            cand_spk_to_utt = utt_index[cand_uid]
            if set(cand_spk_to_utt.keys()) != set(anchor_spk_to_utt.keys()):
                continue
            if all(cand_spk_to_utt[spk] != anchor_spk_to_utt[spk] for spk in speakers):
                candidates.append(cand_uid)

        if len(candidates) == 0:
            raise ValueError(
                "No valid same-speaker-pair candidate found under strict both-speaker "
                f"different-utterance constraint for uid={uid}, split={split}, "
                f"pair={pair_key}"
            )

        return str(rng.choice(candidates))

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
        data2_uid = self._sample_same_pair_uid(uid=uid, split=split, rng=rng)
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
            wham_noise_scale=wham_noise_scale,
            mono=data2["mono"],
            s1_base=data2["s1_base"],
            s2_base=data2["s2_base"],
            s1_temp=data2["s1_temp"],
            s2_temp=data2["s2_temp"],
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
            sample_rate=sample_rate2,
            data_len=data2["data_len"],
            start_samp_16k=data2["start_samp_16k"],
            wsjmix_scale=data2["wsjmix_scale"],
            wham_speech_scale=data2["wham_speech_scale"],
            wham_noise_scale=wham_noise_scale,
            mono=data2["mono"],
            s1_base=data2["s1_base"],
            s2_base=data2["s2_base"],
            s1_temp=data2["s1_temp"],
            s2_temp=data2["s2_temp"],
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
