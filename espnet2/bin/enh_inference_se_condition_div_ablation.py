#!/usr/bin/env python3
import argparse
import logging
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import yaml
from tqdm import trange
from typeguard import typechecked

from espnet2.enh.diffusion_enh import ESPnetDiffusionModel
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh_se_condition_div import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.train.preprocessor_npz_ablation import NpzSpatialAblationPreprocessor
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args

EPS = torch.finfo(torch.get_default_dtype()).eps


def get_train_config(train_config, model_file=None):
    if train_config is None:
        assert model_file is not None, (
            "The argument 'model_file' must be provided "
            "if the argument 'train_config' is not specified."
        )
        train_config = Path(model_file).parent / "config.yaml"
    else:
        train_config = Path(train_config)
    return train_config


def recursive_dict_update(dict_org, dict_patch, verbose=False, log_prefix=""):
    """Update `dict_org` with `dict_patch` in-place recursively."""
    for key, value in dict_patch.items():
        if key not in dict_org:
            if verbose:
                logging.info(
                    "Overwriting config: [{}{}]: None -> {}".format(
                        log_prefix, key, value
                    )
                )
            dict_org[key] = value
        elif isinstance(value, dict):
            recursive_dict_update(
                dict_org[key], value, verbose=verbose, log_prefix=f"{key}."
            )
        else:
            if verbose and dict_org[key] != value:
                logging.info(
                    "Overwriting config: [{}{}]: {} -> {}".format(
                        log_prefix, key, dict_org[key], value
                    )
                )
            dict_org[key] = value


def build_model_from_args_and_file(task, args, model_file, device):
    model = task.build_model(args)
    if not isinstance(model, AbsESPnetModel):
        raise RuntimeError(
            f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
        )
    model.to(device)
    if model_file is not None:
        if device == "cuda":
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            device = f"cuda:{torch.cuda.current_device()}"
        model.load_state_dict(torch.load(model_file, map_location=device))
    return model


class SeparateSpeech:
    """SeparateSpeech class

    Examples:
        >>> import soundfile
        >>> separate_speech = SeparateSpeech("enh_config.yml", "enh.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> separate_speech(audio)
        [separated_audio1, separated_audio2, ...]

    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        inference_config: Union[Path, str, None] = None,
        segment_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        ref_channel: Optional[int] = None,
        normalize_output_wav: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
        enh_s2t_task: bool = False,
    ):

        task = EnhancementTask if not enh_s2t_task else EnhS2TTask

        # 1. Build Enh model

        if inference_config is None:
            enh_model, enh_train_args = task.build_model_from_file(
                train_config, model_file, device
            )
        else:
            # Overwrite model attributes
            train_config = get_train_config(train_config, model_file=model_file)
            with train_config.open("r", encoding="utf-8") as f:
                train_args = yaml.safe_load(f)

            with Path(inference_config).open("r", encoding="utf-8") as f:
                infer_args = yaml.safe_load(f)

            if enh_s2t_task:
                arg_list = ("enh_encoder", "enh_separator", "enh_decoder")
            else:
                arg_list = ("encoder", "separator", "decoder")
            supported_keys = list(chain(*[[k, k + "_conf"] for k in arg_list]))
            for k in infer_args.keys():
                if k not in supported_keys:
                    raise ValueError(
                        "Only the following top-level keys are supported: %s"
                        % ", ".join(supported_keys)
                    )

            recursive_dict_update(train_args, infer_args, verbose=True)
            enh_train_args = argparse.Namespace(**train_args)
            enh_model = build_model_from_args_and_file(
                task, enh_train_args, model_file, device
            )

        if enh_s2t_task:
            enh_model = enh_model.enh_model
        enh_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.enh_train_args = enh_train_args
        self.enh_model = enh_model

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = normalize_segment_scale
        self.normalize_output_wav = normalize_output_wav
        self.show_progressbar = show_progressbar

        self.num_spk = enh_model.num_spk
        task = "enhancement" if self.num_spk == 1 else "separation"

        # reference channel for processing multi-channel speech
        if ref_channel is not None:
            logging.info(
                "Overwrite enh_model.separator.ref_channel with {}".format(ref_channel)
            )
            enh_model.separator.ref_channel = ref_channel
            if hasattr(enh_model.separator, "beamformer"):
                enh_model.separator.beamformer.ref_channel = ref_channel
            self.ref_channel = ref_channel
        else:
            self.ref_channel = enh_model.ref_channel

        self.segmenting = segment_size is not None and hop_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speech %s" % task)
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speech %s on the input" % task)

    def _compute_spatial_embedding(
        self, speech_mix_mc: torch.Tensor, speech_lengths: torch.Tensor
    ) -> torch.Tensor:
        if (
            not hasattr(self.enh_model, "spatial_encoder")
            or self.enh_model.spatial_encoder is None
            or not hasattr(self.enh_model, "spatial_encoder_encoder")
            or self.enh_model.spatial_encoder_encoder is None
        ):
            raise ValueError(
                "separator.use_spatial_encoder=True, but "
                "enh_model.spatial_encoder or enh_model.spatial_encoder_encoder is not configured."
            )

        encoder_input = speech_mix_mc
        num_channels_arg = None
        if speech_mix_mc.dim() == 3 and speech_mix_mc.shape[2] == 1:
            # Keep 1ch oracle behavior aligned with the base inference path:
            # spatial_encoder_encoder expects [B, T] for single-channel input.
            encoder_input = speech_mix_mc.squeeze(-1)
            num_channels_arg = 1
        elif speech_mix_mc.dim() == 2:
            num_channels_arg = 1

        try:
            feature_mix_mc, flens_mc = self.enh_model.spatial_encoder_encoder(
                encoder_input, speech_lengths, fs=None
            )
        except TypeError:
            feature_mix_mc, flens_mc = self.enh_model.spatial_encoder_encoder(
                encoder_input, speech_lengths
            )

        pooling = getattr(self.enh_model, "spatial_encoder_pooling", True)
        try:
            return self.enh_model.spatial_encoder(
                feature_mix_mc,
                flens_mc,
                num_channels=num_channels_arg,
                pooling=pooling,
            )
        except TypeError:
            return self.enh_model.spatial_encoder(
                feature_mix_mc,
                flens_mc,
                num_channels=num_channels_arg,
            )

    @torch.no_grad()
    @typechecked
    def __call__(
        self, speech_mix: Union[torch.Tensor, np.ndarray], fs: int = 8000, **kwargs
    ) -> List[Union[torch.Tensor, np.array]]:
        """Inference.

        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
            kwargs:
                speech_mix_mc: Optional multi-channel mix for spatial branch.
                spatial_embedding_override: Optional precomputed spatial embedding [B,D] or [D].
        """

        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if getattr(self.enh_model, "normalize_variance_per_ch", False):
            dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_
        elif getattr(self.enh_model, "normalize_variance", False):
            dim = (1, 2) if speech_mix.ndim > 2 else 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_

        category = kwargs.get("utt2category", None)
        if (
            self.enh_model.categories
            and category is not None
            and category[0].item() not in self.enh_model.categories
        ):
            raise ValueError(f"Category '{category}' is not listed in self.categories")

        additional = {}
        if category is not None:
            cat = self.enh_model.categories[category[0].item()]
            print(f"category: {cat}", flush=True)
            if cat.endswith("_reverb"):
                additional["mode"] = "dereverb"
            else:
                additional["mode"] = "no_dereverb"

        has_spatial = (
            getattr(self.enh_model.separator, "use_spatial_encoder", False)
            and getattr(self.enh_model, "spatial_encoder", None) is not None
            and getattr(self.enh_model, "spatial_encoder_encoder", None) is not None
        )

        spatial_embedding_override = kwargs.get("spatial_embedding_override", None)
        if spatial_embedding_override is not None:
            if isinstance(spatial_embedding_override, np.ndarray):
                spatial_embedding_override = torch.as_tensor(spatial_embedding_override)
            spatial_embedding_override = spatial_embedding_override.to(
                getattr(torch, self.dtype)
            )
            spatial_embedding_override = to_device(
                spatial_embedding_override, device=self.device
            )
            if spatial_embedding_override.dim() == 1:
                spatial_embedding_override = spatial_embedding_override.unsqueeze(0)
            if spatial_embedding_override.size(0) == 1 and batch_size > 1:
                spatial_embedding_override = spatial_embedding_override.repeat(
                    batch_size, 1
                )
            if spatial_embedding_override.size(0) != batch_size:
                raise ValueError(
                    "spatial_embedding_override batch size mismatch: "
                    f"{spatial_embedding_override.size(0)} != {batch_size}"
                )

        speech_mix_mc = kwargs.get("speech_mix_mc", None)
        if has_spatial and spatial_embedding_override is None:
            if speech_mix_mc is None:
                if speech_mix.dim() == 3 and speech_mix.shape[-1] >= 2:
                    speech_mix_mc = speech_mix
                elif speech_mix.dim() == 2:
                    # Keep oracle behavior for 1ch input by using the same mixture
                    # for spatial embedding computation.
                    speech_mix_mc = speech_mix.unsqueeze(-1)
            else:
                if isinstance(speech_mix_mc, np.ndarray):
                    speech_mix_mc = torch.as_tensor(speech_mix_mc)
                speech_mix_mc = speech_mix_mc.to(getattr(torch, self.dtype))

            if speech_mix_mc is None:
                raise ValueError(
                    "This model requires spatial conditioning, but 'speech_mix_mc' "
                    "was not provided and speech_mix is not multi-channel."
                )

            speech_mix_mc = to_device(speech_mix_mc, device=self.device)
            if speech_mix_mc.dim() == 2:
                speech_mix_mc = speech_mix_mc.unsqueeze(-1)
            if speech_mix_mc.dim() != 3:
                raise ValueError(
                    f"speech_mix_mc must be [B,T,C] or [B,T], got {speech_mix_mc.shape}"
                )

            target_len = int(lengths.max().item())
            cur_len = speech_mix_mc.size(1)
            if cur_len < target_len:
                pad = speech_mix_mc.new_zeros(
                    batch_size, target_len - cur_len, speech_mix_mc.size(2)
                )
                speech_mix_mc = torch.cat([speech_mix_mc, pad], dim=1)
            else:
                speech_mix_mc = speech_mix_mc[:, :target_len, :]

        if speech_mix.dim() == 3 and speech_mix.shape[-1] >= 2:
            speech_mix = speech_mix[:, :, self.ref_channel]

        if self.segmenting and lengths[0] > self.segment_size * fs:
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )

                additional_seg = dict(additional)
                if has_spatial:
                    if spatial_embedding_override is not None:
                        additional_seg["spatial_embedding"] = spatial_embedding_override
                    else:
                        if en >= lengths[0]:
                            pad_shape_mc = (batch_size, T, speech_mix_mc.shape[2])
                            speech_mix_mc_seg = speech_mix_mc.new_zeros(pad_shape_mc)
                            speech_mix_mc_seg[:, :t, :] = speech_mix_mc[:, st:en, :]
                        else:
                            speech_mix_mc_seg = speech_mix_mc[:, st:en, :]
                        additional_seg["spatial_embedding"] = self._compute_spatial_embedding(
                            speech_mix_mc_seg, lengths_seg
                        )

                feats, f_lens = self.enh_model.encoder(speech_seg, lengths_seg)
                if isinstance(self.enh_model, ESPnetDiffusionModel):
                    feats = [self.enh_model.enhance(feats)]
                else:
                    feats, _, _ = self.enh_model.separator(feats, f_lens, additional_seg)
                processed_wav = [
                    self.enh_model.decoder(f, lengths_seg)[0] for f in feats
                ]
                speech_seg_ = speech_seg

                if self.normalize_segment_scale:
                    mix_energy = torch.sqrt(
                        torch.mean(speech_seg_[:, :t].pow(2), dim=1, keepdim=True)
                    )
                    enh_energy = torch.sqrt(
                        torch.mean(
                            sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True
                        )
                    )
                    processed_wav = [w * (mix_energy / enh_energy) for w in processed_wav]
                enh_waves.append(torch.stack(processed_wav, dim=0))

            waves = enh_waves[0]
            for i in range(1, num_segments):
                perm = self.cal_permumation(
                    waves[:, :, -overlap_length:],
                    enh_waves[i][:, :, :overlap_length],
                    criterion="si_snr",
                )
                for batch in range(batch_size):
                    enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                waves[:, :, -overlap_length:] = (
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            additional_full = dict(additional)
            if has_spatial:
                if spatial_embedding_override is not None:
                    additional_full["spatial_embedding"] = spatial_embedding_override
                else:
                    additional_full["spatial_embedding"] = self._compute_spatial_embedding(
                        speech_mix_mc, lengths
                    )

            feats, f_lens = self.enh_model.encoder(speech_mix, lengths)
            if isinstance(self.enh_model, ESPnetDiffusionModel):
                feats = [self.enh_model.enhance(feats)]
            else:
                feats, _, _ = self.enh_model.separator(feats, f_lens, additional_full)
            waves = [self.enh_model.decoder(f, lengths)[0] for f in feats]

        if getattr(self.enh_model, "normalize_variance_per_ch", False):
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_[:, :, self.ref_channel]
            waves = [w * mix_std_ for w in waves]
        elif getattr(self.enh_model, "normalize_variance", False):
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_.squeeze(2)
            waves = [w * mix_std_ for w in waves]

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                for w in waves
            ]
        else:
            waves = [w.cpu().numpy() for w in waves]

        return waves

    @torch.no_grad()
    def cal_permumation(self, ref_wavs, enh_wavs, criterion="si_snr"):
        """Calculate the permutation between seaprated streams in two adjacent segments.

        Args:
            ref_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            enh_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            criterion (str): one of ("si_snr", "mse", "corr)
        Returns:
            perm (torch.Tensor): permutation for enh_wavs (Batch, num_spk)
        """

        criterion_class = {"si_snr": SISNRLoss, "mse": FrequencyDomainMSE}[criterion]

        pit_solver = PITSolver(criterion=criterion_class())

        _, _, others = pit_solver(ref_wavs, enh_wavs)
        perm = others["perm"]
        return perm

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build SeparateSpeech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            SeparateSpeech: SeparateSpeech instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return SeparateSpeech(**kwargs)


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    inference_config: Optional[str],
    allow_variable_data_keys: bool,
    segment_size: Optional[float],
    hop_size: Optional[float],
    normalize_segment_scale: bool,
    show_progressbar: bool,
    ref_channel: Optional[int],
    normalize_output_wav: bool,
    enh_s2t_task: bool,
    spatial_ablation_mode: str,
    ablation_npz_data_dir: Optional[str],
    ablation_pool_npz_scp: Optional[str],
    ablation_seed: int,
    ablation_epoch: int,
    ablation_mean_embedding_path: Optional[str],
    ablation_mean_source_mode: str,
):
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build separate_speech
    separate_speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        inference_config=inference_config,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        show_progressbar=show_progressbar,
        ref_channel=ref_channel,
        normalize_output_wav=normalize_output_wav,
        device=device,
        dtype=dtype,
        enh_s2t_task=enh_s2t_task,
    )
    separate_speech = SeparateSpeech.from_pretrained(
        model_tag=model_tag,
        **separate_speech_kwargs,
    )

    supported_modes = ("oracle", "swap_rir", "swap_audio", "rand_sample", "mean")
    if spatial_ablation_mode not in supported_modes:
        raise ValueError(
            f"Unsupported --spatial_ablation_mode: {spatial_ablation_mode}. "
            f"Supported: {supported_modes}"
        )
    if ablation_mean_source_mode not in (
        "oracle",
        "swap_rir",
        "swap_audio",
        "rand_sample",
    ):
        raise ValueError(
            "Unsupported --ablation_mean_source_mode: "
            f"{ablation_mean_source_mode}. Supported: "
            "('oracle', 'swap_rir', 'swap_audio', 'rand_sample')"
        )

    def _build_loader():
        return EnhancementTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=EnhancementTask.build_preprocess_fn(
                separate_speech.enh_train_args, False
            ),
            collate_fn=EnhancementTask.build_collate_fn(
                separate_speech.enh_train_args, False
            ),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

    npz_ablator = None
    need_npz = spatial_ablation_mode in ("swap_rir", "swap_audio", "rand_sample") or (
        spatial_ablation_mode == "mean"
        and ablation_mean_source_mode in ("swap_rir", "swap_audio", "rand_sample")
    )
    if need_npz:
        if not ablation_npz_data_dir:
            raise ValueError(
                "--ablation_npz_data_dir is required for "
                "swap_rir/swap_audio/rand_sample modes"
            )
        npz_dir = Path(ablation_npz_data_dir)
        npz_scp = npz_dir / "npz.scp"
        if not npz_scp.exists():
            raise FileNotFoundError(f"Missing npz.scp: {npz_scp}")
        pool_npz_scp = (
            Path(ablation_pool_npz_scp)
            if ablation_pool_npz_scp is not None
            else npz_scp
        )
        if not pool_npz_scp.exists():
            raise FileNotFoundError(f"Missing pool npz.scp: {pool_npz_scp}")

        npz_dir_name = npz_dir.name
        if "mix_clean" in npz_dir_name:
            ablation_mix_type = "clean"
        elif "mix_single" in npz_dir_name:
            ablation_mix_type = "single"
        else:
            ablation_mix_type = "both"
        logging.info(
            "Using NpzSpatialAblationPreprocessor mix_type=%s for %s",
            ablation_mix_type,
            npz_dir_name,
        )
        npz_ablator = NpzSpatialAblationPreprocessor(
            npz_scp=str(npz_scp),
            pool_npz_scp=str(pool_npz_scp),
            sample_rate=int(fs),
            seed=ablation_seed,
            epoch=ablation_epoch,
            mix_type=ablation_mix_type,
        )

    def _as_numpy_first_batch(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value[0])

    def _build_spatial_mix(uid: str, speech_mix_value, source_mode: str) -> np.ndarray:
        if source_mode == "oracle":
            spatial_mix = _as_numpy_first_batch(speech_mix_value)
            if spatial_mix.ndim == 1:
                spatial_mix = spatial_mix[:, None]
            return np.asarray(spatial_mix, dtype=np.float32)
        if npz_ablator is None:
            raise ValueError(
                f"Mode '{source_mode}' requires NPZ ablator, but it is not initialized."
            )
        return npz_ablator.build_spatial_mix(uid=uid, mode=source_mode)

    mean_embedding = None
    if spatial_ablation_mode == "mean":
        if ablation_mean_embedding_path:
            mean_embedding_np = np.load(ablation_mean_embedding_path)
            mean_embedding_np = np.asarray(mean_embedding_np, dtype=np.float32).reshape(-1)
            mean_embedding = torch.as_tensor(mean_embedding_np, dtype=torch.float32)
            logging.info(
                "Loaded mean embedding from %s (dim=%d)",
                ablation_mean_embedding_path,
                mean_embedding.numel(),
            )
        else:
            logging.info(
                "Computing mean embedding from data (source_mode=%s)",
                ablation_mean_source_mode,
            )
            loader_for_mean = _build_loader()
            emb_sum = None
            emb_count = 0
            for keys, batch in loader_for_mean:
                batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
                uid = keys[0]
                spatial_mix = _build_spatial_mix(
                    uid=uid,
                    speech_mix_value=batch["speech_mix"],
                    source_mode=ablation_mean_source_mode,
                )
                spatial_mix_t = torch.as_tensor(
                    spatial_mix, dtype=getattr(torch, dtype)
                ).unsqueeze(0)
                lengths_t = torch.as_tensor([spatial_mix_t.size(1)], dtype=torch.long)
                spatial_mix_t = to_device(spatial_mix_t, device=device)
                lengths_t = to_device(lengths_t, device=device)
                emb = (
                    separate_speech._compute_spatial_embedding(spatial_mix_t, lengths_t)
                    .detach()
                    .cpu()
                )
                if emb.dim() == 2:
                    emb = emb[0]
                if emb_sum is None:
                    emb_sum = torch.zeros_like(emb)
                emb_sum += emb
                emb_count += 1
            if emb_count == 0:
                raise RuntimeError("No utterances found for mean embedding computation")
            mean_embedding = emb_sum / emb_count
            logging.info(
                "Computed mean embedding from %d utterances (dim=%d)",
                emb_count,
                mean_embedding.numel(),
            )

    loader = _build_loader()

    # 4. Start for-loop
    output_dir: Path = Path(output_dir).expanduser().resolve()
    writers = []
    for i in range(separate_speech.num_spk):
        writers.append(
            SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
        )

    import tqdm

    for i, (keys, batch) in tqdm.tqdm(enumerate(loader)):
        logging.info(f"[{i}] Enhancing {keys}")
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        uid = keys[0]
        if spatial_ablation_mode in ("swap_rir", "swap_audio", "rand_sample"):
            spatial_mix = _build_spatial_mix(
                uid=uid,
                speech_mix_value=batch["speech_mix"],
                source_mode=spatial_ablation_mode,
            )
            batch["speech_mix_mc"] = np.expand_dims(spatial_mix, axis=0)
        elif spatial_ablation_mode == "mean":
            batch["spatial_embedding_override"] = mean_embedding.numpy()[None, :]

        waves = separate_speech(**batch, fs=fs)
        for spk, w in enumerate(waves):
            for b in range(batch_size):
                writers[spk][keys[b]] = fs, w[b]

    for writer in writers:
        writer.close()


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Output data related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Whether to normalize the predicted wav to [-1~1]",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )
    group.add_argument(
        "--inference_config",
        type=str_or_none,
        default=None,
        help="Optional configuration file for overwriting enh model attributes "
        "during inference",
    )
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="enhancement and asr joint model",
    )

    group = parser.add_argument_group("Data loading related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group = parser.add_argument_group("SeparateSpeech related")
    group.add_argument(
        "--segment_size",
        type=float,
        default=None,
        help="Segment length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--hop_size",
        type=float,
        default=None,
        help="Hop length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--normalize_segment_scale",
        type=str2bool,
        default=True,
        help="Whether to normalize the energy of the separated streams in each segment",
    )
    group.add_argument(
        "--show_progressbar",
        type=str2bool,
        default=False,
        help="Whether to show a progress bar when performing segment-wise speech "
        "enhancement/separation",
    )
    group.add_argument(
        "--ref_channel",
        type=int,
        default=None,
        help="If not None, this will overwrite the ref_channel defined in the "
        "separator module (for multi-channel speech processing)",
    )

    group = parser.add_argument_group("Spatial Ablation")
    group.add_argument(
        "--spatial_ablation_mode",
        type=str,
        default="oracle",
        choices=("oracle", "swap_rir", "swap_audio", "rand_sample", "mean"),
        help=(
            "Ablation mode for spatial conditioning. "
            "oracle: default behavior, "
            "swap_rir: anchor speech/noise with sampled different RIR, "
            "swap_audio: sampled different speech/noise with anchor RIR, "
            "rand_sample: sampled different sample + its RIR, "
            "mean: fixed mean embedding."
        ),
    )
    group.add_argument(
        "--ablation_npz_data_dir",
        type=str_or_none,
        default=None,
        help="Kaldi-style NPZ data dir for the current dset (contains npz.scp).",
    )
    group.add_argument(
        "--ablation_pool_npz_scp",
        type=str_or_none,
        default=None,
        help="Optional NPZ pool scp for sampling alternative utterances.",
    )
    group.add_argument(
        "--ablation_seed",
        type=int,
        default=1234,
        help="Seed for deterministic NPZ sampling in spatial ablation.",
    )
    group.add_argument(
        "--ablation_epoch",
        type=int,
        default=0,
        help="Epoch-like offset used by deterministic NPZ sampling.",
    )
    group.add_argument(
        "--ablation_mean_embedding_path",
        type=str_or_none,
        default=None,
        help="Optional .npy path for fixed mean embedding in mode=mean.",
    )
    group.add_argument(
        "--ablation_mean_source_mode",
        type=str,
        default="oracle",
        choices=("oracle", "swap_rir", "swap_audio", "rand_sample"),
        help="Source mode used when computing mean embedding from data.",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
