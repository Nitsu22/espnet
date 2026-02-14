import argparse
import copy
import os
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.diar.layers.multi_mask import MultiMask
from espnet2.diar.separator.tcn_separator_nomask import TCNSeparatorNomask
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.diffusion.score_based_diffusion import ScoreModel
from espnet2.enh.diffusion_enh import ESPnetDiffusionModel
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model_se_condition_div import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.criterions.tf_domain import (
    FrequencyDomainAbsCoherence,
    FrequencyDomainDPCL,
    FrequencyDomainL1,
    FrequencyDomainMSE,
)
from espnet2.enh.loss.criterions.time_domain import (
    CISDRLoss,
    MultiResL1SpecLoss,
    SDRLoss,
    SISNRLoss,
    SNRLoss,
    TimeDomainL1,
    TimeDomainMSE,
)
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.dpcl_solver import DPCLSolver
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.mixit_solver import MixITSolver
from espnet2.enh.loss.wrappers.multilayer_pit_solver import MultiLayerPITSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter
from espnet2.enh.separator.conformer_separator import ConformerSeparator
from espnet2.enh.separator.dan_separator import DANSeparator
from espnet2.enh.separator.dc_crn_separator import DC_CRNSeparator
from espnet2.enh.separator.dccrn_separator import DCCRNSeparator
from espnet2.enh.separator.dpcl_e2e_separator import DPCLE2ESeparator
from espnet2.enh.separator.dpcl_separator import DPCLSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.dptnet_separator import DPTNetSeparator
from espnet2.enh.separator.fasnet_separator import FaSNetSeparator
from espnet2.enh.separator.ineube_separator import iNeuBe
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.skim_separator import SkiMSeparator
from espnet2.enh.separator.svoice_separator import SVoiceSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.tfgridnet_separator import TFGridNet
from espnet2.enh.separator.tfgridnetv2_separator import TFGridNetV2
from espnet2.enh.separator.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.separator.tflocoformer_separator_nocashe import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocashe,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_aeafusion import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheAEAFusion,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_aeafusion_all import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheAEAFusionAll,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_aeafusion_first import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheAEAFusionFirst,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_aeafusion_film_first import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheAEAFusionFiLMFirst,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_aeafusion_film_all import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheAEAFusionFiLMAll,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_noaeafusion_all import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheNoAEAFusionAll,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_noaeafusion_film_all import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheNoAEAFusionFiLMAll,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_film import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheFiLM,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_film_first import (
    TFLocoformerSeparator as TFLocoformerSeparatorNocasheFiLMFirst,
)
from espnet2.enh.separator.tflocoformer_separator_mc import TFLocoformerSeparatorMC
from espnet2.enh.separator.tflocoformer_separator_nocashe_mc import (
    TFLocoformerSeparatorMC as TFLocoformerSeparatorNocasheMC,
)
from espnet2.enh.separator.tflocoformer_separator_test import (
    TFLocoformerSeparator as TFLocoformerSeparatorTest,
)
from espnet2.enh.separator.fla_tflocoformer_separator import (
    TFLocoformerSeparator as FLATFLocoformerSeparator,
)
from espnet2.enh.separator.transformer_separator import TransformerSeparator
from espnet2.enh.separator.uses_separator import USESSeparator
from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import AbsSpatialEncoder
from espnet2.enh_se.spatial_encoder.mc_conformer_div_spatial_encoder import (
    MCConformerDivSpatialEncoder,
)
from espnet2.enh_se.spatial_encoder.resnet2d_div_spatial_encoder import (
    ResNet2DDivSpatialEncoder,
)
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.optimizers.optim_groups import add_optimizer_hooks
from espnet2.tasks.abs_task import AbsTask, fairscale, optim_classes
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    DynamicMixingPreprocessor,
    EnhPreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(stft=STFTEncoder, conv=ConvEncoder, same=NullEncoder),
    type_check=AbsEncoder,
    default="stft",
)

separator_choices = ClassChoices(
    name="separator",
    classes=dict(
        asteroid=AsteroidModel_Converter,
        conformer=ConformerSeparator,
        dan=DANSeparator,
        dc_crn=DC_CRNSeparator,
        dccrn=DCCRNSeparator,
        dpcl=DPCLSeparator,
        dpcl_e2e=DPCLE2ESeparator,
        dprnn=DPRNNSeparator,
        dptnet=DPTNetSeparator,
        fasnet=FaSNetSeparator,
        rnn=RNNSeparator,
        skim=SkiMSeparator,
        svoice=SVoiceSeparator,
        tcn=TCNSeparator,
        transformer=TransformerSeparator,
        wpe_beamformer=NeuralBeamformer,
        tcn_nomask=TCNSeparatorNomask,
        ineube=iNeuBe,
        tfgridnet=TFGridNet,
        tfgridnetv2=TFGridNetV2,
        uses=USESSeparator,
        tflocoformer=TFLocoformerSeparator,
        tflocoformer_nocashe=TFLocoformerSeparatorNocashe,
        tflocoformer_nocashe_aeafusion=TFLocoformerSeparatorNocasheAEAFusion,
        tflocoformer_nocashe_aeafusion_all=TFLocoformerSeparatorNocasheAEAFusionAll,
        tflocoformer_nocashe_aeafusion_first=TFLocoformerSeparatorNocasheAEAFusionFirst,
        tflocoformer_nocashe_aeafusion_film_first=TFLocoformerSeparatorNocasheAEAFusionFiLMFirst,
        tflocoformer_nocashe_aeafusion_film_all=TFLocoformerSeparatorNocasheAEAFusionFiLMAll,
        tflocoformer_nocashe_noaeafusion_all=TFLocoformerSeparatorNocasheNoAEAFusionAll,
        tflocoformer_nocashe_noaeafusion_film_all=TFLocoformerSeparatorNocasheNoAEAFusionFiLMAll,
        tflocoformer_nocashe_film=TFLocoformerSeparatorNocasheFiLM,
        tflocoformer_nocashe_film_first=TFLocoformerSeparatorNocasheFiLMFirst,
        tflocoformer_nocashe_mc=TFLocoformerSeparatorNocasheMC,
        tflocoformer_mc=TFLocoformerSeparatorMC,
        tflocoformer_test=TFLocoformerSeparatorTest,
        fla_tflocoformer=FLATFLocoformerSeparator,
    ),
    type_check=AbsSeparator,
    default="rnn",
)

mask_module_choices = ClassChoices(
    name="mask_module",
    classes=dict(multi_mask=MultiMask),
    type_check=AbsMask,
    default="multi_mask",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(stft=STFTDecoder, conv=ConvDecoder, same=NullDecoder),
    type_check=AbsDecoder,
    default="stft",
)

loss_wrapper_choices = ClassChoices(
    name="loss_wrappers",
    classes=dict(
        pit=PITSolver,
        fixed_order=FixedOrderSolver,
        multilayer_pit=MultiLayerPITSolver,
        dpcl=DPCLSolver,
        mixit=MixITSolver,
    ),
    type_check=AbsLossWrapper,
    default=None,
)

criterion_choices = ClassChoices(
    name="criterions",
    classes=dict(
        ci_sdr=CISDRLoss,
        coh=FrequencyDomainAbsCoherence,
        sdr=SDRLoss,
        si_snr=SISNRLoss,
        snr=SNRLoss,
        l1=FrequencyDomainL1,
        dpcl=FrequencyDomainDPCL,
        l1_fd=FrequencyDomainL1,
        l1_td=TimeDomainL1,
        mse=FrequencyDomainMSE,
        mse_fd=FrequencyDomainMSE,
        mse_td=TimeDomainMSE,
        mr_l1_tfd=MultiResL1SpecLoss,
    ),
    type_check=AbsEnhLoss,
    default=None,
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        dynamic_mixing=DynamicMixingPreprocessor,
        enh=EnhPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default=None,
)

# Deffusion-based model related choices
diffusion_choices = ClassChoices(
    name="diffusion_model",
    classes=dict(sgmse=ScoreModel),
    type_check=AbsDiffusion,
    default=None,
)

spatial_encoder_choices = ClassChoices(
    name="spatial_encoder",
    classes=dict(
        mc_conformer_div=MCConformerDivSpatialEncoder,
        resnet2d_div=ResNet2DDivSpatialEncoder,
    ),
    type_check=AbsSpatialEncoder,
    default=None,
    optional=True,
)

spatial_encoder_encoder_choices = ClassChoices(
    name="spatial_encoder_encoder",
    classes=dict(stft=STFTEncoder, conv=ConvEncoder, same=NullEncoder),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)


MAX_REFERENCE_NUM = 100


class EnhancementTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --separator and --separator_conf
        separator_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --mask_module and --mask_module_conf
        mask_module_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
        # --diffusion_model and --diffusion_model_conf
        diffusion_choices,
        # --spatial_encoder and --spatial_encoder_conf
        spatial_encoder_choices,
        # --spatial_encoder_encoder and --spatial_encoder_encoder_conf
        spatial_encoder_encoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )
        group.add_argument(
            "--aea_block_lr",
            type=float,
            default=None,
            help=(
                "Optional LR for separator AEA blocks (separator.aea_blocks). "
                "If None, optim_conf.lr is used."
            ),
        )
        group.add_argument(
            "--non_aea_lr",
            type=float,
            default=None,
            help=(
                "Optional LR for non-AEA parameters when split-LR is enabled. "
                "If None, optim_conf.lr is used."
            ),
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--speech_volume_normalize",
            type=str_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value or range. "
            "e.g. --speech_volume_normalize 1.0 scales it to 1.0.\n"
            "--speech_volume_normalize 0.5_1.0 scales it to a random number in "
            "the range [0.5, 1.0)",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of signal-to-noise ratio (SNR) level in decibel.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--use_reverberant_ref",
            type=str2bool,
            default=False,
            help="Whether to use reverberant speech references "
            "instead of anechoic ones",
        )
        group.add_argument(
            "--num_spk",
            type=int,
            default=1,
            help="Number of speakers in the input signal.",
        )
        group.add_argument(
            "--num_noise_type",
            type=int,
            default=1,
            help="Number of noise types.",
        )
        group.add_argument(
            "--sample_rate",
            type=int,
            default=8000,
            help="Sampling rate of the data (in Hz).",
        )
        group.add_argument(
            "--force_single_channel",
            type=str2bool,
            default=False,
            help="Whether to force all data to be single-channel.",
        )
        group.add_argument(
            "--channel_reordering",
            type=str2bool,
            default=False,
            help="Whether to randomly reorder the channels of the "
            "multi-channel signals.",
        )
        group.add_argument(
            "--categories",
            nargs="+",
            default=[],
            type=str,
            help="The set of all possible categories in the dataset. Used to add the "
            "category information to each sample",
        )
        group.add_argument(
            "--speech_segment",
            type=int_or_none,
            default=None,
            help="Truncate the audios to the specified length (in samples) if not None",
        )
        group.add_argument(
            "--avoid_allzero_segment",
            type=str2bool,
            default=True,
            help="Only used when --speech_segment is specified. If True, make sure "
            "all truncated segments are not all-zero",
        )
        group.add_argument(
            "--flexible_numspk",
            type=str2bool,
            default=False,
            help="Whether to load variable numbers of speakers in each sample. "
            "In this case, only the first-speaker files such as 'spk1.scp' and "
            "'dereverb1.scp' are used, which are expected to have multiple columns. "
            "Other numbered files such as 'spk2.scp' and 'dereverb2.scp' are ignored.",
        )

        group.add_argument(
            "--dynamic_mixing",
            type=str2bool,
            default=False,
            help="Apply dynamic mixing",
        )
        group.add_argument(
            "--utt2spk",
            type=str_or_none,
            default=None,
            help="The file path of utt2spk file. Only used in dynamic_mixing mode.",
        )
        group.add_argument(
            "--dynamic_mixing_gain_db",
            type=float,
            default=0.0,
            help="Random gain (in dB) for dynamic mixing sources",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:

        use_preprocessor = getattr(args, "preprocessor", None) is not None

        if use_preprocessor:
            # TODO(simpleoier): To make this as simple as model parts, e.g. encoder
            if args.preprocessor == "dynamic_mixing":
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    train=train,
                    source_scp=os.path.join(
                        os.path.dirname(args.train_data_path_and_name_and_type[0][0]),
                        args.preprocessor_conf.get("source_scp_name", "spk1.scp"),
                    ),
                    ref_num=args.preprocessor_conf.get(
                        "ref_num", args.separator_conf["num_spk"]
                    ),
                    dynamic_mixing_gain_db=args.preprocessor_conf.get(
                        "dynamic_mixing_gain_db", 0.0
                    ),
                    speech_name=args.preprocessor_conf.get("speech_name", "speech_mix"),
                    speech_ref_name_prefix=args.preprocessor_conf.get(
                        "speech_ref_name_prefix", "speech_ref"
                    ),
                    mixture_source_name=args.preprocessor_conf.get(
                        "mixture_source_name", None
                    ),
                    utt2spk=getattr(args, "utt2spk", None),
                    categories=args.preprocessor_conf.get("categories", None),
                )
            elif args.preprocessor == "enh":
                kwargs = dict(
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=getattr(args, "rir_scp", None),
                    rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                    noise_scp=getattr(args, "noise_scp", None),
                    noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                    noise_db_range=getattr(args, "noise_db_range", "13_15"),
                    short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                    speech_volume_normalize=getattr(
                        args, "speech_volume_normalize", None
                    ),
                    use_reverberant_ref=getattr(args, "use_reverberant_ref", None),
                    num_spk=getattr(args, "num_spk", 1),
                    num_noise_type=getattr(args, "num_noise_type", 1),
                    sample_rate=getattr(args, "sample_rate", 8000),
                    force_single_channel=getattr(args, "force_single_channel", False),
                    channel_reordering=getattr(args, "channel_reordering", False),
                    categories=getattr(args, "categories", None),
                    speech_segment=getattr(args, "speech_segment", None),
                    avoid_allzero_segment=getattr(args, "avoid_allzero_segment", True),
                    flexible_numspk=getattr(args, "flexible_numspk", False),
                )
                kwargs.update(args.preprocessor_conf)
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    train=train, **kwargs
                )
            else:
                raise ValueError(
                    f"Preprocessor type {args.preprocessor} is not supported."
                )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_ref1",)
        else:
            # Inference mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["speech_mix"]
        retval += ["dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["category"]
        retval = tuple(retval)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel:

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        spatial_encoder = None
        spatial_encoder_encoder = None
        spatial_encoder_pooling = True
        spatial_encoder_pretrained = None
        spatial_encoder_trainable = True
        separator_conf = args.separator_conf.copy()
        if getattr(args, "spatial_encoder_encoder", None) is not None:
            spatial_encoder_encoder_conf = getattr(
                args, "spatial_encoder_encoder_conf", {}
            )
            spatial_encoder_encoder = spatial_encoder_encoder_choices.get_class(
                args.spatial_encoder_encoder
            )(**spatial_encoder_encoder_conf)
        if getattr(args, "spatial_encoder", None) is not None:
            if spatial_encoder_encoder is None:
                raise ValueError(
                    "spatial_encoder_encoder must be specified when spatial_encoder is used."
                )
            spatial_encoder_conf = getattr(args, "spatial_encoder_conf", {}).copy()
            spatial_encoder_pooling = spatial_encoder_conf.pop("pooling", True)
            spatial_encoder_pretrained = spatial_encoder_conf.pop("path", None)
            spatial_encoder_trainable = spatial_encoder_conf.pop("trainable", True)
            spatial_encoder = spatial_encoder_choices.get_class(args.spatial_encoder)(
                **spatial_encoder_conf
            )
            emb = getattr(spatial_encoder, "embed_dim", None) or getattr(
                spatial_encoder, "embedding_dim", None
            )
            if emb is None:
                emb = spatial_encoder_conf.get("embed_dim") or spatial_encoder_conf.get(
                    "embedding_dim"
                )
            if emb is not None and "spatial_embed_dim" not in separator_conf:
                separator_conf["spatial_embed_dim"] = emb
        else:
            raise ValueError("spatial_encoder must be specified for this task.")

        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        if args.separator.endswith("nomask"):
            mask_module = mask_module_choices.get_class(args.mask_module)(
                input_dim=encoder.output_dim,
                **args.mask_module_conf,
            )
        else:
            mask_module = None

        loss_wrappers = []

        if getattr(args, "criterions", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.criterions:
                criterion_conf = ctr.get("conf", {})
                criterion = criterion_choices.get_class(ctr["name"])(**criterion_conf)
                loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                    criterion=criterion, **ctr["wrapper_conf"]
                )
                loss_wrappers.append(loss_wrapper)

        # 1. Build model
        if getattr(args, "diffusion_model", None) is not None:
            diffusion_model = diffusion_choices.get_class(args.diffusion_model)(
                **args.diffusion_model_conf
            )
            # build diffusion model
            model = ESPnetDiffusionModel(
                encoder=encoder,
                diffusion=diffusion_model,
                decoder=decoder,
                **args.model_conf,
            )

        else:
            model = ESPnetEnhancementModel(
                encoder=encoder,
                separator=separator,
                decoder=decoder,
                loss_wrappers=loss_wrappers,
                mask_module=mask_module,
                spatial_encoder=spatial_encoder,
                spatial_encoder_encoder=spatial_encoder_encoder,
                spatial_encoder_pooling=spatial_encoder_pooling,
                **args.model_conf,
            )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        if spatial_encoder is not None and spatial_encoder_pretrained:
            ckpt_path = Path(spatial_encoder_pretrained).expanduser()
            if not ckpt_path.is_absolute():
                config_path = getattr(args, "config", None)
                if config_path:
                    config_dir = Path(config_path).expanduser().resolve().parent
                    ckpt_path = (config_dir / ckpt_path).resolve()
                else:
                    ckpt_path = ckpt_path.resolve()
            if not ckpt_path.exists():
                raise ValueError(f"spatial_encoder_conf.path not found: {ckpt_path}")

            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state_dict = state["state_dict"]
            elif isinstance(state, dict):
                state_dict = state
            else:
                raise ValueError(
                    f"Unsupported checkpoint format: {type(state)} at {ckpt_path}"
                )

            if hasattr(spatial_encoder, "sc_encoder"):
                prefix = "spatial_encoder.sc_encoder."
                sc_state = {
                    k[len(prefix) :]: v
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
                if not sc_state:
                    raise ValueError(
                        f"No keys with prefix '{prefix}' found in {ckpt_path}"
                    )

                load_result = spatial_encoder.sc_encoder.load_state_dict(
                    sc_state, strict=False
                )
                missing_keys = getattr(load_result, "missing_keys", [])
                unexpected_keys = getattr(load_result, "unexpected_keys", [])
                if missing_keys or unexpected_keys:
                    raise ValueError(
                        "SC checkpoint load mismatch: "
                        f"missing={missing_keys}, unexpected={unexpected_keys}"
                    )
            elif isinstance(spatial_encoder, ResNet2DDivSpatialEncoder):
                sc_prefixes = (
                    "model.ri_stack_sc.",
                    "model.proj_sc.",
                    "model.trunk_sc.",
                    "model.pooling_sc.",
                    "model.embedding_head_sc.",
                )

                remapped_state = {}
                for key, value in state_dict.items():
                    if key.startswith("spatial_encoder.model."):
                        remapped_state[key[len("spatial_encoder.") :]] = value
                    elif key.startswith("model."):
                        remapped_state[key] = value

                sc_state = {
                    key: value
                    for key, value in remapped_state.items()
                    if key.startswith(sc_prefixes)
                }
                if not sc_state:
                    raise ValueError(
                        "No ResNet SC keys found in checkpoint. Expected keys "
                        "starting with 'spatial_encoder.model.*' or 'model.*' "
                        f"for SC branch at {ckpt_path}"
                    )

                load_result = spatial_encoder.load_state_dict(sc_state, strict=False)
                missing_keys = getattr(load_result, "missing_keys", [])
                unexpected_keys = getattr(load_result, "unexpected_keys", [])
                missing_sc = [key for key in missing_keys if key.startswith(sc_prefixes)]
                unexpected_sc = [
                    key for key in unexpected_keys if key.startswith(sc_prefixes)
                ]
                if missing_sc or unexpected_sc:
                    raise ValueError(
                        "ResNet SC checkpoint load mismatch: "
                        f"missing={missing_sc}, unexpected={unexpected_sc}"
                    )
            else:
                raise ValueError(
                    "spatial_encoder does not have sc_encoder and is not "
                    "ResNet2DDivSpatialEncoder; unsupported path loading format."
                )

            if not spatial_encoder_trainable:
                spatial_encoder.eval()
                for param in spatial_encoder.parameters():
                    param.requires_grad = False

        # Keep spatial encoder in eval() during training when frozen.
        model.spatial_encoder_frozen = not spatial_encoder_trainable

        return model

    @classmethod
    def _build_split_lr_param_groups(
        cls, args: argparse.Namespace, model: torch.nn.Module
    ) -> Optional[List[Dict]]:
        aea_block_lr = getattr(args, "aea_block_lr", None)
        non_aea_lr = getattr(args, "non_aea_lr", None)
        if aea_block_lr is None and non_aea_lr is None:
            return None

        base_lr = args.optim_conf.get("lr", None)
        if aea_block_lr is None:
            aea_block_lr = base_lr
        if non_aea_lr is None:
            non_aea_lr = base_lr
        if aea_block_lr is None or non_aea_lr is None:
            raise ValueError(
                "optim_conf.lr must be set when using aea_block_lr/non_aea_lr."
            )

        separator = getattr(model, "separator", None)
        if separator is None or not hasattr(separator, "aea_blocks"):
            raise ValueError(
                "aea_block_lr/non_aea_lr is set, but model.separator.aea_blocks "
                "is not available."
            )

        if args.exclude_weight_decay:
            add_optimizer_hooks(model, **args.exclude_weight_decay_conf)

        aea_param_ids = {
            id(param)
            for param in separator.aea_blocks.parameters()
            if param.requires_grad
        }
        if not aea_param_ids:
            raise ValueError(
                "No trainable AEA parameters found in separator.aea_blocks."
            )

        param_groups = {}
        for param in model.parameters():
            if not param.requires_grad:
                continue

            hp = dict(getattr(param, "_optim", {}))
            hp["lr"] = aea_block_lr if id(param) in aea_param_ids else non_aea_lr
            key = tuple(sorted(hp.items()))
            if key not in param_groups:
                param_groups[key] = {"params": [], **hp}
            param_groups[key]["params"].append(param)

        return list(param_groups.values())

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: torch.nn.Module,
    ) -> List[torch.optim.Optimizer]:
        if cls.num_optimizers != 1:
            raise RuntimeError(
                "build_optimizers() must be overridden if num_optimizers != 1"
            )

        optim_class = optim_classes.get(args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")

        split_groups = cls._build_split_lr_param_groups(args, model)
        if split_groups is None:
            return super().build_optimizers(args, model)

        if args.sharded_ddp:
            if fairscale is None:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim = fairscale.optim.oss.OSS(
                params=split_groups, optim=optim_class, **args.optim_conf
            )
        else:
            optim = optim_class(split_groups, **args.optim_conf)

        return [optim]

    @classmethod
    def build_iter_factory(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
        kwargs: dict = None,
    ) -> AbsIterFactory:
        dynamic_mixing = getattr(args, "dynamic_mixing", False)
        if dynamic_mixing and mode == "train":
            args = copy.deepcopy(args)
            args.fold_length = args.fold_length[0:1]

        return super().build_iter_factory(args, distributed_option, mode, kwargs)
