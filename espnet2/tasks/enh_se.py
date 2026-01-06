import argparse
import copy
import os
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.enh_se.espnet_model import ESPnetSpatialEncoderModel
from espnet2.enh_se.loss.criterions.abs_loss import AbsSELoss
from espnet2.enh_se.loss.criterions.contrastive_loss import PairwiseNegativeLoss
from espnet2.enh_se.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh_se.loss.wrappers.contrastive_loss_wrapper import (
    ContrastiveLossWrapper,
)
from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import AbsSpatialEncoder
from espnet2.enh_se.spatial_encoder.resnet2d_spatial_encoder import (
    ResNet2DSpatialEncoder,
)
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    EnhPreprocessor,
    SePreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

spatial_encoder_choices = ClassChoices(
    name="spatial_encoder",
    classes=dict(resnet2d=ResNet2DSpatialEncoder),
    type_check=AbsSpatialEncoder,
    default="resnet2d",
)

loss_wrapper_choices = ClassChoices(
    name="loss_wrappers",
    classes=dict(contrastive=ContrastiveLossWrapper),
    type_check=AbsLossWrapper,
    default=None,
)

criterion_choices = ClassChoices(
    name="criterions",
    classes=dict(pairwise_negative=PairwiseNegativeLoss),
    type_check=AbsSELoss,
    default=None,
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(enh=EnhPreprocessor, se=SePreprocessor),
    type_check=AbsPreprocessor,
    default=None,
)


class SpatialEncoderTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --spatial_encoder and --spatial_encoder_conf
        spatial_encoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
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
            default=get_default_kwargs(ESPnetSpatialEncoderModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "pairwise_negative",
                    "conf": {},
                    "wrapper": "contrastive",
                    "wrapper_conf": {"weight": 1.0},
                },
            ],
            help="The criterions binded with the loss wrappers.",
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
            help="The probability for applying RIR convolution.",
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

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --spatial_encoder and --spatial_encoder_conf
            class_choices.add_arguments(group)

    @classmethod
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
            if args.preprocessor == "se":
                kwargs = dict(
                    speech_volume_normalize=getattr(
                        args, "speech_volume_normalize", None
                    ),
                    sample_rate=getattr(args, "sample_rate", 8000),
                    force_single_channel=getattr(args, "force_single_channel", True),
                    categories=getattr(args, "categories", None),
                    speech_segment=getattr(args, "speech_segment", None),
                    avoid_allzero_segment=getattr(args, "avoid_allzero_segment", True),
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
            # Training mode: require all three inputs
            retval = ("speech_mix", "speech_mix_mc", "speech_mix_reverse_mc")
        else:
            # Inference mode: only need SC mixture
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["category"]
        retval = tuple(retval)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSpatialEncoderModel:

        spatial_encoder = spatial_encoder_choices.get_class(args.spatial_encoder)(
            **args.spatial_encoder_conf
        )

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

        # Build model
        model = ESPnetSpatialEncoderModel(
            spatial_encoder=spatial_encoder,
            loss_wrappers=loss_wrappers,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model

    @classmethod
    def build_iter_factory(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
        kwargs: dict = None,
    ) -> AbsIterFactory:
        return super().build_iter_factory(args, distributed_option, mode, kwargs)
