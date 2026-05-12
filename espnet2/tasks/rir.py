import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.criterions.time_domain import (
    MultiResL1SpecLoss,
    TimeDomainL1,
    TimeDomainMSE,
)
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.fla_tflocoformer_separator import (
    TFLocoformerSeparator as FLATFLocoformerSeparator,
)
from espnet2.enh.separator.tflocoformer_separator import TFLocoformerSeparator
from espnet2.rir.espnet_model import ESPnetRIRModel
from espnet2.rir.loss.criterions.time_domain import (
    RIRCorrelationLoss,
    RIRMultiTaskLoss,
)
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import AbsPreprocessor
from espnet2.train.preprocessor_rir import RIRPreprocessor
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
        tflocoformer=TFLocoformerSeparator,
        fla_tflocoformer=FLATFLocoformerSeparator,
    ),
    type_check=AbsSeparator,
    default="tflocoformer",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(stft=STFTDecoder, conv=ConvDecoder, same=NullDecoder),
    type_check=AbsDecoder,
    default="stft",
)

loss_wrapper_choices = ClassChoices(
    name="loss_wrapper",
    classes=dict(fixed_order=FixedOrderSolver, pit=PITSolver),
    type_check=AbsLossWrapper,
    default="fixed_order",
)

criterion_choices = ClassChoices(
    name="criterion",
    classes=dict(
        l1=TimeDomainL1,
        mse=TimeDomainMSE,
        mrstft=MultiResL1SpecLoss,
        corr=RIRCorrelationLoss,
        rir_multitask=RIRMultiTaskLoss,
    ),
    type_check=AbsEnhLoss,
    default="mrstft",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(rir=RIRPreprocessor),
    type_check=AbsPreprocessor,
    default="rir",
)


class RIRTask(AbsTask):
    num_optimizers: int = 1

    class_choices_list = [
        encoder_choices,
        separator_choices,
        decoder_choices,
        preprocessor_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
            help="The initialization method",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetRIRModel),
            help="The keyword arguments for model class.",
        )
        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "mrstft",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {"weight": 1.0},
                }
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--sample_rate",
            type=int,
            default=8000,
            help="Sampling rate of the input speech data in Hz.",
        )
        group.add_argument(
            "--force_single_channel",
            type=str2bool,
            default=True,
            help="Whether to force speech input to a single channel.",
        )
        group.add_argument(
            "--speech_segment",
            type=int_or_none,
            default=None,
            help="Truncate input speech to this number of samples if not None.",
        )
        group.add_argument(
            "--avoid_allzero_segment",
            type=str2bool,
            default=True,
            help="Avoid all-zero speech segments when speech_segment is set.",
        )

        for class_choices in cls.class_choices_list:
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
    ) -> Optional[Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]]:
        if getattr(args, "preprocessor", None) is None:
            return None

        kwargs = dict(
            speech_sample_rate=getattr(args, "sample_rate", 8000),
            force_single_channel=getattr(args, "force_single_channel", True),
            speech_segment=getattr(args, "speech_segment", None),
            avoid_allzero_segment=getattr(args, "avoid_allzero_segment", True),
        )
        kwargs.update(args.preprocessor_conf)
        return preprocessor_choices.get_class(args.preprocessor)(
            train=train,
            **kwargs,
        )

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if inference:
            return ("speech_mix",)
        return ("speech_mix", "rir_path", "room_param_path")

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ("category", "fs")

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetRIRModel:
        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        loss_wrappers = []
        for ctr in args.criterions:
            criterion_conf = ctr.get("conf", {})
            criterion = criterion_choices.get_class(ctr["name"])(**criterion_conf)
            loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                criterion=criterion,
                **ctr["wrapper_conf"],
            )
            loss_wrappers.append(loss_wrapper)

        model = ESPnetRIRModel(
            encoder=encoder,
            separator=separator,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            **args.model_conf,
        )

        if args.init is not None:
            initialize(model, args.init)

        return model
