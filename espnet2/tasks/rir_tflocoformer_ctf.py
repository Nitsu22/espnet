import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.rir.rec_rir.tflocoformer_ctf_pit import (
    ESPnetRecRIRTFLocoformerPITModel,
)
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import AbsPreprocessor
from espnet2.train.preprocessor_rec_rir_pit import RecRIRPITPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none


preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(rec_rir_pit=RecRIRPITPreprocessor),
    type_check=AbsPreprocessor,
    default="rec_rir_pit",
)


class RIRTFLocoformerCTFTask(AbsTask):
    num_optimizers: int = 1
    class_choices_list = [preprocessor_choices]
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
            default=get_default_kwargs(ESPnetRecRIRTFLocoformerPITModel),
            help="The keyword arguments for the TF-Locoformer CTF model.",
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
        return ("speech_mix",)

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return (
            "speech_direct1",
            "speech_direct2",
            "speech_reverb1",
            "speech_reverb2",
            "rir_ref1",
            "rir_ref2",
            "room_param_path",
            "t60",
            "category",
            "fs",
        )

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        model = ESPnetRecRIRTFLocoformerPITModel(**args.model_conf)
        if args.init is not None:
            initialize(model, args.init)
        return model
