"""Train DAMSEP using ESPnet's trainer, checkpointing and distributed support."""

from espnet2.enh.damsep.espnet_model import ESPnetDAMSEPModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor_rec_rir_pit import RecRIRPITPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction


class DAMSEPTask(AbsTask):
    num_optimizers = 1
    trainer = Trainer
    class_choices_list = []

    @classmethod
    def add_task_arguments(cls, parser):
        parser.add_argument("--model_conf", action=NestedDictAction, default={})
        parser.add_argument("--preprocessor_conf", action=NestedDictAction, default={})

    @classmethod
    def build_model(cls, args):
        return ESPnetDAMSEPModel(**args.model_conf)

    @classmethod
    def build_collate_fn(cls, args, train):
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(cls, args, train):
        return RecRIRPITPreprocessor(
            train=train, speech_direct_prefix="speech_ref", **args.preprocessor_conf
        )

    @classmethod
    def required_data_names(cls, train=True, inference=False):
        if inference:
            return ("speech_mix",)
        return (
            "speech_mix",
            "speech_ref1",
            "speech_ref2",
            "speech_reverb1",
            "speech_reverb2",
        )

    @classmethod
    def optional_data_names(cls, train=True, inference=False):
        return ()
