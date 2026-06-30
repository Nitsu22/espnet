import argparse

from typeguard import typechecked

from espnet2.enh.diffusion_enh import ESPnetDiffusionModel
from espnet2.enh.sepreformer_espnet_model import SepReformerESPnetEnhancementModel
from espnet2.tasks.enh import (
    EnhancementTask,
    decoder_choices,
    diffusion_choices,
    encoder_choices,
    loss_wrapper_choices,
    mask_module_choices,
    separator_choices,
)
from espnet2.tasks.enh import criterion_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.sepreformer_trainer import SepReformerTrainer


class SepReformerEnhancementTask(EnhancementTask):
    """Enhancement task using the SepReformer epoch-aware trainer/model."""

    trainer = SepReformerTrainer

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
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
            for ctr in args.criterions:
                criterion_conf = ctr.get("conf", {})
                criterion = criterion_choices.get_class(ctr["name"])(**criterion_conf)
                loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                    criterion=criterion,
                    **ctr["wrapper_conf"],
                )
                loss_wrappers.append(loss_wrapper)

        if getattr(args, "diffusion_model", None) is not None:
            diffusion_model = diffusion_choices.get_class(args.diffusion_model)(
                **args.diffusion_model_conf,
            )
            model = ESPnetDiffusionModel(
                encoder=encoder,
                diffusion=diffusion_model,
                decoder=decoder,
                **args.model_conf,
            )
        else:
            model = SepReformerESPnetEnhancementModel(
                encoder=encoder,
                separator=separator,
                decoder=decoder,
                loss_wrappers=loss_wrappers,
                mask_module=mask_module,
                **args.model_conf,
            )

        if args.init is not None:
            initialize(model, args.init)

        return model
