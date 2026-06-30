from espnet2.train.trainer import Trainer


class SepReformerTrainer(Trainer):
    """Trainer that exposes the current epoch to SepReformer loss wrappers."""

    @classmethod
    def _set_model_epoch(cls, model, epoch):
        module = getattr(model, "module", model)
        if hasattr(module, "set_epoch"):
            module.set_epoch(epoch)
        elif hasattr(model, "set_epoch"):
            model.set_epoch(epoch)

    @classmethod
    def train_one_epoch(
        cls,
        model,
        iterator,
        optimizers,
        schedulers,
        scaler,
        reporter,
        summary_writer,
        options,
        distributed_option,
    ):
        cls._set_model_epoch(model, reporter.get_epoch())
        return super().train_one_epoch(
            model=model,
            iterator=iterator,
            optimizers=optimizers,
            schedulers=schedulers,
            scaler=scaler,
            reporter=reporter,
            summary_writer=summary_writer,
            options=options,
            distributed_option=distributed_option,
        )

    @classmethod
    def validate_one_epoch(
        cls,
        model,
        iterator,
        reporter,
        options,
        distributed_option,
    ):
        cls._set_model_epoch(model, reporter.get_epoch())
        return super().validate_one_epoch(
            model=model,
            iterator=iterator,
            reporter=reporter,
            options=options,
            distributed_option=distributed_option,
        )
