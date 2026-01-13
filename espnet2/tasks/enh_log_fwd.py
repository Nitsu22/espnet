from espnet2.tasks.abs_task_log import apply_log_patches
from espnet2.tasks.enh import EnhancementTask
from espnet2.torch_utils.forward_log import attach_forward_logging
from espnet2.train.trainer_log import TrainerLog


class EnhancementTaskFwdLog(EnhancementTask):
    trainer = TrainerLog

    @classmethod
    def build_model(cls, args):
        model = super().build_model(args)
        attach_forward_logging(model)
        return model

    @classmethod
    def main_worker(cls, args):
        apply_log_patches()
        return super().main_worker(args)
