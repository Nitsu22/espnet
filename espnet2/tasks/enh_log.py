from espnet2.tasks.abs_task_log import apply_log_patches
from espnet2.tasks.enh import EnhancementTask
from espnet2.train.trainer_log import TrainerLog


class EnhancementTaskLog(EnhancementTask):
    trainer = TrainerLog

    @classmethod
    def main_worker(cls, args):
        apply_log_patches()
        return super().main_worker(args)
