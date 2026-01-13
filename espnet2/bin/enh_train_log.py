#!/usr/bin/env python3
from espnet2.tasks.abs_task_log import apply_log_patches
from espnet2.tasks.enh import EnhancementTask
from espnet2.train.trainer_log import TrainerLog


def _apply_patches() -> None:
    apply_log_patches()
    EnhancementTask.trainer = TrainerLog


def get_parser():
    _apply_patches()
    parser = EnhancementTask.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancemnet frontend training with extra logging."""
    _apply_patches()
    EnhancementTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
