#!/usr/bin/env python3
from espnet2.tasks.enh_log import EnhancementTaskLog


def get_parser():
    parser = EnhancementTaskLog.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancemnet frontend training with extra logging."""
    EnhancementTaskLog.main(cmd=cmd)


if __name__ == "__main__":
    main()
