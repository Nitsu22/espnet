#!/usr/bin/env python3
from espnet2.tasks.enh_log_fwd import EnhancementTaskFwdLog


def get_parser():
    parser = EnhancementTaskFwdLog.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancement training with forward-internal logging."""
    EnhancementTaskFwdLog.main(cmd=cmd)


if __name__ == "__main__":
    main()
