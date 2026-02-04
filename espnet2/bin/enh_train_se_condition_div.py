#!/usr/bin/env python3
from espnet2.tasks.enh_se_condition_div import EnhancementTask


def get_parser():
    parser = EnhancementTask.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancement training for SE-condition-div."""
    EnhancementTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
