#!/usr/bin/env python3
from espnet2.tasks.enh_sepreformer import SepReformerEnhancementTask


def get_parser():
    parser = SepReformerEnhancementTask.get_parser()
    return parser


def main(cmd=None):
    SepReformerEnhancementTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
