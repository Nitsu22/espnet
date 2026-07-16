#!/usr/bin/env python3
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask


def get_parser():
    parser = EnhancementRIRCTFTask.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancement frontend training with precomputed Rec-RIR CTF."""
    EnhancementRIRCTFTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
