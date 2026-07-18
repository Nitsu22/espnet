#!/usr/bin/env python3
from espnet2.tasks.rir_tflocoformer_ctf import RIRTFLocoformerCTFTask


def get_parser():
    return RIRTFLocoformerCTFTask.get_parser()


def main(cmd=None):
    RIRTFLocoformerCTFTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
