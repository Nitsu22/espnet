#!/usr/bin/env python3
from espnet2.tasks.enh_se_npz import SpatialEncoderTask


def get_parser():
    parser = SpatialEncoderTask.get_parser()
    return parser


def main(cmd=None):
    r"""Spatial Encoder training for NPZ-based contrastive learning."""
    SpatialEncoderTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
