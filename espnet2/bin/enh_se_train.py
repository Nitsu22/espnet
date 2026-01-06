#!/usr/bin/env python3
from espnet2.tasks.enh_se import SpatialEncoderTask


def get_parser():
    parser = SpatialEncoderTask.get_parser()
    return parser


def main(cmd=None):
    r"""Spatial Encoder training.

    Example:

        % python enh_se_train.py enh_se --print_config --optim adadelta \
                > conf/train_enh_se.yaml
        % python enh_se_train.py --config conf/train_enh_se.yaml
    """
    SpatialEncoderTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
