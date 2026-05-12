#!/usr/bin/env python3
from espnet2.tasks.rir import RIRTask


def get_parser():
    return RIRTask.get_parser()


def main(cmd=None):
    RIRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
