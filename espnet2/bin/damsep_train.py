#!/usr/bin/env python3
from espnet2.tasks.damsep import DAMSEPTask


def get_parser():
    return DAMSEPTask.get_parser()


def main(cmd=None):
    DAMSEPTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
