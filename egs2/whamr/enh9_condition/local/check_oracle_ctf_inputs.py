"""Require a complete, unmodified Oracle dump before training or scoring."""
import argparse
import json
from pathlib import Path

from oracle_condition import validate_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--condition', choices=('nf_whamr', 'whamr'), default='nf_whamr')
    args = parser.parse_args()
    report = validate_dump(args.dump, args.condition, require_full=True)
    print(json.dumps({'status': report['status'], 'counts': report['counts']}, indent=2))


if __name__ == '__main__':
    main()
