"""Require a complete, unmodified Oracle dump before training or scoring."""
import argparse
import json
from pathlib import Path

from prepare_oracle_ctf_dump import validate_prepared_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, required=True)
    args = parser.parse_args()
    report = validate_prepared_dump(args.dump, require_full=True)
    print(json.dumps({'status': report['status'], 'counts': report['counts']}, indent=2))


if __name__ == '__main__':
    main()
