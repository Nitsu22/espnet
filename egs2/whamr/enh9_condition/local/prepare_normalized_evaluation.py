"""Reuse trained Oracle weights in a separate evaluation output directory."""
import argparse
import json
from pathlib import Path

from check_oracle_ctf_readiness import digest, verify_readiness


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.parent != source or output == source:
        raise ValueError("Evaluation output must be a new child of the trained experiment")
    config = source / "config.yaml"
    model = source / "valid.loss.best.pth"
    original = json.loads((source / ".pipeline/inference.done.json").read_text())
    expected = dict(source=str(source), model_file=str(model.resolve()),
                    model_sha256=digest(model), config_sha256=digest(config),
                    source_run_sha256=digest(source / ".pipeline/run.json"),
                    normalize_output_wav=True)
    if original["model_sha256"] != expected["model_sha256"]:
        raise RuntimeError("Trained model differs from the original evaluation")
    marker = output / "evaluation_source.json"
    if output.exists():
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise RuntimeError("Evaluation destination already exists with different provenance")
    else:
        output.mkdir()
        marker.write_text(json.dumps(expected, indent=2) + "\n")
    for name in ("config.yaml", "checkpoint.pth", "train.log", "valid.loss.best.pth"):
        target = (source / name).resolve(strict=True)
        link = output / name
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError("Unexpected evaluation input: " + str(link))
        link.symlink_to(target)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_readiness()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
