"""Pin paired test-evaluation inputs/checkpoints and validate complete outputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import soundfile as sf


METRICS = ("SI_SNR", "SDR", "SAR", "SIR", "STOI", "ESTOI")
MODELS = ("baseline", "conditioned")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_index(path):
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in values:
            raise ValueError(f"Malformed/duplicate index entry: {path}:{number}")
        values[fields[0]] = fields[1]
    if not values:
        raise ValueError(f"Empty index: {path}")
    return values


def inspect_audio(index, expected=None):
    lengths = {}
    for uid, value in index.items():
        # These test dumps contain plain WAV paths, never shell pipelines.
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Expected existing absolute WAV path: {uid}: {value}")
        info = sf.info(path)
        if info.samplerate != 8000 or info.channels != 1 or info.frames <= 128:
            raise ValueError(f"Expected nonempty mono 8-kHz WAV: {uid}: {path}")
        if expected is not None and info.frames != expected[uid]:
            raise ValueError(f"Audio length mismatch: {uid}: {path}")
        lengths[uid] = info.frames
    return lengths


def input_spec(test_dir):
    names = ("wav.scp", "spk1.scp", "spk2.scp")
    indexes = [read_index(test_dir / name) for name in names]
    if any(set(index) != set(indexes[0]) for index in indexes[1:]):
        raise ValueError("Mixture/reference test key sets differ")
    lengths = inspect_audio(indexes[0])
    for index in indexes[1:]:
        inspect_audio(index, lengths)
    spec = {"test_dir": str(test_dir), "count": len(lengths),
            "indexes": {name: digest(test_dir / name) for name in names},
            "lengths": lengths}
    manifest = test_dir.parent.parent / "preparation.json"
    if manifest.is_file():
        spec["generation_manifest_sha256"] = digest(manifest)
    return spec


def verify_generation_transfer(test_dir):
    """Once per fresh evaluation, verify all transferred WAVs and audit files."""
    root = test_dir.parent.parent
    marker = root / "preparation.json"
    if not marker.is_file():
        return  # The generic runner also supports independent plain test dumps.
    manifest = json.loads(marker.read_text())
    if "output_hashes" not in manifest:
        raise ValueError("Generated dump is missing its transfer checksums")
    for relative, expected in manifest["output_hashes"].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Transferred generation file changed: {relative}")


def verify(root):
    # Resume can rewrite incomplete inference/score files only in this dedicated
    # output tree. Refuse symlinked destinations, including individual WAVs.
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Evaluation output contains symlinked write destinations")
    spec = json.loads((root / "provenance.json").read_text())
    if input_spec(Path(spec["inputs"]["test_dir"])) != spec["inputs"]:
        raise ValueError("Test inputs changed since this evaluation was prepared")
    for model in spec["models"].values():
        for kind in ("config", "checkpoint"):
            if digest(model[kind]) != model[f"{kind}_sha256"]:
                raise ValueError(f"Pinned {kind} changed: {model[kind]}")
    return spec


def prepare(args):
    root = args.output_dir.resolve()
    test = args.test_dir.resolve(strict=True)
    exps = {name: getattr(args, f"{name}_exp").resolve(strict=True)
            for name in MODELS}
    normalize_output_wav = args.normalize_output_wav == "true"
    allowed = (Path("/net/midgar/work2/nitsu"), Path("/gs/bs/tga-shinoda/nitsu"))
    if not any(base in root.parents for base in allowed):
        raise ValueError("Evaluation output must be inside a permitted user directory")
    if any(root == source or source in root.parents or root in source.parents
           for source in (test, *exps.values())):
        raise ValueError("Evaluation output must be separate from test/training directories")
    if root.exists():
        if not args.resume:
            raise ValueError("Evaluation output exists; use a new directory or --resume true")
        spec = verify(root)
        if spec["inference"]["normalize_output_wav"] != normalize_output_wav:
            raise ValueError("Output normalization differs from the recorded evaluation")
        if spec["inputs"]["test_dir"] != str(test) or any(
                spec["models"][name]["experiment"] != str(exp)
                for name, exp in exps.items()):
            raise ValueError("Resume arguments do not match the recorded evaluation")
        # Keep the resolved epoch checkpoint, even if valid.loss.best.pth moved.
        return
    verify_generation_transfer(test)
    inputs = input_spec(test)
    models = {}
    for name, exp in exps.items():
        config = (exp / "config.yaml").resolve(strict=True)
        checkpoint = (exp / args.checkpoint).resolve(strict=True)
        models[name] = {"experiment": str(exp), "config": str(config),
                        "config_sha256": digest(config), "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest(checkpoint),
                        "requested_checkpoint": args.checkpoint}
    root.mkdir(parents=True)
    spec = {"schema_version": 1, "inputs": inputs, "models": models,
            "inference": {"fs": 8000, "batch_size": 1, "ngpu": 1,
                          "segmented": False, "normalize_output_wav": normalize_output_wav},
            "conditioning": "Frozen BiMamba estimates CTF from each new mixture; no cached CTF",
            "scoring": "ESPnet enh_scoring, ref_channel=0, standard BSS_EVAL permutation"}
    (root / "provenance.json").write_text(json.dumps(spec, indent=2) + "\n")


def check_inference(root, model):
    spec = verify(root)
    expected = spec["inputs"]["lengths"]
    for speaker in (1, 2):
        values = read_index(root / model / f"spk{speaker}.scp")
        if set(values) != set(expected):
            raise ValueError(f"Incomplete inference keys: {model}, speaker {speaker}")
        inspect_audio(values, expected)


def scores(root, model):
    spec = verify(root)
    expected = set(spec["inputs"]["lengths"])
    values = {}
    for metric in METRICS:
        for speaker in (1, 2):
            index = read_index(root / model / "score" / f"{metric}_spk{speaker}")
            if set(index) != expected:
                raise ValueError(f"Incomplete score keys: {model}/{metric}/spk{speaker}")
            converted = {uid: float(value) for uid, value in index.items()}
            if not all(math.isfinite(value) for value in converted.values()):
                raise ValueError(f"Nonfinite scores: {model}/{metric}/spk{speaker}")
            values[metric, speaker] = converted
    return values


def summary(root):
    by_model = {model: scores(root, model) for model in MODELS}
    rows = []
    for metric in METRICS:
        row = {"metric": metric}
        for model, values in by_model.items():
            numbers = [value for speaker in (1, 2)
                       for value in values[metric, speaker].values()]
            row[model] = math.fsum(numbers) / len(numbers)
        row["gain"] = row["conditioned"] - row["baseline"]
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "baseline", "conditioned", "gain"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("test-dir", "baseline-exp", "conditioned-exp", "output-dir"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--checkpoint", default="valid.loss.best.pth")
    prep.add_argument("--normalize-output-wav", choices=("true", "false"), default="true")
    prep.add_argument("--resume", action="store_true")
    for command in ("paths", "verify", "check-inference", "check-score", "summary"):
        item = sub.add_parser(command)
        item.add_argument("output_dir", type=Path)
        if command in ("paths", "check-inference", "check-score"):
            item.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "paths":
        spec = verify(args.output_dir)["models"][args.model]
        print(spec["config"])
        print(spec["checkpoint"])
    elif args.command == "verify":
        verify(args.output_dir)
    elif args.command == "check-inference":
        check_inference(args.output_dir, args.model)
    elif args.command == "check-score":
        scores(args.output_dir, args.model)
    elif args.command == "summary":
        summary(args.output_dir)


if __name__ == "__main__":
    main()
