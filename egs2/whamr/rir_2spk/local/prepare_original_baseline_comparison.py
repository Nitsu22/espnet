"""Package the trained baseline's original audio bytes, indices, and shapes.

This never resamples or rewrites audio.  The package SCPs point to --destination,
so the complete output directory can be transported there without rewriting it.
Audio hardlinks, when possible, are immutable aliases of current-user-owned
sources; do not edit packaged WAVs in place.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil

import soundfile as sf
import yaml


NAMES = (
    "speech_mix", "speech_direct1", "speech_direct2", "speech_reverb1", "speech_reverb2"
)
ALLOWED_ROOTS = (
    Path("/net/midgar/work2/nitsu"), Path("/gs/bs/tga-shinoda/nitsu")
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(value, recipe):
    path = Path(value)
    return (path if path.is_absolute() else recipe / path).resolve(strict=True)


def writable_output(value):
    path = value.resolve()
    if not any(path.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"Output must be inside the user's storage: {path}")
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_uid != os.getuid():
        raise ValueError(f"Output parent is not owned by the current user: {ancestor}")
    if path.exists():
        raise FileExistsError(f"Use a fresh package directory: {path}")
    return path


def read_scp(path):
    rows = []
    keys = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        pair = line.split(maxsplit=1)
        if len(pair) != 2 or pair[0] in keys:
            raise ValueError(f"Invalid/duplicate SCP row: {path}:{line_number}")
        uid, value = pair
        if uid in (".", "..") or "/" in uid or "\\" in uid:
            raise ValueError(f"Unsafe utterance key in {path}: {uid}")
        if value.endswith("|"):
            raise ValueError(f"Expected a real audio file, got a pipeline: {value}")
        rows.append((uid, value))
        keys.add(uid)
    if not rows:
        raise ValueError(f"Empty SCP: {path}")
    return rows


def saved_command(log):
    with log.open() as stream:
        first_line = stream.readline().strip()
    if not first_line.startswith("# "):
        raise ValueError(f"Missing original launch command in {log}")
    command = first_line[2:]
    options = defaultdict(list)
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token.startswith("--") and index + 1 < len(tokens):
            options[token[2:]].append(tokens[index + 1])
    return command, options


def authoritative_specs(config, command_options, split):
    key = f"{split}_data_path_and_name_and_type"
    specs = config[key]
    saved = [",".join(spec) for spec in specs]
    if command_options.get(key) != saved:
        raise ValueError(f"Saved config and actual launch command disagree: {key}")
    if [spec[1] for spec in specs] != list(NAMES):
        raise ValueError(f"Expected the original five speech inputs: {key}")
    if any(spec[2] != "sound" for spec in specs):
        raise ValueError(f"Expected sound inputs: {key}")
    shape_key = f"{split}_shape_file"
    if command_options.get(shape_key) != config[shape_key]:
        raise ValueError(f"Saved config and actual launch shapes disagree: {shape_key}")
    if len(config[shape_key]) != len(NAMES):
        raise ValueError(f"Expected all five original shape files: {shape_key}")
    return specs


def copy_audio(source, target):
    """Return source and staged SHA256, preserving the complete WAV file."""
    if not source.is_file() or source.suffix.lower() != ".wav":
        raise ValueError(f"Expected an existing WAV file: {source}")
    info = sf.info(str(source))
    if info.format not in ("WAV", "WAVEX") or info.samplerate != 8000:
        raise ValueError(f"Expected the original native 8 kHz WAV: {source}: {info}")
    target.parent.mkdir(parents=True, exist_ok=True)
    linked = False
    if source.stat().st_uid == os.getuid():
        try:
            os.link(source, target)
            linked = True
        except OSError:
            pass
    if not linked:
        shutil.copyfile(source, target)
    original_hash = sha256(source)
    staged_hash = original_hash if linked else sha256(target)
    if staged_hash != original_hash:
        raise ValueError(f"Copied audio checksum mismatch: {source}")
    return dict(
        sha256=original_hash,
        staged_sha256=staged_hash,
        bytes=source.stat().st_size,
        frames=info.frames,
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype,
        hardlinked=linked,
    )


def prepare(args):
    baseline = args.baseline_exp.resolve(strict=True)
    recipe = baseline.parent.parent
    config_path = baseline / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["rir_model_type"] != "rec_rir_pit":
        raise ValueError("Expected the original rec_rir_pit baseline")
    command, options = saved_command(baseline / "train.log")
    if options.get("fold_length") != [str(v) for v in config["fold_length"]]:
        raise ValueError("Saved config and actual launch fold lengths disagree")
    if config["sample_rate"] != 8000 or config["model_conf"]["sr"] != 8000:
        raise ValueError("Expected the original 8 kHz experiment")
    output = writable_output(args.output)
    destination = args.destination
    if not destination.is_absolute() or not any(
        destination.is_relative_to(root) for root in ALLOWED_ROOTS
    ):
        raise ValueError("Destination must be an absolute path inside the user's storage")

    specs = {split: authoritative_specs(config, options, split)
             for split in ("train", "valid")}
    # Test was not part of training. Include its exact indices only when all five
    # corresponding original test indices are present.
    test_specs = []
    for index, name, kind in specs["train"]:
        original = source_path(index, recipe)
        if not original.parent.name.startswith("tr_"):
            break
        test_index = original.parent.with_name("tt_" + original.parent.name[3:]) / original.name
        if not test_index.is_file():
            break
        test_specs.append([str(test_index), name, kind])
    if len(test_specs) == len(NAMES):
        specs["test"] = test_specs

    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copyfile(config_path, output / "baseline_config.yaml")
    (provenance / "original_train_command.txt").write_text(command + "\n")
    report = dict(
        schema="rec_rir_original_baseline_comparison_v1",
        baseline_exp=str(baseline),
        original_recipe=str(recipe),
        baseline_config_sha256=sha256(config_path),
        config_sha256=sha256(config_path),
        destination=str(destination),
        no_audio_conversion=True,
        source_scp_sha256={},
        destination_scp_sha256={},
        index_sha256={},
        source_shape_sha256={},
        destination_shape_sha256={},
        shape_sha256={},
        counts={},
        data_specs={},
        shape_files={},
        checkpoints={},
    )
    for checkpoint_name in ("valid.loss.best.pth", "valid.loss.ave_5best.pth"):
        path = baseline / checkpoint_name
        if path.is_file():
            report["checkpoints"][checkpoint_name] = dict(
                path=str(path), resolved_path=str(path.resolve()), sha256=sha256(path)
            )

    audio_manifest = output / "audio_manifest.jsonl"
    with audio_manifest.open("w") as manifest:
        for split, input_specs in specs.items():
            rows_by_name = {}
            report["data_specs"][split] = []
            for index, name, kind in input_specs:
                index_path = source_path(index, recipe)
                rows = read_scp(index_path)
                rows_by_name[name] = rows
                index_relative = Path("raw") / index_path.parent.name / index_path.name
                staged_index = output / index_relative
                staged_index.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for uid, source_value in rows:
                    source = source_path(source_value, recipe)
                    audio_relative = Path("audio") / split / name / f"{uid}.wav"
                    metadata = copy_audio(source, output / audio_relative)
                    manifest.write(json.dumps(dict(
                        split=split, name=name, uid=uid, source=str(source),
                        package_path=str(audio_relative),
                        relative_path=str(audio_relative),
                        destination=str(destination / audio_relative), **metadata
                    ), ensure_ascii=False) + "\n")
                    lines.append(f"{uid} {destination / audio_relative}\n")
                staged_index.write_text("".join(lines))
                report["source_scp_sha256"][str(index_path)] = sha256(index_path)
                report["destination_scp_sha256"][str(index_relative)] = sha256(staged_index)
                report["index_sha256"][str(index_relative)] = sha256(staged_index)
                report["data_specs"][split].append(
                    [str(destination / index_relative), name, kind]
                )
            expected = {uid for uid, _ in rows_by_name["speech_mix"]}
            if any({uid for uid, _ in rows} != expected for rows in rows_by_name.values()):
                raise ValueError(f"Original speech input IDs disagree: {split}")
            report["counts"][split] = len(expected)

            if split == "test":
                continue
            report["shape_files"][split] = []
            for original_value in config[f"{split}_shape_file"]:
                if args.source_stats_dir is None:
                    original_shape = source_path(original_value, recipe)
                else:
                    original_shape = (
                        args.source_stats_dir / split / Path(original_value).name
                    ).resolve(strict=True)
                shape_keys = [line.split(maxsplit=1)[0]
                              for line in original_shape.read_text().splitlines()]
                if len(shape_keys) != len(expected) or set(shape_keys) != expected:
                    raise ValueError(f"Shape IDs disagree with original audio: {original_shape}")
                relative_shape = Path("stats") / split / original_shape.name
                target_shape = output / relative_shape
                target_shape.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_shape, target_shape)
                original_hash = sha256(original_shape)
                staged_hash = sha256(target_shape)
                if original_hash != staged_hash:
                    raise ValueError(f"Shape checksum mismatch: {original_shape}")
                report["source_shape_sha256"][str(original_shape)] = original_hash
                report["destination_shape_sha256"][str(relative_shape)] = staged_hash
                report["shape_sha256"][str(relative_shape)] = staged_hash
                report["shape_files"][split].append(str(destination / relative_shape))
            print(f"{split}: {len(expected)} original utterances packaged", flush=True)

    report["audio_manifest_sha256"] = sha256(audio_manifest)
    report["audio_manifest"] = audio_manifest.name
    # The completion manifest is written last. An interrupted package never
    # acquires this success marker and cannot be mistaken for prepared data.
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(dict(output=str(output), counts=report["counts"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exp", type=Path, required=True)
    parser.add_argument("--source-stats-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
