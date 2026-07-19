#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ScpEntry = Tuple[str, str]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic, within-split derangement of an RIR CTF SCP. "
            "Output keys remain aligned with wav.scp while CTF paths come from "
            "different utterances."
        )
    )
    parser.add_argument("--wav-scp", type=Path, required=True)
    parser.add_argument("--ctf-scp", type=Path, required=True)
    parser.add_argument("--output-scp", type=Path, required=True)
    parser.add_argument("--mapping-tsv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def read_scp(path: Path, label: str) -> List[ScpEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")

    entries: List[ScpEntry] = []
    first_line_for_uid: Dict[str, int] = {}
    with path.open(encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"{label}:{line_number}: empty lines are not allowed")
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"{label}:{line_number}: expected '<uid> <value>', got: {line!r}"
                )
            uid, value = fields
            if uid in first_line_for_uid:
                raise ValueError(
                    f"{label}:{line_number}: duplicate uid {uid!r}; "
                    f"first seen on line {first_line_for_uid[uid]}"
                )
            first_line_for_uid[uid] = line_number
            entries.append((uid, value))
    return entries


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sattolo_derangement(size: int, seed: int) -> List[int]:
    if size < 2:
        raise ValueError(
            f"At least two utterances are required for a derangement, got {size}"
        )

    permutation = list(range(size))
    rng = random.Random(seed)
    for index in range(size - 1, 0, -1):
        swap_index = rng.randrange(index)
        permutation[index], permutation[swap_index] = (
            permutation[swap_index],
            permutation[index],
        )

    if sorted(permutation) != list(range(size)):
        raise RuntimeError("Internal error: Sattolo output is not a permutation")
    if any(target == donor for target, donor in enumerate(permutation)):
        raise RuntimeError("Internal error: Sattolo output contains a fixed point")
    return permutation


def validate_inputs(
    wav_entries: Sequence[ScpEntry], ctf_entries: Sequence[ScpEntry]
) -> Tuple[List[str], Dict[str, str]]:
    wav_keys = [uid for uid, _ in wav_entries]
    ctf_by_uid = dict(ctf_entries)
    wav_key_set = set(wav_keys)
    ctf_key_set = set(ctf_by_uid)

    if wav_key_set != ctf_key_set:
        missing_ctf = sorted(wav_key_set - ctf_key_set)
        extra_ctf = sorted(ctf_key_set - wav_key_set)
        raise ValueError(
            "wav.scp and rir_ctf.scp key sets differ: "
            f"missing_ctf={missing_ctf[:5]} (total={len(missing_ctf)}), "
            f"extra_ctf={extra_ctf[:5]} (total={len(extra_ctf)})"
        )
    if len(wav_keys) < 2:
        raise ValueError(
            "At least two utterances are required for a derangement, "
            f"got {len(wav_keys)}"
        )

    missing_paths = []
    for uid in wav_keys:
        # Match NpyScpReader semantics: relative paths are resolved from cwd,
        # while shell-only shortcuts such as "~" are not expanded.
        ctf_path = Path(ctf_by_uid[uid])
        resolved_path = ctf_path if ctf_path.is_absolute() else Path.cwd() / ctf_path
        if not resolved_path.is_file():
            missing_paths.append((uid, str(ctf_path)))
    if missing_paths:
        examples = ", ".join(f"{uid}={path}" for uid, path in missing_paths[:5])
        raise FileNotFoundError(
            f"CTF paths do not exist: {examples} (total={len(missing_paths)})"
        )

    return wav_keys, ctf_by_uid


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(cmd: Optional[Sequence[str]] = None) -> None:
    args = get_parser().parse_args(cmd)
    wav_entries = read_scp(args.wav_scp, "wav.scp")
    ctf_entries = read_scp(args.ctf_scp, "rir_ctf.scp")
    target_uids, ctf_by_uid = validate_inputs(wav_entries, ctf_entries)

    donor_indices = sattolo_derangement(len(target_uids), args.seed)
    assignments = [
        (target_uid, target_uids[donor_index], ctf_by_uid[target_uids[donor_index]])
        for target_uid, donor_index in zip(target_uids, donor_indices)
    ]

    if any(target_uid == donor_uid for target_uid, donor_uid, _ in assignments):
        raise RuntimeError(
            "Internal error: generated assignment contains a fixed point"
        )
    if {donor_uid for _, donor_uid, _ in assignments} != set(target_uids):
        raise RuntimeError("Internal error: donor assignment is not one-to-one")

    output_scp = "".join(
        f"{target_uid} {donor_path}\n"
        for target_uid, _, donor_path in assignments
    )
    mapping_tsv = "target_uid\tdonor_uid\tdonor_ctf_path\n" + "".join(
        f"{target_uid}\t{donor_uid}\t{donor_path}\n"
        for target_uid, donor_uid, donor_path in assignments
    )
    manifest = {
        "algorithm": "sattolo_derangement_v1",
        "seed": args.seed,
        "num_entries": len(assignments),
        "wav_scp": str(args.wav_scp.resolve()),
        "wav_scp_sha256": sha256sum(args.wav_scp),
        "source_ctf_scp": str(args.ctf_scp.resolve()),
        "source_ctf_scp_sha256": sha256sum(args.ctf_scp),
        "output_scp": str(args.output_scp.resolve()),
        "mapping_tsv": str(args.mapping_tsv.resolve()),
    }

    atomic_write_text(args.output_scp, output_scp)
    atomic_write_text(args.mapping_tsv, mapping_tsv)
    atomic_write_text(
        args.manifest_json,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Wrote {len(assignments)} fixed-point-free CTF assignments "
        f"with seed={args.seed} to {args.output_scp}"
    )


if __name__ == "__main__":
    main()
