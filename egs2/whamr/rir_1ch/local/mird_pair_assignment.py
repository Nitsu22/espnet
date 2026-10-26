"""Reproducible two-source assignments for all 72 MIRD receiver conditions.

Receiver IDs refer to recordings, not independent rooms or physical positions.
MIRD uses one room, three reverberation settings and overlapping array layouts.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations, product
from numbers import Integral
from pathlib import Path

import numpy as np


RT60_MS = (160, 360, 610)
SPACINGS = ("3-3-3-8-3-3-3", "4-4-4-8-4-4-4", "8-8-8-8-8-8-8")
MICROPHONES = tuple(range(1, 9))
SOURCE_COUNT = 26
PAIR_TYPES = tuple(combinations(range(SOURCE_COUNT), 2))
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_TYPES)}


def _receiver_quotas(count, rng):
    """Balance RT, array, and microphone counts, including rounding remainders."""
    rt_order = rng.permutation(RT60_MS).tolist()
    array_order = rng.permutation(SPACINGS).tolist()
    mic_order = rng.permutation(MICROPHONES).tolist()
    rt_base, rt_extra = divmod(count, len(rt_order))
    array_cursor = 0
    quotas = {}
    for rt_index, rt in enumerate(rt_order):
        rt_count = rt_base + (rt_index < rt_extra)
        array_base, array_extra = divmod(rt_count, len(array_order))
        extra_arrays = {
            array_order[(array_cursor + i) % len(array_order)]
            for i in range(array_extra)
        }
        array_cursor += array_extra
        mic_cursor = 0
        for spacing in array_order:
            array_count = array_base + (spacing in extra_arrays)
            mic_base, mic_extra = divmod(array_count, len(mic_order))
            extra_mics = {
                mic_order[(mic_cursor + i) % len(mic_order)]
                for i in range(mic_extra)
            }
            mic_cursor += mic_extra
            for mic in mic_order:
                quotas[rt, spacing, mic] = mic_base + (mic in extra_mics)
    return quotas


def _candidate_pairs(count, rng):
    """Cycle through every unordered pair with balanced speaker/source roles.

    For the production receiver quotas (41 or 42), the first complete block
    visits every source once on both speaker sides; the next partial block
    visits distinct sources on each side. No unordered pair is repeated.
    """
    result = []
    while len(result) < count:
        labels = rng.permutation(SOURCE_COUNT)
        blocks = []
        for offset in rng.permutation(np.arange(1, SOURCE_COUNT // 2)):
            order = rng.permutation(SOURCE_COUNT)
            block = np.column_stack(
                (labels[order], labels[(order + offset) % SOURCE_COUNT])
            )
            if rng.integers(2):
                block = block[:, ::-1]
            blocks.extend(block.tolist())
        half = SOURCE_COUNT // 2
        matching = np.column_stack((labels[:half], labels[half:]))
        result.extend(blocks)
        result.extend(matching.tolist())
    return np.asarray(result[:count], dtype=int).reshape(count, 2)


def _histogram(pairs):
    return np.bincount(
        [PAIR_INDEX[tuple(sorted(pair))] for pair in pairs],
        minlength=len(PAIR_TYPES),
    )


def assign_pairs(count=3000, seed=20261027):
    """Return shuffled records with distinct source ranks at one shared receiver.

    RT/array/microphone quotas are exact up to integer rounding. Candidate
    schedules also reduce source-pair imbalance across receivers; their actual
    global coverage is reported by the independent audit, rather than assumed.
    """
    if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    quotas = _receiver_quotas(int(count), rng)
    total = np.zeros(len(PAIR_TYPES), dtype=int)
    per_rt = {rt: np.zeros_like(total) for rt in RT60_MS}
    records = []
    keys = sorted(quotas)
    for position in rng.permutation(len(keys)):
        rt, spacing, mic = key = keys[position]
        quota = quotas[key]
        if not quota:
            continue
        best = None
        for _ in range(64):
            pairs = _candidate_pairs(quota, rng)
            histogram = _histogram(pairs)
            score = int(
                np.dot(total + histogram, total + histogram)
                + np.dot(per_rt[rt] + histogram, per_rt[rt] + histogram)
            )
            if best is None or score < best[0]:
                best = score, pairs, histogram
        _, pairs, histogram = best
        total += histogram
        per_rt[rt] += histogram
        records.extend(
            dict(rt60_ms=rt, spacing=spacing, microphone=mic, sources=pair.tolist())
            for pair in pairs
        )
    return [records[i] for i in rng.permutation(len(records))]


def audit_assignment(records, require_production=False):
    """Inspect generated assignments without calling the allocation algorithm."""
    rt_counts, array_counts, mic_counts = Counter(), Counter(), Counter()
    by_receiver = defaultdict(list)
    for record in records:
        rt, spacing, mic = (record[k] for k in ("rt60_ms", "spacing", "microphone"))
        first, second = record["sources"]
        if (
            rt not in RT60_MS or spacing not in SPACINGS or mic not in MICROPHONES
            or not 0 <= first < SOURCE_COUNT or not 0 <= second < SOURCE_COUNT
            or first == second
        ):
            raise ValueError("Invalid MIRD receiver or source pair")
        rt_counts[rt] += 1
        array_counts[spacing] += 1
        mic_counts[mic] += 1
        by_receiver[rt, spacing, mic].append((first, second))
    rows = []
    global_pairs = Counter()
    for rt, spacing, mic in product(RT60_MS, SPACINGS, MICROPHONES):
        pairs = by_receiver[rt, spacing, mic]
        counts = Counter(tuple(sorted(pair)) for pair in pairs)
        global_pairs.update(counts)
        roles = [np.bincount([p[side] for p in pairs], minlength=SOURCE_COUNT)
                 for side in (0, 1)]
        if require_production:
            if len(pairs) not in (41, 42) or len(counts) != len(pairs):
                raise ValueError("Expected 41/42 distinct source pairs per receiver")
            if any(np.ptp(side) > 1 for side in roles):
                raise ValueError("Unbalanced speaker/source roles at receiver")
        rows.append(dict(rt60_ms=rt, spacing=spacing, microphone=mic,
                         mixtures=len(pairs), unique_unordered_pairs=len(counts),
                         source_role_counts=[side.tolist() for side in roles]))
    if require_production:
        if len(records) != 3000 or set(rt_counts.values()) != {1000}:
            raise ValueError("Expected 3000 mixtures with 1000 per RT setting")
        if set(array_counts.values()) != {1000} or set(mic_counts.values()) != {375}:
            raise ValueError("Unbalanced array or microphone coverage")
        if len(global_pairs) != len(PAIR_TYPES):
            raise ValueError("Not all 325 source-pair types are represented")
    return dict(
        mixtures=len(records), physical_rooms=1, receiver_conditions=72,
        used_receiver_conditions=sum(row["mixtures"] > 0 for row in rows),
        rt_counts=dict(rt_counts), array_counts=dict(array_counts),
        microphone_counts=dict(mic_counts),
        unique_recorded_unordered_conditions=sum(row["unique_unordered_pairs"] for row in rows),
        available_recorded_unordered_conditions=23400,
        source_pair_types_covered=len(global_pairs),
        source_pair_count_min=min((global_pairs[pair] for pair in PAIR_TYPES), default=0),
        source_pair_count_max=max(global_pairs.values(), default=0),
        receivers=rows,
        limitation="Receiver recording IDs include overlapping physical microphone positions",
    )


def prepare_plan(filenames, scaling, output, seed=20261027):
    """Bind the production assignment to the original WHAMR test mixture IDs.

    This is an assignment plan, not an audio dump or a completion marker.
    Measured MIRD RIR files must be acquired and verified before synthesis.
    """
    filenames, scaling, output = map(Path, (filenames, scaling, output))
    if output.exists():
        raise ValueError("Use a new assignment-plan directory")
    with filenames.open() as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["output_filename"]: row for row in rows}
    with np.load(scaling, allow_pickle=True) as data:
        ids = [str(uid) for uid in data["utterance_id"]]
    if (len(ids) != 3000 or len(set(ids)) != len(ids)
            or len(by_id) != len(rows) or set(by_id) != set(ids)):
        raise ValueError("Expected matching, distinct 3000-mixture WHAMR test IDs")
    records = assign_pairs(len(ids), seed)
    for uid, record in zip(ids, records):
        row = by_id[uid]
        if Path(row["s1_path"]).parent.name == Path(row["s2_path"]).parent.name:
            raise ValueError("Expected different WSJ speakers")
        record.update(utterance_id=Path(uid).stem, filename=uid,
                      source_paths=[row["s1_path"], row["s2_path"]])
    audit = audit_assignment(records, require_production=True)
    payload = dict(
        schema=1, artifact_type="mird_two_speaker_assignment_plan",
        status="awaiting_measured_rir_data", seed=seed, count=3000,
        source_index_sha256={
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (filenames, scaling)
        },
        assignment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        planned_sample_rates=[8000, 16000], planned_conditions=["clean", "noisy"],
        source_grid=[dict(distance_m=distance, angle_deg=angle)
                     for distance in (1, 2) for angle in range(-90, 91, 15)],
        angle_convention="Paper broadside convention; native MAT mapping requires verification",
        records=records,
    )
    # A unique new directory ensures existing plans/data are never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    (output / "assignment_plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "assignment_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--filenames", type=Path, required=True)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    audit = prepare_plan(args.filenames, args.scaling, args.output, args.seed)
    print(json.dumps({key: value for key, value in audit.items() if key != "receivers"},
                     indent=2))


if __name__ == "__main__":
    main()
