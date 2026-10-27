"""Cover all 2,925 MIRD source-pair/session cases, then add 75 mixtures.

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
BASE_CASE_COUNT = len(PAIR_TYPES) * len(RT60_MS) * len(SPACINGS)
PRODUCTION_COUNT = 3000


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


def _complete_pairs(rng, high_sources=None):
    """Orient every edge of K26 once; each source has 12/13 slots per side.

    The first 12 blocks are directed cycle covers (one incoming/outgoing edge
    per vertex); the final matching gives 13 chosen vertices one extra outgoing
    edge, and the other 13 one extra incoming edge.
    """
    half = SOURCE_COUNT // 2
    if high_sources is None:
        labels = rng.permutation(SOURCE_COUNT)
    else:
        high = sorted(high_sources)
        low = sorted(set(range(SOURCE_COUNT)) - set(high))
        labels = np.concatenate((rng.permutation(high), rng.permutation(low)))
    blocks = []
    for offset in rng.permutation(np.arange(1, half)):
        order = rng.permutation(SOURCE_COUNT)
        block = np.column_stack(
            (labels[order], labels[(order + offset) % SOURCE_COUNT])
        )
        if rng.integers(2):
            block = block[:, ::-1]
        blocks.append(block)
    blocks.append(np.column_stack((labels[:half], labels[half:])))
    return np.concatenate(blocks)


def _additional_pairs(rng):
    """75 distinct pairs, with each source used 2/3 times on each side."""
    pairs = _complete_pairs(rng)
    third_block = pairs[2 * SOURCE_COUNT:3 * SOURCE_COUNT]
    removed, occupied = set(), set()
    for position in rng.permutation(SOURCE_COUNT):
        pair = third_block[position]
        if not occupied.intersection(pair):
            removed.add(int(position))
            occupied.update(pair)
            if len(removed) == 3:
                break
    # Removing three vertex-disjoint edges keeps the deficient incoming and
    # outgoing source sets disjoint, so both speaker roles can be balanced later.
    keep = [i for i in range(SOURCE_COUNT) if i not in removed]
    return np.concatenate((pairs[:2 * SOURCE_COUNT], third_block[keep]))


def _assign_microphones(records, quotas, rng):
    """Select a seeded random channel under balanced receiver-count quotas."""
    channels = np.repeat(MICROPHONES, [quotas[mic] for mic in MICROPHONES])
    rng.shuffle(channels)
    base = {
        tuple(sorted(record["sources"])): i
        for i, record in enumerate(records) if record["assignment_phase"] == "base"
    }
    additional = [i for i, record in enumerate(records)
                  if record["assignment_phase"] == "additional"]
    extra_pairs = {tuple(sorted(records[i]["sources"])) for i in additional}
    singles = [i for pair, i in base.items() if pair not in extra_pairs]
    for i in additional:
        original = base[tuple(sorted(records[i]["sources"]))]
        if channels[i] == channels[original]:
            # Swap with a single-occurrence case: this preserves all channel
            # quotas and cannot create another repeated full condition.
            choices = [j for j in singles if channels[j] != channels[i]]
            partner = int(rng.choice(choices))
            channels[i], channels[partner] = channels[partner], channels[i]
    for record, microphone in zip(records, channels):
        record["microphone"] = int(microphone)


def assign_pairs(count=3000, seed=20261027):
    """Cover 325 unordered pairs in each of nine RT/array recording sessions.

    Choose one balanced random microphone per base case, and use another
    microphone for each of 75 additional cases. Additional pair types are
    distinct globally. Smaller counts are explicit subsets of the base phase,
    followed by additional cases only after all 2,925 base cases are included.
    """
    if (not isinstance(count, Integral) or isinstance(count, bool)
            or not 0 <= count <= PRODUCTION_COUNT):
        raise ValueError("count must be an integer from 0 through 3000")
    if not count:
        return []
    rng = np.random.default_rng(seed)
    quotas = _receiver_quotas(PRODUCTION_COUNT, rng)
    additional = _additional_pairs(rng)
    out_counts = np.bincount(additional[:, 0], minlength=SOURCE_COUNT)
    in_counts = np.bincount(additional[:, 1], minlength=SOURCE_COUNT)
    high = set(np.flatnonzero(out_counts == 2).tolist())
    eligible = sorted(set(range(SOURCE_COUNT)) - high
                      - set(np.flatnonzero(in_counts == 2).tolist()))
    high.update(rng.choice(eligible, SOURCE_COUNT // 2 - len(high), replace=False).tolist())
    low = set(range(SOURCE_COUNT)) - high
    additional = additional[rng.permutation(len(additional))]
    sessions = list(product(RT60_MS, SPACINGS))
    records = []
    cursor = 0
    for position, session_index in enumerate(rng.permutation(len(sessions))):
        rt, spacing = sessions[session_index]
        receiver_quotas = {mic: quotas[rt, spacing, mic] for mic in MICROPHONES}
        extra_count = sum(receiver_quotas.values()) - len(PAIR_TYPES)
        pairs = _complete_pairs(rng, high if position % 2 == 0 else low)
        current = [dict(rt60_ms=rt, spacing=spacing, sources=pair.tolist(),
                        assignment_phase="base") for pair in pairs]
        current.extend(
            dict(rt60_ms=rt, spacing=spacing, sources=pair.tolist(),
                 assignment_phase="additional")
            for pair in additional[cursor:cursor + extra_count]
        )
        cursor += extra_count
        _assign_microphones(current, receiver_quotas, rng)
        records.extend(current)
    records = [records[i] for i in rng.permutation(len(records))]
    if count < PRODUCTION_COUNT:
        base = [r for r in records if r["assignment_phase"] == "base"]
        extras = [r for r in records if r["assignment_phase"] == "additional"]
        records = base[:min(count, BASE_CASE_COUNT)] + extras[:max(0, count - BASE_CASE_COUNT)]
        records = [records[i] for i in rng.permutation(len(records))]
    return records


def audit_assignment(records, require_production=False):
    """Inspect generated assignments without calling the allocation algorithm."""
    rt_counts, array_counts, mic_counts = Counter(), Counter(), Counter()
    by_receiver = defaultdict(list)
    cases, base_cases, extra_cases, extra_pairs = Counter(), Counter(), Counter(), Counter()
    base_microphones, extra_microphones = {}, {}
    for record in records:
        rt, spacing, mic = (record[k] for k in ("rt60_ms", "spacing", "microphone"))
        first, second = record["sources"]
        if (
            rt not in RT60_MS or spacing not in SPACINGS or mic not in MICROPHONES
            or not 0 <= first < SOURCE_COUNT or not 0 <= second < SOURCE_COUNT
            or first == second
            or record.get("assignment_phase") not in ("base", "additional")
        ):
            raise ValueError("Invalid MIRD receiver or source pair")
        rt_counts[rt] += 1
        array_counts[spacing] += 1
        mic_counts[mic] += 1
        by_receiver[rt, spacing, mic].append((first, second))
        pair = tuple(sorted((first, second)))
        case = rt, spacing, pair
        cases[case] += 1
        if record["assignment_phase"] == "base":
            base_cases[case] += 1
            base_microphones[case] = mic
        else:
            extra_cases[case] += 1
            extra_pairs[pair] += 1
            extra_microphones[case] = mic
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
        rows.append(dict(rt60_ms=rt, spacing=spacing, microphone=mic,
                         mixtures=len(pairs), unique_unordered_pairs=len(counts),
                         source_role_counts=[side.tolist() for side in roles]))
    global_roles = [np.bincount([r["sources"][side] for r in records], minlength=SOURCE_COUNT)
                    for side in (0, 1)]
    if require_production:
        if len(records) != 3000 or set(rt_counts.values()) != {1000}:
            raise ValueError("Expected 3000 mixtures with 1000 per RT setting")
        if set(array_counts.values()) != {1000} or set(mic_counts.values()) != {375}:
            raise ValueError("Unbalanced array or microphone coverage")
        if len(global_pairs) != len(PAIR_TYPES):
            raise ValueError("Not all 325 source-pair types are represented")
        expected = set(product(RT60_MS, SPACINGS, PAIR_TYPES))
        if set(cases) != expected or base_cases != Counter({key: 1 for key in expected}):
            raise ValueError("Expected one base occurrence of every one of 2925 cases")
        if (len(extra_cases) != 75 or set(extra_cases.values()) != {1}
                or len(extra_pairs) != 75 or set(extra_pairs.values()) != {1}):
            raise ValueError("Expected 75 distinct additional cases and source-pair types")
        if any(extra_microphones[key] == base_microphones[key] for key in extra_cases):
            raise ValueError("Additional cases must use a different microphone")
        if any(np.ptp(role) > 1 for role in global_roles):
            raise ValueError("Unbalanced global speaker/source roles")
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
        pair_session_cases_available=BASE_CASE_COUNT,
        pair_session_cases_covered=len(cases),
        base_records=sum(base_cases.values()), additional_records=sum(extra_cases.values()),
        additional_cases=len(extra_cases),
        pair_session_count_histogram=dict(Counter(cases.values())),
        global_source_role_counts=[role.tolist() for role in global_roles],
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
        schema=2, artifact_type="mird_two_speaker_assignment_plan",
        status="awaiting_measured_rir_data", seed=seed, count=3000,
        sampling="all_2925_pair_rt_array_cases_plus_75_distinct_additions",
        microphone_selection="seeded balanced random; additions differ from their base mic",
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
