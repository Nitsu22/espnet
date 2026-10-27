"""Balanced, reproducible BUT source pairs for a fixed number of mixtures.

Groups identify a room and one receiver.  Sorted RIR indices within each group
must have the same source-position order across receivers in a room.  This is
the order of the source-position names in the BUT inventory.
"""

from collections import defaultdict
from numbers import Integral

import numpy as np


def _pair_schedule(size, count, rng):
    """Cover every unordered pair before reuse, with balanced speaker roles."""
    pairs = []
    while len(pairs) < count:
        labels = rng.permutation(size)
        blocks = []
        for offset in rng.permutation(np.arange(1, (size + 1) // 2)):
            order = rng.permutation(size)
            block = np.column_stack((labels[order], labels[(order + offset) % size]))
            if rng.integers(2):
                block = block[:, ::-1]
            blocks.append(block)
        matching = np.empty((0, 2), dtype=int)
        if size % 2 == 0:
            matching = np.column_stack((labels[: size // 2], labels[size // 2 :]))
            matching = matching[rng.permutation(len(matching))]
            flip = rng.integers(2, size=len(matching)).astype(bool)
            matching[flip] = matching[flip, ::-1]
        # Each full offset visits every source once on each speaker side.
        # For even sizes the final matching leaves a one-count difference;
        # reverse that matching first to restore equal counts in the next cycle.
        first = np.concatenate(blocks + [matching], axis=0)
        second = np.concatenate([matching[:, ::-1]] + [b[:, ::-1] for b in blocks])
        pairs.extend(first.tolist())
        pairs.extend(second.tolist())
    return np.asarray(pairs[:count], dtype=int).reshape(count, 2)


def _pair_histogram(pairs, size):
    lo, hi = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(
        pairs[:, 0], pairs[:, 1]
    )
    # Lexicographic index of (lo, hi) among the upper-triangular entries.
    index = lo * (2 * size - lo - 1) // 2 + hi - lo - 1
    return np.bincount(index, minlength=size * (size - 1) // 2)


def _balanced_schedules(size, quotas, rng):
    """Keep room-wide pair counts close while preserving receiver guarantees."""
    pair_types = size * (size - 1) // 2
    total = np.zeros(pair_types, dtype=int)
    schedules, histograms = [], []
    candidates = 64 if pair_types > 1 else 1
    for quota in quotas:
        best = None
        for _ in range(candidates):
            pairs = _pair_schedule(size, quota, rng)
            histogram = _pair_histogram(pairs, size)
            score = int(np.dot(total + histogram, total + histogram))
            if best is None or score < best[0]:
                best = score, pairs, histogram
        _, pairs, histogram = best
        schedules.append(pairs)
        histograms.append(histogram)
        total += histogram

    # Coordinate descent can improve the last few overrepresented pair types.
    # The lower bound is attainable exactly when every pair count differs by <=1.
    base, extra = divmod(sum(quotas), pair_types)
    lower_bound = pair_types * base * base + extra * (2 * base + 1)
    score = int(np.dot(total, total))
    for _ in range(6):
        if score == lower_bound:
            break
        improved = False
        for position in rng.permutation(len(quotas)):
            remaining = total - histograms[position]
            for _ in range(candidates):
                pairs = _pair_schedule(size, quotas[position], rng)
                histogram = _pair_histogram(pairs, size)
                proposed = remaining + histogram
                proposed_score = int(np.dot(proposed, proposed))
                if proposed_score < score:
                    schedules[position] = pairs
                    histograms[position] = histogram
                    total = proposed
                    score = proposed_score
                    improved = True
            if score == lower_bound:
                break
        if not improved:
            break
    return schedules


def assign_all_microphone_pairs(groups, count, seed):
    """Assign distinct sources at a shared receiver to ``count`` mixtures.

    ``groups`` maps ``(room, microphone_setup, microphone)`` to distinct global
    integer RIR indices, one per physical source position.  Input iteration
    order does not affect the result.  All groups require at least two sources.

    Room totals differ by at most one; within each room receiver totals differ
    by at most one.  If rooms share the same receiver IDs, their global totals
    also differ by at most one.  For each receiver, unordered pair counts and
    each speaker's source-position counts differ by at most one.  Room-wide
    pair-type balance is optimized heuristically, without an exact guarantee.
    Finally, shuffle assignments to avoid a room/receiver block in utterance
    order.  Seeded NumPy randomness makes all these choices reproducible.
    """
    if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    normalized = {}
    used_indices = set()
    for key, values in groups.items():
        if not isinstance(key, tuple) or len(key) != 3 or not all(
            isinstance(part, str) for part in key
        ):
            raise ValueError("group keys must be (room, setup, microphone) strings")
        values = list(values)
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError("RIR indices must be nonnegative integers")
        indices = sorted(int(value) for value in values)
        if len(indices) < 2 or len(set(indices)) != len(indices):
            raise ValueError("each receiver needs at least two distinct RIR indices")
        if used_indices.intersection(indices):
            raise ValueError("RIR indices must belong to exactly one receiver")
        used_indices.update(indices)
        normalized[key] = np.asarray(indices, dtype=int)
    if not normalized:
        if count:
            raise ValueError("cannot assign mixtures without receiver groups")
        return np.empty((0, 2), dtype=int)

    rng = np.random.default_rng(seed)
    rooms = defaultdict(list)
    for key in sorted(normalized):
        rooms[key[0]].append(key)
    room_order = [sorted(rooms)[i] for i in rng.permutation(len(rooms))]
    room_base, room_extra = divmod(count, len(rooms))
    signatures = [{key[1:] for key in rooms[room]} for room in room_order]
    shared_receivers = all(signature == signatures[0] for signature in signatures)
    if shared_receivers:
        receiver_ids = sorted(signatures[0])
        receiver_order = [receiver_ids[i] for i in rng.permutation(len(receiver_ids))]
    cursor = 0
    assignments = []
    for room_position, room in enumerate(room_order):
        keys = rooms[room]
        room_quota = room_base + (room_position < room_extra)
        receiver_base, receiver_extra = divmod(room_quota, len(keys))
        if shared_receivers:
            extras = {
                receiver_order[(cursor + offset) % len(receiver_order)]
                for offset in range(receiver_extra)
            }
            cursor += receiver_extra
        else:
            extras = {
                keys[i][1:] for i in rng.permutation(len(keys))[:receiver_extra]
            }
        quotas = {key: receiver_base + (key[1:] in extras) for key in keys}
        # Source-count subsets support inventories with missing source positions;
        # source ranks must agree among receivers in any one subset.
        sizes = defaultdict(list)
        for key in keys:
            sizes[len(normalized[key])].append(key)
        for size in sorted(sizes):
            active = [key for key in sizes[size] if quotas[key]]
            if not active:
                continue
            active = [active[i] for i in rng.permutation(len(active))]
            schedules = _balanced_schedules(size, [quotas[key] for key in active], rng)
            for key, schedule in zip(active, schedules):
                assignments.append(normalized[key][schedule])
    result = (
        np.concatenate(assignments, axis=0)
        if assignments
        else np.empty((0, 2), dtype=int)
    )
    rng.shuffle(result)
    return result
