"""Load checked native MIRD responses and save selected mono receiver RIRs.

Native archives remain untouched. Each required eight-channel MAT is read once;
the selected microphone column retains its original timing and complete tail.
"""

import zipfile
from collections import defaultdict
from numbers import Integral
from pathlib import Path

import numpy as np

import create_whamr_but_dump as common
from mird_pair_assignment import MICROPHONES, SOURCE_COUNT
from mird_rir import (
    NATIVE_ANGLES,
    NATIVE_FS,
    NATIVE_SAMPLES,
    archive_stem,
    inspect_response,
    member_name,
)


def _source_positions(source_grid):
    """Bind planner source indices to verified native azimuths explicitly."""
    if len(source_grid) != SOURCE_COUNT:
        raise ValueError("Expected exactly 26 MIRD source positions")
    positions = []
    for entry in source_grid:
        try:
            distance = entry["distance_m"]
            angle = entry["angle_deg"]
            native_angle = entry["native_azimuth_deg"]
        except (KeyError, TypeError) as error:
            raise ValueError("Source grid lacks explicit native azimuths") from error
        if (distance not in (1, 2) or isinstance(distance, bool)
                or not isinstance(angle, Integral) or isinstance(angle, bool)
                or not isinstance(native_angle, Integral)
                or isinstance(native_angle, bool)
                or native_angle not in NATIVE_ANGLES
                or angle < -90 or angle > 90 or angle % 15
                or native_angle != angle % 360):
            raise ValueError("Invalid signed/native MIRD source coordinates")
        positions.append((int(distance), int(native_angle)))
    expected = {(distance, angle) for distance in (1, 2)
                for angle in NATIVE_ANGLES}
    if len(set(positions)) != SOURCE_COUNT or set(positions) != expected:
        raise ValueError("MIRD source grid must cover every native position once")
    return positions


def prepare_rirs(archives, records, source_grid, out, rates, process_rir):
    """Return cached mono RIRs and provenance for the plan's selected receivers.

    ``process_rir(wave, native_rate, target_rate)`` must return the complete
    resampled, peak-normalized response, its peak index, and its normalization
    divisor. Cache keys are ``(rate, rt60_ms, spacing, mic, source_index)``.
    """
    archives, out = Path(archives), Path(out)
    positions = _source_positions(source_grid)
    rates = tuple(rates)
    if (not rates or len(set(rates)) != len(rates)
            or any(not isinstance(rate, Integral) or isinstance(rate, bool)
                   or rate <= 0 for rate in rates)):
        raise ValueError("Expected distinct positive integer sample rates")
    if not out.is_dir():
        raise ValueError("Dataset output directory must already exist")
    needed = defaultdict(lambda: defaultdict(set))
    for record in records:
        rt60_ms, spacing = record["rt60_ms"], record["spacing"]
        archive_stem(rt60_ms, spacing)  # Reject unsafe or unknown conditions.
        microphone = record["microphone"]
        sources = record["sources"]
        if (not isinstance(microphone, Integral) or isinstance(microphone, bool)
                or microphone not in MICROPHONES):
            raise ValueError("MIRD microphone must be an integer from 1 to 8")
        if (len(sources) != 2 or sources[0] == sources[1]
                or any(not isinstance(index, Integral) or isinstance(index, bool)
                       or index < 0 or index >= SOURCE_COUNT for index in sources)):
            raise ValueError("Expected two distinct MIRD source indices")
        for source_index in sources:
            needed[rt60_ms, spacing][int(source_index)].add(int(microphone))

    cached, metadata = {}, []
    for (rt60_ms, spacing), source_microphones in sorted(needed.items()):
        native_archive = archives / f"{archive_stem(rt60_ms, spacing)}.zip"
        archive_sha256 = common.digest(native_archive)
        with zipfile.ZipFile(native_archive) as archive:
            names = [member.filename for member in archive.infolist()
                     if not member.is_dir()]
            expected = {member_name(rt60_ms, spacing, distance, angle)
                        for distance in (1, 2) for angle in NATIVE_ANGLES}
            if len(names) != SOURCE_COUNT or set(names) != expected:
                raise ValueError(
                    "Archive must contain the 26 declared native MATs: "
                    + str(native_archive)
                )
            for source_index, microphones in sorted(source_microphones.items()):
                distance, native_angle = positions[source_index]
                native_member = member_name(rt60_ms, spacing, distance, native_angle)
                native, _ = inspect_response(
                    archive.read(native_member), rt60_ms, spacing,
                    distance, native_angle,
                )
                for microphone in sorted(microphones):
                    wave = native[:, microphone - 1]
                    for sr in rates:
                        h, peak, divisor = process_rir(wave, NATIVE_FS, sr)
                        h = np.asarray(h)
                        if (h.ndim != 1 or len(h) != 10 * sr
                                or not np.all(np.isfinite(h))
                                or not isinstance(peak, Integral)
                                or peak != int(np.argmax(np.abs(h)))
                                or not np.isfinite(divisor) or divisor <= 0
                                or not np.isclose(np.max(np.abs(h)), 1,
                                                  rtol=0, atol=1e-7)):
                            raise ValueError(
                                "RIR processing must retain ten seconds "
                                "and normalize its peak"
                            )
                        filename = (
                            f"rt{rt60_ms}_{spacing}_source{source_index:02d}"
                            f"_mic{microphone:02d}.wav"
                        )
                        target = out / "audio" / str(sr) / "rir" / filename
                        if target.exists():
                            raise FileExistsError(
                                f"Refusing existing RIR output: {target}"
                            )
                        path = common.write_audio(target, h, sr)
                        key = (sr, rt60_ms, spacing, microphone, source_index)
                        cached[key] = (h, int(peak), path)
                        metadata.append(dict(
                            sample_rate=int(sr), rt60_ms=rt60_ms, spacing=spacing,
                            microphone=microphone, source_index=source_index,
                            distance_m=distance, native_azimuth_deg=native_angle,
                            native_archive=str(native_archive.resolve()),
                            native_member=native_member,
                            archive_sha256=archive_sha256, path=path,
                            samples=len(h), peak_sample=int(peak),
                            normalization_divisor=float(divisor),
                            sha256=common.digest(path),
                            native_sample_rate=NATIVE_FS,
                            native_samples=NATIVE_SAMPLES,
                        ))
    return cached, metadata
