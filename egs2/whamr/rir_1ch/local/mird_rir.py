#!/usr/bin/env python3
"""Check native MIRD archives against their declared recording conditions.

ZIP CRC and filenames alone do not establish the acoustic condition. Check
every native MAT's metadata and eight-channel signal before allowing synthesis.
Never extract archives or execute the accompanying MATLAB package.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mird_pair_assignment import RT60_MS, SPACINGS


PREFIX = "Impulse_response_Acoustic_Lab_Bar-Ilan_University_"
NATIVE_ANGLES = (0, 15, 30, 45, 60, 75, 90, 270, 285, 300, 315, 330, 345)
NATIVE_FS = 48000
NATIVE_SAMPLES = 480000


def sha256(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def archive_stem(rt60_ms, spacing):
    if rt60_ms not in RT60_MS or spacing not in SPACINGS:
        raise ValueError("Unknown MIRD recording condition")
    return f"{PREFIX}(Reverberation_{rt60_ms / 1000:.3f}s)_{spacing}"


def member_name(rt60_ms, spacing, distance, angle):
    if distance not in (1, 2) or angle not in NATIVE_ANGLES:
        raise ValueError("Unknown native MIRD source position")
    return f"{archive_stem(rt60_ms, spacing)}_{distance}m_{angle:03d}.mat"


def inspect_response(payload, rt60_ms, spacing, distance, angle):
    """Return checked native audio and a small metadata record; retain timing."""
    mat = loadmat(io.BytesIO(payload), squeeze_me=True, struct_as_record=False)
    try:
        meta, sim = mat["metapar"], mat["simpar"]
        checks = {
            "type": meta.type == "impulse_response",
            "room": meta.room == "Acoustic_Lab_Bar-Ilan_University",
            "reverberation": np.isclose(float(meta.reverberation),
                                        rt60_ms / 1000, rtol=0, atol=1e-9),
            "spacing": np.array_equal(np.asarray(meta.mic_spacing),
                                      [int(x) for x in spacing.split("-")]),
            "distance": float(meta.distance) == distance,
            "azimuth": float(meta.azimuth) == angle,
            "mic_position": meta.mic_position == "mic1@90deg, mic8@270deg",
            "sample_rate": float(sim.fs) == NATIVE_FS,
            "channel_order": np.array_equal(np.asarray(sim.in_ch),
                                            np.arange(1, 9)),
        }
        rir = np.asarray(mat["impulse_response"])
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or malformed native MAT fields: {error}") from error
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Native MAT condition mismatch: {', '.join(failed)}")
    if rir.shape != (NATIVE_SAMPLES, 8) or not np.issubdtype(rir.dtype, np.floating):
        raise ValueError("Expected ten seconds of eight-channel floating-point RIR")
    if not np.all(np.isfinite(rir)) or np.any(np.max(np.abs(rir), axis=0) == 0):
        raise ValueError("Native RIR contains nonfinite values or a silent channel")
    record = dict(rt60_ms=rt60_ms, spacing=spacing, distance_m=distance,
                  native_azimuth_deg=angle, sample_rate=NATIVE_FS,
                  samples=NATIVE_SAMPLES, channels=8,
                  peak_samples=np.argmax(np.abs(rir), axis=0).tolist())
    return rir, record


def inspect_archive(path, rt60_ms, spacing):
    path = Path(path)
    with path.open("rb") as stream:
        checksum = sha256(stream)
    report = dict(path=str(path.resolve()), bytes=path.stat().st_size,
                  sha256=checksum, rt60_ms=rt60_ms, spacing=spacing)
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if not x.is_dir()]
        expected = {member_name(rt60_ms, spacing, d, a): (d, a)
                    for d in (1, 2) for a in NATIVE_ANGLES}
        names = [x.filename for x in members]
        if len(names) != 26 or set(names) != set(expected):
            report.update(valid=False, error="Expected exactly 26 native MAT files",
                          archive_members=names,
                          missing_members=sorted(set(expected) - set(names)))
            # Record evidence for mispackaged official archives without using
            # nested ZIP data as if it belonged to the declared condition.
            nested = []
            for member in members:
                if member.filename.endswith(".zip"):
                    with archive.open(member) as stream:
                        nested.append(dict(name=member.filename,
                                           sha256=sha256(stream)))
            report["nested_archives"] = nested
            return report
        recordings = []
        for name, (distance, angle) in expected.items():
            try:
                _, record = inspect_response(archive.read(name), rt60_ms,
                                             spacing, distance, angle)
            except (ValueError, OSError, zipfile.BadZipFile) as error:
                report.update(valid=False, error=f"{name}: {error}")
                return report
            recordings.append(dict(member=name, **record))
    report.update(valid=True, native_recordings=len(recordings),
                  recordings=recordings)
    return report


def audit_dataset(root):
    root = Path(root)
    archives = []
    for rt60_ms in RT60_MS:
        for spacing in SPACINGS:
            path = root / f"{archive_stem(rt60_ms, spacing)}.zip"
            print(f"Checking {path.name}", flush=True)
            try:
                result = inspect_archive(path, rt60_ms, spacing)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                result = dict(path=str(path.resolve()), rt60_ms=rt60_ms,
                              spacing=spacing, valid=False, error=str(error))
            archives.append(result)
    valid = sum(row["valid"] for row in archives)
    return dict(schema=1,
                status="ready" if valid == 9 else "blocked_missing_native_rir",
                all_conditions_valid=valid == 9, valid_conditions=valid,
                expected_conditions=9,
                native_recordings=sum(row.get("native_recordings", 0)
                                      for row in archives),
                native_angle_convention="0 front; 90 left (mic1); 270 right (mic8)",
                archives=archives)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.archives)
    # Refuse overwriting an earlier audit or unrelated output.
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "archives"}, indent=2))
    if not report["all_conditions_valid"]:
        raise SystemExit(
            "MIRD source verification failed; synthesis must remain blocked"
        )


if __name__ == "__main__":
    main()
