"""Index the exact enh4_clean NF-WHAMR split without copying audio."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scp(path):
    result = {}
    for line in path.read_text().splitlines():
        uid, value = line.split(maxsplit=1)
        if uid in result:
            raise ValueError(f"Duplicate utterance in {path}: {uid}")
        if value.endswith("|"):
            raise ValueError(f"WAV files are required, not pipes: {path}")
        result[uid] = value
    return result


def prepare(source, whamr_root, output, workers=4, waveform_checks=8):
    source = source.resolve(strict=True)
    whamr_root = whamr_root.resolve(strict=True)
    output = output.absolute()
    recipe = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError(f"Dump already exists: {output}; use --verify-existing")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".damsep_prepare_", dir=output.parent))
    report = dict(
        source=str(source),
        whamr_root=str(whamr_root),
        sample_rate=8000,
        channel=0,
        noise=False,
        script_sha256=digest(Path(__file__)),
        input_hashes={},
        splits={},
    )
    try:
        for split in ("tr", "cv", "tt"):
            dataset = f"{split}_mix_clean_reverb_min_8k"
            baseline = source / "dump_clean/raw" / dataset
            maps = {}
            for target, original in (
                ("wav", "wav"),
                ("spk1", "spk1"),
                ("spk2", "spk2"),
            ):
                scp = baseline / f"{original}.scp"
                report["input_hashes"][str(scp)] = digest(scp)
                values = read_scp(scp)
                maps[target] = {
                    u: (source / p).resolve(strict=True) for u, p in values.items()
                }
            ids = sorted(maps["wav"])
            for speaker in (1, 2):
                scp = source / "data" / dataset / f"spk{speaker}_reverb.scp"
                report["input_hashes"][str(scp)] = digest(scp)
                values = read_scp(scp)
                if not set(ids).issubset(values):
                    raise ValueError(f"Missing reverberant sources in {scp}")
                # Old absolute /net/midgar/work/... paths in these lists are
                # deliberately resolved against the explicit WHAMR audio root.
                maps[f"spk{speaker}_reverb"] = {
                    u: (
                        whamr_root / split / f"s{speaker}_reverb" / Path(values[u]).name
                    ).resolve(strict=True)
                    for u in ids
                }
            if not ids or any(set(m) != set(ids) for m in maps.values()):
                raise ValueError(f"Mixture/source ID mismatch: {dataset}")

            def inspect(uid):
                headers = {k: sf.info(p[uid]) for k, p in maps.items()}
                if any(
                    h.samplerate != 8000 or h.channels != 2 for h in headers.values()
                ):
                    raise ValueError(f"{uid}: expected stereo 8 kHz baseline audio")
                lengths = {h.frames for h in headers.values()}
                if len(lengths) != 1 or min(lengths) <= 256:
                    raise ValueError(f"{uid}: unaligned or too short audio: {lengths}")
                return headers["wav"].frames

            with ThreadPoolExecutor(max_workers=workers) as pool:
                lengths = list(pool.map(inspect, ids))
            checks = (
                ids[:: max(1, len(ids) // waveform_checks)][:waveform_checks]
                if waveform_checks
                else []
            )
            worst_sum_error = 0.0
            worst_clean_error = 0.0
            for uid in checks:
                audio = {
                    k: sf.read(p[uid], dtype="float32", always_2d=True)[0][:, 0]
                    for k, p in maps.items()
                }
                if any(not np.isfinite(x).all() for x in audio.values()):
                    raise ValueError(f"{uid}: nonfinite audio")
                error = float(
                    np.max(
                        np.abs(
                            audio["wav"] - audio["spk1_reverb"] - audio["spk2_reverb"]
                        )
                    )
                )
                # Existing formatted mixture is PCM16; original reverb images
                # are float WAVs. Allow only the expected quantization error.
                if error > 4e-5:
                    raise ValueError(
                        f"{uid}: mixture is not the sum of reverb sources ({error})"
                    )
                worst_sum_error = max(worst_sum_error, error)
                for speaker in (1, 2):
                    name = maps[f"spk{speaker}_reverb"][uid].name
                    direct_path = whamr_root / split / f"s{speaker}_anechoic" / name
                    direct = sf.read(direct_path, dtype="float32", always_2d=True)[0][
                        :, 0
                    ]
                    error = float(np.max(np.abs(direct - audio[f"spk{speaker}"])))
                    if error > 4e-5:
                        raise ValueError(
                            f"{uid}: clean reference pairing/gain mismatch ({error})"
                        )
                    worst_clean_error = max(worst_clean_error, error)
            dest = temporary / "raw" / dataset
            dest.mkdir(parents=True)
            for name, values in maps.items():
                (dest / f"{name}.scp").write_text(
                    "".join(f"{u} {os.path.relpath(values[u], recipe)}\n" for u in ids)
                )
            (dest / "speech_mix_shape").write_text(
                "".join(f"{u} {n}\n" for u, n in zip(ids, lengths))
            )
            (dest / "utt2num_samples").write_text(
                "".join(f"{u} {n}\n" for u, n in zip(ids, lengths))
            )
            (dest / "feats_type").write_text("raw\n")
            report["splits"][split] = dict(
                count=len(ids),
                all_headers_checked=True,
                waveform_ids=checks,
                max_sum_error=worst_sum_error,
                max_clean_error=worst_clean_error,
            )
            print(
                f"{split}: {len(ids)} aligned 8 kHz examples; "
                f"{len(checks)} waveform checks",
                flush=True,
            )
        report["output_hashes"] = {
            str(p.relative_to(temporary)): digest(p)
            for p in temporary.rglob("*")
            if p.is_file()
        }
        (temporary / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../enh4_clean"))
    parser.add_argument(
        "--whamr-root",
        type=Path,
        default=Path("../enh1/data/whamr/2speakers/wav8k/min"),
    )
    parser.add_argument("--output", type=Path, default=Path("dump_nf_8k_min"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--waveform-checks", type=int, default=8)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.waveform_checks < 0:
        parser.error("workers must be positive and waveform-checks nonnegative")
    if args.verify_existing and args.output.exists():
        report = json.loads((args.output / "preparation.json").read_text())
        if report["script_sha256"] != digest(Path(__file__)):
            raise ValueError("Preparation code changed; use a new output dump")
        if report["source"] != str(args.source.resolve()) or report[
            "whamr_root"
        ] != str(args.whamr_root.resolve()):
            raise ValueError("Existing dump uses different audio roots")
        for path, expected in report["input_hashes"].items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Source list changed: {path}")
        for path, expected in report["output_hashes"].items():
            if digest(args.output / path) != expected:
                raise ValueError(f"Prepared list changed: {path}")
        print("Existing dump verified:", args.output)
        return
    prepare(
        args.source, args.whamr_root, args.output, args.workers, args.waveform_checks
    )


if __name__ == "__main__":
    main()
