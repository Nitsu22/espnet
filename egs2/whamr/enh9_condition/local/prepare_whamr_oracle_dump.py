"""Reuse verified Direct/Reverb Sweep CTFs with the exact noisy Baseline audio.

All utterance IDs, both clean teachers and lengths must match the NF source.
Where its independent noise track is retained, also check noisy = NF + noise
up to PCM quantization. Test noise tracks are absent from the saved Baseline;
that split is checked by IDs/teachers/lengths and a nonzero mixture difference.
Audio and CTF arrays are symlinked, never copied or regenerated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from prepare_oracle_ctf_dump import (
    AUDIO_NAMES, COUNTS, digest, load_wave, read_scp, resolve, validate_prepared_dump,
)

SUFFIX = "mix_both_reverb_min_8k"
SHARED_RUNTIME = (
    "espnet2/enh/separator/tflocoformer_separator_nocashe.py",
    "espnet2/enh/encoder/stft_encoder.py",
    "espnet2/enh/decoder/stft_decoder.py",
    "espnet2/enh/loss/criterions/time_domain.py",
    "espnet2/enh/loss/wrappers/pit_solver.py",
    "espnet2/bin/enh_scoring.py",
    "espnet2/fileio/sound_scp.py",
)


def compare_config(config, baseline):
    current, saved = [yaml.safe_load(Path(p).read_text()) for p in (config, baseline)]
    for key, value in current.items():
        if key not in ("separator", "separator_conf", "model_conf") and saved.get(key) != value:
            raise ValueError("Baseline training setting differs: " + key)
    trunk = {key: value for key, value in current["separator_conf"].items()
             if key not in ("ctf_taps", "random_ctf_swap")}
    if (trunk != saved["separator_conf"] or saved["separator"] != "tflocoformer_nocashe"
            or current["separator"] != "tflocoformer_nocashe_ctf_concat"
            or saved["model_conf"] != {"normalize_variance": True}
            or current["model_conf"] != {"normalize_variance": True, "rir_condition_type": "rir_ctf"}
            or saved["ngpu"] != 1 or saved["accum_grad"] != 1
            or saved["fold_length"] != [200000] * 3):
        raise ValueError("Baseline architecture, normalization or batching differs")


def verify_example(uid, paths, clean, source, noise=None):
    waves, headers = {}, {}
    for name in AUDIO_NAMES:
        waves[name], headers[name] = load_wave(paths[name])
        if (headers[name]["frames"] != source["wav.scp"]["frames"]
                or not 128 < headers[name]["frames"] < 200000):
            raise ValueError("Baseline/NF lengths differ: " + uid)
    for name in ("spk1.scp", "spk2.scp"):
        if headers[name]["sha256"] != source[name]["sha256"]:
            raise ValueError("Oracle speaker/clean teacher mismatch: " + uid + "/" + name)
    clean_wave, _ = load_wave(clean)
    difference = waves["wav.scp"] - clean_wave
    if not np.any(np.abs(difference) > 3 / 32768):
        raise ValueError("Expected a noisy mixture: " + uid)
    check = dict(noise_track_checked=noise is not None,
                 mixture_difference_rms=float(np.sqrt(np.mean(difference ** 2))))
    if noise is not None:
        noise_wave, noise_header = load_wave(noise)
        if noise_wave.shape != clean_wave.shape or not np.any(noise_wave):
            raise ValueError("Invalid independent noise track: " + uid)
        # All tracks were independently PCM16-quantized by the original recipe.
        # Compare only unsaturated NF samples; pre-existing NF clipping loses
        # information and is recorded separately rather than treated as noise.
        interior = (clean_wave > -1) & (clean_wave < 32767 / 32768)
        expected = np.clip(clean_wave + noise_wave, -1, 32767 / 32768)
        residual = float(np.max(np.abs(waves["wav.scp"][interior] - expected[interior])))
        if residual > 3 / 32768 + 1e-10:
            raise ValueError(f"Noisy/NF mixture correspondence differs: {uid}/{residual}")
        check.update(maximum_noise_residual=residual, noise_sha256=noise_header["sha256"],
                     excluded_saturated_nf_samples=int(np.count_nonzero(~interior)))
    return headers, check


def validate(output, require_full=True, workers=4):
    output = Path(output).resolve(strict=True)
    report = json.loads((output / "preparation.json").read_text())
    if (report.get("schema_version") != "whamr_oracle_v1" or report.get("status") != "complete"
            or report["spec"].get("condition") != "whamr"
            or (require_full and report["counts"] != COUNTS)):
        raise ValueError("Incomplete noisy WHAMR Oracle dump")
    source = Path(report["spec"]["source_dump"])
    if digest(source / "preparation.json") != report["spec"]["source_manifest_sha256"]:
        raise ValueError("Oracle source manifest changed")
    native = validate_prepared_dump(source, require_full=require_full, workers=workers)
    if report["counts"] != native["counts"]:
        raise ValueError("Oracle source counts changed")
    for name, target in report["links"].items():
        path = output / name
        if not path.is_symlink() or os.readlink(path) != target or not path.is_dir():
            raise ValueError("Oracle/audio link changed: " + name)
    actual = {str(p.relative_to(output)) for p in output.rglob("*")
              if p.is_file() and p.name != "preparation.json"}
    if actual != set(report["output_hashes"]):
        raise ValueError("WHAMR dump inventory changed")
    for name, expected in report["output_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError("WHAMR index changed: " + name)
    for split, examples in report["baseline_audio"].items():
        folder = output / "raw" / f"{split}_{SUFFIX}"
        maps = {name: read_scp(folder / name) for name in (*AUDIO_NAMES, "rir_ctf.scp")}
        source_ctf = read_scp(source / "raw" / f"{split}_mix_clean_reverb_min_8k/rir_ctf.scp")
        lengths = read_scp(folder / "utt2num_samples")
        shapes = read_scp(folder / "rir_ctf_shape")
        if (set(examples) != set(native["baseline_audio"][split])
                or any(set(values) != set(examples) for values in (*maps.values(), lengths, shapes))):
            raise ValueError("WHAMR/NF utterance identities changed")
        def check(item):
            uid, files = item
            if (Path(maps["rir_ctf.scp"][uid]).resolve() != Path(source_ctf[uid]).resolve()
                    or shapes[uid] != "60" or lengths[uid] != str(files["wav.scp"]["frames"])):
                raise ValueError("WHAMR Oracle feature/shape mapping changed: " + uid)
            for name in AUDIO_NAMES:
                if digest(maps[name][uid]) != files[name]["sha256"]:
                    raise ValueError("Baseline audio changed: " + uid + "/" + name)
                if name != "wav.scp" and files[name]["sha256"] != native["baseline_audio"][split][uid][name]["sha256"]:
                    raise ValueError("Oracle speaker mapping changed: " + uid)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(check, examples.items()))
    return report


def prepare(source_dump, baseline_recipe, baseline_config, baseline_stats,
            config, output, workers=4, runtime_roots=None):
    source_dump, baseline_recipe, baseline_config, baseline_stats, config, output = [
        Path(p).resolve() for p in (source_dump, baseline_recipe, baseline_config,
                                   baseline_stats, config, output)]
    compare_config(config, baseline_config)
    roots = runtime_roots or (Path(__file__).resolve().parents[4], baseline_recipe.parents[2])
    shared_hashes = {}
    for name in SHARED_RUNTIME:
        current, saved = [digest(root / name) for root in roots]
        if current != saved:
            raise ValueError("Baseline shared runtime differs: " + name)
        shared_hashes[name] = current
    source = validate_prepared_dump(source_dump, workers=workers)
    spec = dict(condition="whamr", source_dump=str(source_dump),
                source_manifest_sha256=digest(source_dump / "preparation.json"),
                baseline_recipe=str(baseline_recipe), baseline_config_sha256=digest(baseline_config),
                config_sha256=digest(config), method=source["spec"]["method"],
                config=source["spec"]["config"], shape=source["spec"]["shape"])
    if output.exists():
        previous = validate(output, workers=workers)
        if previous["spec"] != spec:
            raise ValueError("WHAMR preparation source/config changed; use a new dump")
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".whamr_oracle_", dir=output.parent))
    report = dict(schema_version="whamr_oracle_v1", status="complete", spec=spec,
                  counts=source["counts"], baseline_audio={}, correspondence={}, links={},
                  source_index_hashes={}, baseline_shape_hashes={}, shared_runtime_hashes=shared_hashes)
    try:
        (temp / "features").symlink_to(source_dump / "features", target_is_directory=True)
        report["links"]["features"] = str(source_dump / "features")
        for split, expected in COUNTS.items():
            folder = baseline_recipe / "dump/raw" / f"{split}_{SUFFIX}"
            maps = {name: read_scp(folder / name) for name in AUDIO_NAMES}
            ids = sorted(maps["wav.scp"])
            if len(ids) != expected or any(set(m) != set(source["baseline_audio"][split]) for m in maps.values()):
                raise ValueError("WHAMR/NF utterance IDs differ: " + split)
            for name in AUDIO_NAMES:
                report["source_index_hashes"][str(folder / name)] = digest(folder / name)
            paths = {name: {uid: resolve(value, baseline_recipe) for uid, value in values.items()}
                     for name, values in maps.items()}
            clean = read_scp(source_dump / "raw" / f"{split}_mix_clean_reverb_min_8k/wav.scp")
            noise_path = folder / "noise1.scp"
            noise = read_scp(noise_path) if noise_path.exists() else None
            if noise is None and split != "tt":
                raise ValueError("Missing training/validation noise tracks")
            if noise is not None and set(noise) != set(ids):
                raise ValueError("Noise IDs differ")
            def inspect(uid):
                return verify_example(uid, {name: values[uid] for name, values in paths.items()},
                                      Path(clean[uid]), source["baseline_audio"][split][uid],
                                      resolve(noise[uid], baseline_recipe) if noise is not None else None)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                inspected = list(pool.map(inspect, ids))
            examples = dict(zip(ids, (value[0] for value in inspected)))
            report["baseline_audio"][split] = examples
            report["correspondence"][split] = dict(zip(ids, (value[1] for value in inspected)))
            lengths = {uid: str(examples[uid]["wav.scp"]["frames"]) for uid in ids}
            if split != "tt":
                phase = "train" if split == "tr" else "valid"
                for name in ("speech_mix", "speech_ref1", "speech_ref2"):
                    shape_path = baseline_stats / phase / (name + "_shape")
                    shapes = {uid: shape.split(",")[0] for uid, shape in read_scp(shape_path).items()}
                    if shapes != lengths:
                        raise ValueError("Baseline batch sampler lengths differ: " + name)
                    report["baseline_shape_hashes"][str(shape_path)] = digest(shape_path)
            dest = temp / "raw" / f"{split}_{SUFFIX}"
            dest.mkdir(parents=True)
            audio_root = Path(os.path.commonpath([str(p.parent) for values in paths.values() for p in values.values()]))
            if audio_root == Path(audio_root.anchor):
                raise ValueError("Missing shared audio root")
            (dest / "audio").symlink_to(audio_root, target_is_directory=True)
            report["links"][str(dest.relative_to(temp) / "audio")] = str(audio_root)
            for name, values in paths.items():
                (dest / name).write_text("".join(f"{uid} {output / dest.relative_to(temp) / 'audio' / path.relative_to(audio_root)}\n" for uid, path in values.items()))
            (dest / "utt2num_samples").write_text("".join(f"{uid} {lengths[uid]}\n" for uid in ids))
            (dest / "rir_ctf.scp").write_text("".join(f"{uid} {output / 'features' / split / (uid + '.npy')}\n" for uid in ids))
            (dest / "rir_ctf_shape").write_text("".join(f"{uid} 60\n" for uid in ids))
            print(f"Verified WHAMR/Oracle correspondence: {split}={len(ids)}", flush=True)
        report["output_hashes"] = {str(p.relative_to(temp)): digest(p) for p in temp.rglob("*") if p.is_file()}
        (temp / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        temp.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temp)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dump", "baseline-recipe", "baseline-config", "baseline-stats", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(**vars(args))
