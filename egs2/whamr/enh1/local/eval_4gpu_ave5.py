"""Evaluate the saved WHAMR S/M four-GPU runs with their five-best average."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


TAGS = {
    "small": "enh_train_enh_tflocoformer_nocashe_small_4gpu",
    "medium": "enh_train_enh_tflocoformer_nocashe_repro_4gpu",
}
SOURCE_ROOT = Path("/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/enh1")
MODEL = "valid.loss.ave_5best.pth"
SETS = {"cv": 5000, "tt": 3000}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_scp(path):
    entries = {}
    for row in Path(path).read_text().splitlines():
        key, value = row.split(maxsplit=1)
        if key in entries:
            raise ValueError("Duplicate key in " + str(path))
        entries[key] = value
    return entries


def dataset(split):
    return split + "_mix_both_reverb_min_8k"


def validate_data():
    if not Path("dump/raw/RESULTS.md").is_file():
        raise RuntimeError("Require existing observation scores; do not modify source dump")
    hashes = {}
    for split, count in SETS.items():
        folder = Path("dump/raw") / dataset(split)
        keys = set(read_scp(folder / "wav.scp"))
        if len(keys) != count:
            raise ValueError("Unexpected subset size: " + split)
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            path = folder / name
            if set(read_scp(path)) != keys:
                raise ValueError("Mixture/reference IDs differ: " + str(path))
            hashes[str(path)] = sha256(path)
        # Require complete saved input scores before launching either stage.
        for protocol in ("SI_SNR", "SDR"):
            for speaker in (1, 2):
                path = folder / "scoring" / (protocol + "_spk" + str(speaker))
                if set(read_scp(path)) != keys:
                    raise ValueError("Incomplete observation scores: " + str(path))
    return hashes


def verify_average(source):
    import torch
    from espnet2.train.reporter import Reporter

    checkpoint = torch.load(source / "checkpoint.pth", map_location="cpu")
    reporter = Reporter()
    reporter.load_state_dict(checkpoint["reporter"])
    epochs = reporter.sort_epochs("valid", "loss", "min")[:5]
    if len(epochs) != 5:
        raise RuntimeError("Five validation checkpoints required")
    del checkpoint
    expected = None
    for epoch in epochs:
        state = torch.load(source / (str(epoch) + "epoch.pth"), map_location="cpu")
        if expected is None:
            expected = {key: value.clone() for key, value in state.items()}
        else:
            if expected.keys() != state.keys():
                raise RuntimeError("Checkpoint keys differ")
            for key in expected:
                expected[key] = expected[key] + state[key]
    average = torch.load(source / MODEL, map_location="cpu")
    if average.keys() != expected.keys():
        raise RuntimeError("Averaged checkpoint keys differ")
    for key, value in expected.items():
        if not str(value.dtype).startswith("torch.int"):
            value = value / 5
        if not torch.equal(value, average[key]):
            raise RuntimeError("Saved checkpoint is not the exact five-best average: " + key)
    return epochs


def stage_command(output, stage):
    return [
        "bash", "enh.sh", "--stage", str(stage), "--stop_stage", str(stage),
        "--skip_data_prep", "true", "--skip_packing", "true",
        "--train_set", dataset("tr"), "--valid_set", dataset("cv"),
        "--test_sets", dataset("tt"), "--fs", "8k", "--ref_num", "2",
        "--enh_config", str(output / "config.yaml"), "--enh_exp", str(output),
        "--dumpdir", "dump", "--use_dereverb_ref", "false", "--use_noise_ref", "false",
        "--inference_model", MODEL, "--inference_tag", "enhanced_ave5",
        "--audio_format", "wav", "--ngpu", "1" if stage == 7 else "0",
        "--gpu_inference", "true" if stage == 7 else "false",
        "--inference_nj", "1" if stage == 7 else "8",
    ]


def verify_inference(output):
    for split in SETS:
        keys = set(read_scp(Path("dump/raw") / dataset(split) / "wav.scp"))
        for speaker in (1, 2):
            path = output / ("enhanced_ave5_" + dataset(split)) / ("spk" + str(speaker) + ".scp")
            entries = read_scp(path)
            if set(entries) != keys or any(not Path(wav).is_file() for wav in entries.values()):
                raise RuntimeError("Incomplete inference outputs: " + str(path))


def metric(folder, protocol, keys):
    speakers = []
    for speaker in (1, 2):
        values = {key: float(value) for key, value in read_scp(
            folder / (protocol + "_spk" + str(speaker))).items()}
        if set(values) != keys or not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Incomplete/nonfinite scores: " + str(folder))
        speakers.append(values)
    return sum((speakers[0][key] + speakers[1][key]) / 2 for key in sorted(keys)) / len(keys)


def summarize(source, output):
    comparison = {}
    rows = ["# Five-best versus single-best WHAMR evaluation", "",
            "Scores in dB. Improvements use the same stored observation scores.", "",
            "|Subset|Metric|Single best|Five-best average|Change|", "|---|---|---:|---:|---:|"]
    for split in SETS:
        subset = dataset(split)
        keys = set(read_scp(Path("dump/raw") / subset / "wav.scp"))
        values = {}
        for protocol in ("SI_SNR", "SDR"):
            observation = metric(Path("dump/raw") / subset / "scoring", protocol, keys)
            best = metric(source / ("enhanced_" + subset) / "scoring", protocol, keys)
            averaged = metric(output / ("enhanced_ave5_" + subset) / "scoring", protocol, keys)
            values[protocol] = dict(count=len(keys), observation=observation, single_best=best,
                                    five_best=averaged, single_best_improvement=best - observation,
                                    five_best_improvement=averaged - observation, change=averaged - best)
            for label, old, new in ((protocol, best, averaged),
                                    (protocol + "i", best - observation, averaged - observation)):
                rows.append("|{}|{}|{:.4f}|{:.4f}|{:+.4f}|".format(split, label, old, new, new - old))
        comparison[split] = values
    write_json(output / "comparison.json", comparison)
    (output / "COMPARISON.md").write_text("\n".join(rows) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=TAGS, required=True)
    parser.add_argument("--phase", choices=("infer", "score"), required=True)
    args = parser.parse_args()
    source = SOURCE_ROOT / "exp" / TAGS[args.size]
    output = Path("exp") / (TAGS[args.size] + "_ave5_eval")
    marker = output / ".evaluation"
    data_hashes = validate_data()
    if args.phase == "infer":
        if output.exists():
            raise RuntimeError("Refuse to overwrite an existing evaluation: " + str(output))
        epochs = verify_average(source)
        output.mkdir(parents=True)
        for name in ("config.yaml", MODEL):
            shutil.copy2(source / name, output / name)
        metadata = dict(source=str(source), source_config_sha256=sha256(source / "config.yaml"),
                        model=MODEL, model_sha256=sha256(output / MODEL), best_epochs=epochs,
                        data_sha256=data_hashes,
                        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        write_json(marker / "provenance.json", metadata)
        subprocess.run(stage_command(output, 7), check=True)
        verify_inference(output)
        write_json(marker / "inference.done.json", metadata)
    else:
        # hold_jid only waits for exit. This marker also requires successful inference.
        metadata = json.loads((marker / "inference.done.json").read_text())
        if metadata["data_sha256"] != data_hashes or metadata["model_sha256"] != sha256(output / MODEL):
            raise RuntimeError("Inputs or averaged model changed after inference")
        verify_inference(output)
        subprocess.run(stage_command(output, 8), check=True)
        if not (output / "RESULTS.md").is_file():
            raise RuntimeError("Stage 8 summary missing")
        comparison = summarize(source, output)
        write_json(marker / "score.done.json", dict(results=str(output / "RESULTS.md"), comparison=comparison))
        print((output / "COMPARISON.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
