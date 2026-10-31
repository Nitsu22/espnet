"""Verify the exact existing NF baseline inputs and the pinned Rec-RIR source."""
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_readiness(config_path=Path("conf/tuning/train_enh_tflocoformer_s_nf8k_recrir_ctf_film_all.yaml")):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    baseline = yaml.safe_load(Path("../enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_8k.yaml").read_text())
    for key in baseline.keys() - {"separator", "separator_conf", "model_conf"}:
        if baseline[key] != config[key]:
            raise RuntimeError("Baseline training setting differs: " + key)
    separator_conf = dict(config["separator_conf"])
    if config["separator"] == "tflocoformer_recrir_ctf_concat":
        if separator_conf.pop("glue_frames", None) != 4 or separator_conf.pop("glue_value", None) != 0.0:
            raise RuntimeError("Match DAI-117: four zero Glue tokens after each CTF")
    if separator_conf != baseline["separator_conf"]:
        raise RuntimeError("Baseline separator trunk configuration differs")
    source = config["model_conf"]["rec_rir_checkpoint"]
    expected = config["model_conf"]["rec_rir_sha256"]
    if digest(source) != expected:
        raise RuntimeError("Rec-RIR source checkpoint changed")
    inputs = Path("exp/bimamba_nf8k_baseline_inputs")
    manifest = inputs / "preparation.json"
    prepared = json.loads(manifest.read_text())
    if prepared["counts"] != {"tr": 20000, "cv": 5000, "tt": 3000}:
        raise RuntimeError("Incomplete baseline input preparation")
    for name, sha in prepared["output_hashes"].items():
        if digest(inputs / name) != sha:
            raise RuntimeError("Prepared baseline index changed: " + name)
    for name, sha in prepared["spec"]["source_hashes"].items():
        split, field = name.split("/")
        path = Path(prepared["spec"]["source"]) / "raw" / f"{split}_mix_clean_reverb_min_8k" / field
        if digest(path) != sha:
            raise RuntimeError("Original baseline index changed: " + name)
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        config_sha256=digest(config_path), manifest_sha256=digest(manifest),
        source_sha256=expected,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("conf/tuning/train_enh_tflocoformer_s_nf8k_recrir_ctf_film_all.yaml"))
    print(json.dumps(verify_readiness(parser.parse_args().config), indent=2))
