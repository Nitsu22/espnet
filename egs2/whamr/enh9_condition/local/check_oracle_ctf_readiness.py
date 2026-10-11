"""Reuse CUDA validation only while its model, configuration and data are unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


CRITICAL_PATHS = [
    "espnet", "espnet2", "egs2/whamr/enh9_condition/conf",
    *["egs2/whamr/enh9_condition/" + name for name in (
        "run.sh", "path.sh", "cmd.sh",
        "local/prepare_oracle_ctf_dump.py", "local/check_oracle_ctf_inputs.py",
        "local/check_oracle_ctf_training.py",
    )],
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_readiness(root=Path(".")):
    root = Path(root).resolve()
    report = json.loads((root / "exp/oracle_ctf_cuda_check.json").read_text())
    if (report["status"] != "passed" or report["device"] != "cuda"
            or report["counts"] != {"tr": 20000, "cv": 5000, "tt": 3000}):
        raise RuntimeError("Complete native CUDA validation is required")
    for key, path in (
        ("config_sha256", "conf/tuning/train_enh_tflocoformer_s_nf8k_oracle_ctf_concat.yaml"),
        ("manifest_sha256", "dump_oracle_ctf_nf8k/preparation.json"),
    ):
        if report[key] != digest(root / path):
            raise RuntimeError("CUDA validation input changed: " + path)
    subprocess.run(["git", "merge-base", "--is-ancestor", report["commit"], "HEAD"],
                   cwd=root, check=True)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", report["commit"], "--",
         *[":(top)" + path for path in CRITICAL_PATHS]],
        cwd=root, text=True,
    ).strip()
    if changed:
        raise RuntimeError("Validated runtime source changed:\n" + changed)
    # Stage 8 is CPU scoring, outside the CUDA model/training/inference probe.
    # Preserve every command before that block while allowing scoring fixes.
    name = "egs2/whamr/enh9_condition/run_oracle_ctf_concat.sh"
    original = subprocess.check_output(
        ["git", "show", report["commit"] + ":" + name], cwd=root, text=True)
    current = (root / "run_oracle_ctf_concat.sh").read_text()
    stage8 = "    if (( stage <= 8 && stop_stage >= 8 )); then"
    if original.split(stage8, 1)[0] != current.split(stage8, 1)[0]:
        raise RuntimeError("Validated training/inference commands changed")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    p = verify_readiness()
    print("Verified unchanged CUDA-tested runtime, configuration and dump: " + p["commit"])
