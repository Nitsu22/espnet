"""Reuse CUDA validation only while its model, configuration and data are unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from oracle_condition import settings


CRITICAL_PATHS = [
    "espnet", "espnet2", "egs2/whamr/enh9_condition/conf",
    *["egs2/whamr/enh9_condition/" + name for name in (
        "run.sh", "path.sh", "cmd.sh",
        "local/prepare_oracle_ctf_dump.py", "local/check_oracle_ctf_inputs.py",
        "local/check_oracle_ctf_training.py", "local/oracle_condition.py",
        "local/prepare_whamr_oracle_dump.py",
        "qsub/oracle_environment.sh", "qsub/train_oracle_ctf_concat_1gpu.sh",
    )],
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_readiness(root=Path("."), variant="concat", condition="nf_whamr"):
    if variant not in ("concat", "film"):
        raise ValueError("Unknown Oracle variant")
    root = Path(root).resolve()
    paths = settings(condition, variant)
    marker = paths["marker"]
    report = json.loads((root / "exp" / marker).read_text())
    if (report["status"] != "passed" or report["device"] != "cuda"
            or report["counts"] != {"tr": 20000, "cv": 5000, "tt": 3000}):
        raise RuntimeError("Complete native CUDA validation is required")
    if report.get("condition", "nf_whamr") != condition:
        raise RuntimeError("CUDA validation dataset differs")
    for key, path in (
        ("config_sha256", paths["config"]),
        ("manifest_sha256", paths["dump"] / "preparation.json"),
    ):
        if report[key] != digest(root / path):
            raise RuntimeError("CUDA validation input changed: " + str(path))
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
    # Preserve every command before that block except enabling the existing
    # peak-normalization option on Stage 7's inference call. This changes WAV
    # serialization, not the CUDA-tested model or its forward computation.
    name = "egs2/whamr/enh9_condition/run_oracle_ctf_concat.sh"
    original = subprocess.check_output(
        ["git", "show", report["commit"] + ":" + name], cwd=root, text=True)
    current = (root / "run_oracle_ctf_concat.sh").read_text()
    stage8 = "    if (( stage <= 8 && stop_stage >= 8 )); then"
    original_prefix = original.split(stage8, 1)[0]
    allowed_prefixes = {original_prefix}
    inference_command = (
        '        "${python}" -m espnet2.bin.enh_rir_ctf_inference \\\n'
        '            --ngpu "${ngpu}" --fs 8000 --batch_size 1 \\\n'
    )
    if original_prefix.count(inference_command) == 1:
        normalized_command = inference_command.replace(
            '--batch_size 1 \\\n',
            '--batch_size 1 --normalize_output_wav true \\\n',
        )
        allowed_prefixes.add(original_prefix.replace(
            inference_command, normalized_command, 1))
    if current.split(stage8, 1)[0] not in allowed_prefixes:
        raise RuntimeError("Validated training/inference commands changed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("concat", "film"), default="concat")
    parser.add_argument("--condition", choices=("nf_whamr", "whamr"), default="nf_whamr")
    args = parser.parse_args()
    p = verify_readiness(variant=args.variant, condition=args.condition)
    print("Verified unchanged CUDA-tested runtime, configuration and dump: " + p["commit"])
