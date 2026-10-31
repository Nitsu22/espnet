"""Stage 6 checkpoint resume, Stage 7 inference and Stage 8 CPU scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from check_oracle_ctf_readiness import digest, verify_readiness
from oracle_condition import settings

EXP = Path("exp/enh_train_tflocoformer_s_nf8k_oracle_ctf_concat_seed0")
STATE = EXP / ".pipeline"
VARIANT = "concat"
CONDITION = "nf_whamr"
DUMP = Path("dump_oracle_ctf_nf8k")
SUFFIX = "mix_clean_reverb_min_8k"
RESOURCES = {
    "control": ("cpu_4", "00:10:00", 0.015),
    "train": ("gpu_1", "24:00:00", 0.2),
    "infer": ("gpu_1", "04:00:00", 0.2),
    "score": ("cpu_8", "06:00:00", 0.03),
}
LIMITS = {"train": 12, "infer": 2, "score": 3}
METRICS = ("SI_SNR", "SDR", "SIR", "SAR", "STOI", "ESTOI")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def read_scp(path):
    pairs = [line.split(maxsplit=1) for line in path.read_text().splitlines()]
    values = dict(pairs)
    if len(values) != len(pairs):
        raise RuntimeError("Duplicate IDs: " + str(path))
    return values


def training_state():
    if not (EXP / "checkpoint.pth").exists():
        return 0, False
    import torch
    import yaml
    from espnet2.train.reporter import Reporter
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    checkpoint = torch.load(EXP / "checkpoint.pth", map_location="cpu", weights_only=False)
    reporter = Reporter()
    reporter.load_state_dict(checkpoint["reporter"])
    epoch = reporter.get_epoch()
    if "The gradients at all steps are invalid" in (EXP / "train.log").read_text():
        raise RuntimeError("Invalid-gradient training exit")
    finished = epoch >= config["max_epoch"] or (
        config.get("patience") is not None and reporter.check_early_stopping(
            config["patience"], *config["early_stopping_criterion"]))
    if finished:
        best = reporter.get_best_epoch("valid", "loss", "min")
        link = EXP / "valid.loss.best.pth"
        if not link.is_file() or link.resolve().name != f"{best}epoch.pth":
            raise RuntimeError("Best-model artifact differs from checkpoint reporter")
    return epoch, finished


def accounting(job):
    for _ in range(6):
        reply = subprocess.run(["qacct", "-j", str(job)], capture_output=True,
                               text=True, timeout=30)
        failed = re.search(r"^failed\s+(\d+)", reply.stdout, re.M)
        status = re.search(r"^exit_status\s+(\d+)", reply.stdout, re.M)
        if failed and status:
            return int(failed.group(1)), int(status.group(1))
        time.sleep(10)
    raise RuntimeError("Accounting unavailable: " + str(job))


def next_phase(jobs, previous, failed, status, epoch, trained, inference_done, score_done):
    if previous:
        if (failed, status) not in ((0, 0), (44, 137)):
            raise RuntimeError(f"Non-timeout failure: {failed}/{status}")
        phase = previous["phase"]
        complete = {"train": trained, "infer": inference_done, "score": score_done}[phase]
        if status == 0 and not complete:
            raise RuntimeError(f"Successful {phase} exit without completed stage")
        if phase == "train" and not trained and epoch <= previous["epoch"]:
            raise RuntimeError("Timeout without checkpoint progress")
    if score_done:
        return None
    phase = "score" if inference_done else "infer" if trained else "train"
    if sum(j["phase"] == phase for j in jobs) >= LIMITS[phase]:
        raise RuntimeError("Automatic segment limit reached: " + phase)
    return phase


def submit(phase, dependency=None, previous=None):
    resource, duration, coefficient = RESOURCES[phase]
    args = ["qsub", "-terse", "-g", "tga-shinoda", "-p", "-5",
            "-l", resource + "=1", "-l", "h_rt=" + duration,
            "-N", "oracle_" + phase]
    if dependency:
        args += ["-hold_jid", str(dependency)]
    args += ["qsub/oracle_ctf_pipeline.sh", "--phase", phase]
    args += ["--variant", VARIANT, "--condition", CONDITION]
    if previous:
        args += ["--previous-job", str(previous)]
    reply = subprocess.check_output(args, text=True).strip()
    job = reply.splitlines()[-1] if reply else ""
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError("Unexpected qsub response: " + reply)
    hours = sum(int(v) * w for v, w in zip(duration.split(":"), (1, 1/60, 1/3600)))
    entry = dict(job=job, phase=phase, resource=resource, units=1,
                 priority=-5, h_rt=duration, upper_points=round(coefficient * 0.8 * hours, 6))
    print(json.dumps(dict(submitted=entry)), flush=True)
    return entry


def control(previous_job=None):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = STATE / "jobs.json"
        jobs = json.loads(journal.read_text()) if journal.exists() else []
        previous = None
        failed = status = None
        if previous_job:
            previous = next(j for j in jobs if j["job"] == previous_job)
            if previous.get("handled"):
                print("Completed job already handled", flush=True)
                return
            failed, status = accounting(previous_job)
        elif jobs:
            raise RuntimeError("Pipeline already started; use the held controller")
        epoch, trained = training_state()
        phase = next_phase(jobs, previous, failed, status, epoch, trained,
                           (STATE / "inference.done.json").exists(),
                           (STATE / "score.done.json").exists())
        if previous:
            previous.update(handled=True, failed=failed, exit_status=status)
        if phase is None:
            save(journal, jobs)
            print("Stage 8 completed", flush=True)
            return
        job = submit(phase, dependency=os.environ.get("JOB_ID"))
        job["epoch"] = epoch
        jobs.append(job)
        save(journal, jobs)
        job["controller"] = submit("control", dependency=job["job"], previous=job["job"])
        save(journal, jobs)


def run_stage(stage):
    subprocess.run(["bash", "run_oracle_ctf_concat.sh", "--stage", str(stage),
                    "--stop_stage", str(stage), "--ngpu", "1",
                    "--python", sys.executable, "--enh_exp", str(EXP),
                    "--variant", VARIANT, "--condition", CONDITION], check=True)


def train():
    epoch, finished = training_state()
    if not finished:
        log = EXP / "train.log"
        if log.exists():
            shutil.copy2(log, STATE / ("train_before_" + os.environ["JOB_ID"] + ".log"))
        subprocess.run(["bash", "qsub/train_oracle_ctf_concat_1gpu.sh",
                        "true" if (EXP / "checkpoint.pth").exists() else "false",
                        VARIANT, CONDITION], check=True)
        epoch, finished = training_state()
    if not finished:
        raise RuntimeError("Training exited without reaching completion criterion")
    save(STATE / "training.done.json", dict(epoch=epoch))


def infer():
    epoch, finished = training_state()
    if not finished:
        raise RuntimeError("Stage 6 completion required")
    run_stage(7)
    for split, count in (("cv", 5000), ("tt", 3000)):
        expected = read_scp(DUMP / "raw" / f"{split}_{SUFFIX}" / "wav.scp")
        if len(expected) != count:
            raise RuntimeError("Unexpected inference input count")
        for speaker in (1, 2):
            values = read_scp(EXP / f"inference_{split}/spk{speaker}.scp")
            if set(values) != set(expected) or any(not Path(p).is_file() for p in values.values()):
                raise RuntimeError("Incomplete inference output: " + split)
    save(STATE / "inference.done.json", dict(epoch=epoch, model="valid.loss.best.pth",
         model_sha256=digest(EXP / "valid.loss.best.pth")))


def metric_values(folder, ids):
    result = {}
    for metric in METRICS:
        for speaker in (1, 2):
            name = f"{metric}_spk{speaker}"
            values = read_scp(folder / name)
            if set(values) != set(ids):
                raise RuntimeError("Incomplete metric: " + str(folder / name))
            numbers = {uid: float(v) for uid, v in values.items()}
            if metric not in ("SIR", "SAR") and any(not math.isfinite(v) for v in numbers.values()):
                raise RuntimeError("Nonfinite metric: " + name)
            result[name] = numbers
    return result


def score():
    marker = STATE / "inference.done.json"
    if not marker.exists():
        raise RuntimeError("Stage 7 completion required")
    if json.loads(marker.read_text())["model_sha256"] != digest(EXP / "valid.loss.best.pth"):
        raise RuntimeError("Inference model changed")
    subprocess.run([sys.executable, "local/check_oracle_ctf_inputs.py",
                    "--dump", str(DUMP), "--condition", CONDITION], check=True)
    summary = {}
    for split in ("cv", "tt"):
        folder = DUMP / "raw" / f"{split}_{SUFFIX}"
        ids = list(read_scp(folder / "wav.scp"))
        work = STATE / "score_work" / split
        work.mkdir(parents=True, exist_ok=True)
        def shard(index):
            keys = ids[index::8]
            destination = work / str(index)
            try:
                return metric_values(destination, keys)
            except (OSError, ValueError, RuntimeError):
                pass
            key_file = work / f"keys.{index}"
            key_file.write_text("\n".join(keys) + "\n")
            command = [sys.executable, "-m", "espnet2.bin.enh_scoring",
                       "--key_file", str(key_file), "--ref_channel", "0",
                       "--ref_scp", str(folder / "spk1.scp"),
                       "--ref_scp", str(folder / "spk2.scp"),
                       "--inf_scp", str(EXP / f"inference_{split}/spk1.scp"),
                       "--inf_scp", str(EXP / f"inference_{split}/spk2.scp"),
                       "--output_dir", str(destination)]
            with (work / f"score.{index}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            return metric_values(destination, keys)
        with ThreadPoolExecutor(max_workers=8) as pool:
            pieces = list(pool.map(shard, range(8)))
        merged = {name: {} for name in pieces[0]}
        for piece in pieces:
            for name, values in piece.items():
                if merged[name].keys() & values.keys():
                    raise RuntimeError("Overlapping scoring shards")
                merged[name].update(values)
        output = EXP / f"inference_{split}/score"
        output.mkdir(parents=True, exist_ok=True)
        for name, values in merged.items():
            if set(values) != set(ids):
                raise RuntimeError("Incomplete merged scoring")
            (output / name).write_text("".join(f"{uid} {values[uid]}\n" for uid in ids))
        for speaker in (1, 2):
            waves = {}
            for index in range(8):
                values = read_scp(work / str(index) / f"wav_spk{speaker}")
                if set(values) != set(ids[index::8]):
                    raise RuntimeError("Incomplete scoring permutation map")
                waves.update(values)
            (output / f"wav_spk{speaker}").write_text(
                "".join(f"{uid} {waves[uid]}\n" for uid in ids))
        summary[split] = {metric: sum(sum(merged[f"{metric}_spk{s}"].values()) for s in (1, 2))
                          / (2 * len(ids)) for metric in METRICS}
    lines = ["# Oracle CTF " + ("WHAMR!" if CONDITION == "whamr" else "NF-WHAMR") + " 8 kHz", "", "Model: valid.loss.best.pth", "",
             "| Split | " + " | ".join(METRICS) + " |",
             "| --- | " + " | ".join(["---"] * len(METRICS)) + " |"]
    for split, values in summary.items():
        lines.append("| " + split + " | " + " | ".join(f"{values[m]:.4f}" for m in METRICS) + " |")
    (EXP / "RESULTS.md").write_text("\n".join(lines) + "\n")
    save(STATE / "score.done.json", dict(results=str(EXP / "RESULTS.md"), metrics=summary))


def main():
    global EXP, STATE, VARIANT, CONDITION, DUMP, SUFFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(RESOURCES), required=True)
    parser.add_argument("--previous-job")
    parser.add_argument("--variant", choices=("concat", "film", "concat_glue"), default="concat")
    parser.add_argument("--condition", choices=("nf_whamr", "whamr"), default="nf_whamr")
    parser.add_argument("--exp", type=Path)
    args = parser.parse_args()
    VARIANT = args.variant
    CONDITION = args.condition
    paths = settings(CONDITION, VARIANT)
    if CONDITION == "whamr":
        RESOURCES.update(infer=("gpu_1", "01:00:00", 0.2),
                         score=("cpu_8", "01:00:00", 0.03))
    DUMP, SUFFIX = paths["dump"], paths["suffix"]
    default_exp = paths["exp"]
    if args.exp is not None and args.exp != default_exp and args.phase not in ("infer", "score"):
        parser.error("A separate --exp is supported for inference/scoring only")
    EXP = args.exp or default_exp
    STATE = EXP / ".pipeline"
    report = verify_readiness(variant=VARIANT, condition=CONDITION)
    STATE.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    pin = STATE / "run.json"
    if pin.exists():
        if json.loads(pin.read_text())["commit"] != commit:
            raise RuntimeError("Pipeline checkout commit changed")
    else:
        save(pin, dict(commit=commit, validated_commit=report["commit"],
                       config_sha256=report["config_sha256"],
                       manifest_sha256=report["manifest_sha256"]))
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ":(top)espnet",
                    ":(top)espnet2", ":(top)egs2/whamr/enh9_condition"], check=True)
    if args.phase == "control":
        control(args.previous_job)
    else:
        {"train": train, "infer": infer, "score": score}[args.phase]()


if __name__ == "__main__":
    main()
