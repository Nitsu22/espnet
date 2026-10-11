"""Event-driven checkpoint resume -> GPU inference -> CPU scoring pipeline."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time


def experiment(condition):
    tag, rate = {"whamr8": ("whamr", "8k"), "nf8": ("nf", "8k"),
                 "nf16": ("nf", "16k")}[condition]
    return Path("exp/enh_train_tflocoformer_s_{}_{}_1gpu_batch4".format(tag, rate))


def training_state(exp):
    import torch
    import yaml
    from espnet2.train.reporter import Reporter

    config = yaml.safe_load((exp / "config.yaml").read_text())
    checkpoint = torch.load(exp / "checkpoint.pth", map_location="cpu", weights_only=False)
    reporter = Reporter()
    reporter.load_state_dict(checkpoint["reporter"])
    epoch = reporter.get_epoch()
    # Do not accept an invalid-gradient exit or an h_rt kill as completion.
    finished = epoch >= config["max_epoch"] or (
        config.get("patience") is not None and reporter.check_early_stopping(
            config["patience"], *config["early_stopping_criterion"]))
    log = exp / "train.log"
    if log.exists() and "The gradients at all steps are invalid" in log.read_text():
        raise RuntimeError("Invalid-gradient training exit; do not infer or retry")
    if finished:
        best = reporter.get_best_epoch("valid", "loss", "min")
        link = exp / "valid.loss.best.pth"
        if not link.is_file() or link.resolve().name != "{}epoch.pth".format(best):
            raise RuntimeError("Best-model artifact differs from the saved reporter")
    return epoch, finished


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def accounting(job):
    for _ in range(6):
        result = subprocess.run(["qacct", "-j", str(job)], capture_output=True, text=True,
                                timeout=30)
        match = re.search(r"^exit_status\s+(\d+)", result.stdout, re.M)
        failed = re.search(r"^failed\s+(\d+)", result.stdout, re.M)
        if match and failed:
            return int(failed.group(1)), int(match.group(1))
        time.sleep(10)
    raise RuntimeError("Accounting unavailable; do not assume success or resubmit: " + str(job))


def submit(condition, phase, dependency=None, previous=None, previous_phase=None,
           inference_only=False):
    resource, hours = {"control": ("cpu_4", "00:10:00"),
                       "gpu": ("gpu_1", "02:00:00" if inference_only else "24:00:00"),
                       "score": ("cpu_8", "06:00:00")}[phase]
    args = ["qsub", "-terse", "-g", "tga-shinoda", "-l", resource + "=1",
            "-l", "h_rt=" + hours, "-N", "tfs_{}_{}".format(condition, phase)]
    if dependency:
        args += ["-hold_jid", str(dependency)]
    args += ["qsub/baseline_pipeline.sh", "--condition", condition, "--phase", phase]
    if previous:
        args += ["--previous-job", str(previous), "--previous-phase", previous_phase]
    reply = subprocess.check_output(args, text=True).strip()
    job = reply.splitlines()[-1] if reply else ""
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError("Unexpected qsub response: " + job)
    print("Submitted {} {}: {}".format(condition, phase, job), flush=True)
    return job


def control(condition, previous=None, previous_phase=None):
    exp = experiment(condition)
    folder = exp / ".pipeline"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = folder / "jobs.json"
        jobs = json.loads(journal.read_text()) if journal.exists() else []
        if previous:
            failed, status = accounting(previous)
            if (failed, status) not in ((0, 0), (44, 137)):
                raise RuntimeError("Non-timeout failure: {} failed={} exit={}".format(previous, failed, status))
            prior = next(j for j in jobs if j["job"] == str(previous))
            if prior.get("handled"):
                print("This completed job was already handled", flush=True)
                return
            prior.update(handled=True, failed=failed, exit_status=status)
        elif jobs:
            raise RuntimeError("Pipeline already submitted; continue via its held controller")
        if (folder / "score.done.json").exists():
            save(journal, jobs)
            print("Stage 8 completed", flush=True)
            return
        epoch, trained = training_state(exp)
        if previous_phase == "gpu" and status == 0 and not trained:
            raise RuntimeError("Successful GPU exit without completed training")
        if previous and failed == 44 and previous_phase == "gpu" and not trained:
            if epoch <= prior["epoch"]:
                raise RuntimeError("Timeout without checkpoint progress; stopping automatic retries")
        if sum(j["phase"] == "gpu" for j in jobs) >= 12:
            raise RuntimeError("Unexpectedly many GPU segments; stopping automatic retries")
        inference_done = (folder / "inference.done.json").exists()
        if previous_phase == "score" and not (folder / "score.done.json").exists():
            if status == 0:
                raise RuntimeError("Scoring exited successfully without completion marker")
        phase = "score" if trained and inference_done else "gpu"
        if phase == "score" and sum(j["phase"] == "score" for j in jobs) >= 3:
            raise RuntimeError("Repeated scoring timeouts; stopping automatic retries")
        job = submit(condition, phase, dependency=os.environ.get("JOB_ID"),
                     inference_only=trained)
        jobs.append(dict(job=job, phase=phase, epoch=epoch))
        save(journal, jobs)
        follow = submit(condition, "control", dependency=job, previous=job,
                        previous_phase=phase)
        jobs[-1]["controller"] = follow
        save(journal, jobs)


def run_stage(condition, stage):
    exp = str(experiment(condition))
    args = ["bash", "run_nf16k_tflocoformer_s.sh" if condition == "nf16"
            else "run_tflocoformer_s.sh"]
    if condition != "nf16":
        args += ["--condition", "whamr" if condition == "whamr8" else "nf_whamr"]
    args += ["--stage", str(stage), "--stop_stage", str(stage), "--enh_exp", exp,
             "--ngpu", "0" if stage == 8 else "1", "--gpu_inference",
             "false" if stage == 8 else "true", "--inference_nj", "8" if stage == 8 else "1"]
    subprocess.run(args, check=True)


def gpu(condition):
    exp = experiment(condition)
    folder = exp / ".pipeline"
    epoch, trained = training_state(exp)
    if not trained:
        if condition == "nf16":
            args = ["bash", "qsub/train_nf16k_tflocoformer_s_1gpu.sh"]
        else:
            args = ["bash", "qsub/train_tflocoformer_s_1gpu.sh",
                    "whamr" if condition == "whamr8" else "nf_whamr"]
        subprocess.run(args, check=True)
        epoch, trained = training_state(exp)
    if not trained:
        raise RuntimeError("Trainer exited without reaching its completion criterion")
    save(folder / "training.done.json", dict(epoch=epoch))
    if not (exp / "valid.loss.best.pth").exists():
        raise RuntimeError("Best validation-loss model missing")
    if not (folder / "inference.done.json").exists():
        run_stage(condition, 7)
        subset = "mix_both_reverb_min_8k" if condition == "whamr8" else (
            "mix_clean_reverb_min_16k" if condition == "nf16" else "mix_clean_reverb_min_8k")
        for split, count in (("cv", 5000), ("tt", 3000)):
            for speaker in (1, 2):
                paths = list(exp.glob("*_{}_{}/spk{}.scp".format(split, subset, speaker)))
                if len(paths) != 1 or len(paths[0].read_text().splitlines()) != count:
                    raise RuntimeError("Incomplete inference SCPs")
        save(folder / "inference.done.json", dict(epoch=epoch, model="valid.loss.best.pth"))


def score(condition):
    exp = experiment(condition)
    if not (exp / ".pipeline/inference.done.json").exists():
        raise RuntimeError("Successful Stage 7 required")
    run_stage(condition, 8)
    if not (exp / "RESULTS.md").is_file():
        raise RuntimeError("Stage 8 summary missing")
    save(exp / ".pipeline/score.done.json", dict(results=str(exp / "RESULTS.md")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("whamr8", "nf8", "nf16"), required=True)
    parser.add_argument("--phase", choices=("control", "gpu", "score"), required=True)
    parser.add_argument("--previous-job")
    parser.add_argument("--previous-phase", choices=("gpu", "score"))
    args = parser.parse_args()
    if args.phase == "control":
        control(args.condition, args.previous_job, args.previous_phase)
    elif args.phase == "gpu":
        gpu(args.condition)
    else:
        score(args.condition)


if __name__ == "__main__":
    main()
