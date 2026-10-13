#!/usr/bin/env python3
"""Run a TF-Locoformer DAMSEP ablation through GPU training/inference and CPU scoring.

Only a scheduler-confirmed walltime termination is continued automatically.
Every submission and result is recorded under the experiment's pipeline directory.
This controller uses only the Python standard library on the login node.
"""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

VARIANTS = {
    "clean_reverb": "clean_reverb",
    "three_loss": "three_loss",
}
GPU_SECONDS = 24 * 3600
MAX_GPU_CHUNKS = 25
METRICS = ("SI_SNR", "SDR", "SIR", "SAR", "STOI")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_command(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def commit():
    return run_command(["git", "rev-parse", "HEAD"], capture_output=True).stdout.strip()


def read_ids(path):
    ids = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"Empty or duplicate utterances: {path}")
    return set(ids)


def checkpoint_exists(path):
    return path.is_file() and path.stat().st_size > 0


def parse_accounting(output):
    record = {}
    for line in output.splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2 and fields[0] in (
            "jobnumber",
            "failed",
            "exit_status",
            "ru_wallclock",
            "hostname",
        ):
            record[fields[0]] = fields[1].split()[0]
    if not all(key in record for key in ("failed", "exit_status", "ru_wallclock")):
        raise RuntimeError("Scheduler accounting is incomplete")
    record["failed"] = int(record["failed"])
    record["exit_status"] = int(record["exit_status"])
    record["ru_wallclock"] = float(record["ru_wallclock"])
    return record


def is_walltime(record):
    # A killed process alone could be an OOM or another error. Require the
    # scheduler's known time-limit result and runtime close to the requested cap.
    return (
        record["failed"] == 44
        and record["exit_status"] in (137, 143)
        and GPU_SECONDS - 300 <= record["ru_wallclock"] <= GPU_SECONDS + 300
    )


class Pipeline:
    def __init__(self, variant, recipe=None):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.recipe = Path(recipe or Path.cwd()).resolve()
        self.config = (
            self.recipe
            / "conf/tuning"
            / f"train_tflocoformer_damsep_whamr_8k_{variant}.yaml"
        )
        self.exp = self.recipe / "exp" / f"tflocoformer_damsep_whamr_8k_{variant}"
        self.directory = self.exp / "pipeline"
        self.state_file = self.directory / "state.json"
        self.test_scp = self.recipe / "dump_whamr/raw/tt_mix_both_reverb_min_8k/wav.scp"

    @contextlib.contextmanager
    def locked_state(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(self.state_file.read_text())
                if self.state_file.exists()
                else {}
            )
            try:
                yield state
            finally:
                if state:
                    state["updated_at"] = now()
                    write_json(self.state_file, state)

    def marker(self, name):
        return self.directory / f"{name}_complete.json"

    def verify_source(self, state):
        if commit() != state["commit"]:
            raise RuntimeError(
                "Checkout commit changed while the pipeline was queued/running"
            )
        if (
            hashlib.sha256(self.config.read_bytes()).hexdigest()
            != state["config_sha256"]
        ):
            raise RuntimeError(
                "Experiment config changed while the pipeline was queued/running"
            )

    def save_phase(self, phase, **details):
        with self.locked_state() as state:
            self.verify_source(state)
            state.update(phase=phase, **details)

    def qsub(self, state, kind, hold=None, user_hold=False):
        scripts = {
            "gpu": ("gpu_1", "24:00:00", 3.8400),
            "control": ("cpu_4", "00:20:00", 0.0040),
            "score": ("cpu_8", "02:00:00", 0.0480),
        }
        resource, walltime, upper = scripts[kind]
        command = ["qsub", "-terse", "-g", state["group"]]
        if hold is not None:
            command += ["-hold_jid", str(hold)]
        if user_hold:
            command += ["-h"]
        command += [
            "-N",
            f"tfd_{kind}_{'cr' if self.variant == 'clean_reverb' else 'ctf'}",
            f"qsub/tflocoformer_damsep_{kind}.sh",
            "--variant",
            self.variant,
        ]
        if kind == "control":
            command += ["--gpu-job", str(hold)]
        result = run_command(command, cwd=self.recipe, capture_output=True)
        job = result.stdout.strip()
        if not re.fullmatch(r"[0-9]+", job):
            raise RuntimeError(
                f"Unexpected qsub response; do not resubmit blindly: {job!r}"
            )
        state["jobs"].append(
            {
                "id": job,
                "kind": kind,
                "submitted_at": now(),
                "command": command,
                "resource": resource,
                "units": 1,
                "priority": -5,
                "h_rt": walltime,
                "upper_points": upper,
                "commit": state["commit"],
                "hold_jid": str(hold) if hold is not None else None,
            }
        )
        # Save each actual ID before making any further scheduler mutation.
        write_json(self.state_file, state)
        print(
            f"Submitted {kind} job {job}; upper estimate {upper:.4f} points", flush=True
        )
        return job

    def enqueue_gpu(self, state):
        chunks = [job for job in state["jobs"] if job["kind"] == "gpu"]
        if len(chunks) >= MAX_GPU_CHUNKS:
            raise RuntimeError(
                "GPU continuation safety limit reached; inspect progress"
            )
        checkpoint = self.exp / "checkpoint.pth"
        state["checkpoint_before_job"] = (
            checkpoint.stat().st_mtime_ns if checkpoint_exists(checkpoint) else None
        )
        # Install the held controller before releasing GPU execution, so a
        # controller submission error cannot leave unmonitored paid training.
        job = self.qsub(state, "gpu", user_hold=True)
        state["current_gpu_job"] = job
        state["phase"] = "gpu_queued"
        write_json(self.state_file, state)
        self.qsub(state, "control", hold=job)
        run_command(["qrls", job], cwd=self.recipe, capture_output=True)
        state["gpu_user_hold_released"] = True

    def submit(self, group):
        with self.locked_state() as state:
            if state:
                raise RuntimeError(
                    f"Pipeline already exists; inspect {self.state_file}"
                )
            if any(self.exp.glob("*.pth")) or (self.exp / "config.yaml").exists():
                raise RuntimeError(
                    "Fresh experiment output already contains model artifacts"
                )
            if not self.config.is_file():
                raise RuntimeError(f"Missing experiment config: {self.config}")
            data_report = json.loads(
                (self.recipe / "exp/tflocoformer_damsep_data_check.json").read_text()
            )
            if data_report["condition"] != "whamr":
                raise RuntimeError("CPU preflight did not validate noisy WHAMR")
            for source, digest in data_report["hashes"].items():
                if (
                    hashlib.sha256((self.recipe / source).read_bytes()).hexdigest()
                    != digest
                ):
                    raise RuntimeError(f"Data changed after CPU preflight: {source}")
            expected = read_ids(self.test_scp)
            if len(expected) != 3000:
                raise RuntimeError(
                    f"WHAMR test split must contain 3000 utterances, got {len(expected)}"
                )
            state.update(
                version=1,
                variant=self.variant,
                group=group,
                commit=commit(),
                config_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest(),
                recipe=str(self.recipe),
                expdir=str(self.exp),
                created_at=now(),
                expected_test_count=len(expected),
                jobs=[],
                phase="submitting",
            )
            (self.recipe / "qsub_logs").mkdir(exist_ok=True)
            try:
                self.enqueue_gpu(state)
            except Exception as error:
                state.update(phase="submission_failed", failure=str(error))
                raise

    def command(self, stage):
        return [
            "bash",
            "run_tflocoformer_damsep.sh",
            "--variant",
            self.variant,
            "--stage",
            str(stage),
            "--stop_stage",
            str(stage),
            "--ngpu",
            "0" if stage == 8 else "1",
            "--nj",
            "4",
            "--batch_size",
            "4",
            "--accum_grad",
            "1",
            "--resume",
            "true",
            "--expdir",
            str(self.exp.relative_to(self.recipe)),
            "--inference_model",
            "valid.loss_clean.best.pth",
        ]

    def stage(self, stage, job):
        log = self.directory / f"stage{stage}.{job}.log"
        with log.open("w") as output:
            run_command(
                self.command(stage),
                cwd=self.recipe,
                stdout=output,
                stderr=subprocess.STDOUT,
            )

    def normal_training_stop(self):
        logs = list(self.exp.glob("train*.log")) + list(self.exp.glob("**/train.*.log"))
        for path in logs:
            text = path.read_text(errors="replace")
            if re.search(
                r"\[Early stopping\].*The training was stopped at \d+epoch", text
            ):
                return "early_stopping"
            if re.search(r"The training was finished at \d+ epochs", text):
                return "max_epoch"
        raise RuntimeError(
            "Stage 6 returned without a normal ESPnet stop; inspect native training logs"
        )

    def verify_inference(self):
        expected = read_ids(self.test_scp)
        output = self.exp / "enhanced_tt"
        metadata = json.loads((output / "inference.json").read_text())
        if (
            metadata["count"] != len(expected)
            or metadata["clean_only"]
            or metadata.get("ctf_enabled") != (self.variant == "three_loss")
        ):
            raise RuntimeError("Incomplete or unexpected inference metadata")
        if (
            Path(metadata["model_file"]).resolve()
            != (self.exp / "valid.loss_clean.best.pth").resolve()
        ):
            raise RuntimeError("Inference did not use valid.loss_clean.best.pth")
        for branch in ("clean", "reverb"):
            for speaker in (1, 2):
                scp = output / branch / f"spk{speaker}.scp"
                if read_ids(scp) != expected:
                    raise RuntimeError(f"Incomplete inference utterance set: {scp}")
                for line in scp.read_text().splitlines():
                    _, waveform = line.split(None, 1)
                    if not checkpoint_exists(Path(waveform)):
                        raise RuntimeError(f"Missing output waveform: {waveform}")
        if self.variant == "three_loss":
            if read_ids(output / "ctf.scp") != expected:
                raise RuntimeError("Incomplete CTF inference set")
        return len(expected)

    def gpu(self):
        job = os.environ.get("JOB_ID")
        if not job:
            raise RuntimeError("GPU execution requires a scheduler allocation (JOB_ID)")
        with self.locked_state() as state:
            self.verify_source(state)
            if state["current_gpu_job"] != job:
                raise RuntimeError(
                    "This GPU job is not the current pipeline allocation"
                )
            resumed = sum(entry["kind"] == "gpu" for entry in state["jobs"]) > 1
        preflight_marker = self.marker("preflight")
        if not preflight_marker.exists():
            self.save_phase("preflight")
            run_command(
                [
                    "python",
                    "local/check_tflocoformer_damsep.py",
                    "--variant",
                    self.variant,
                    "--report",
                    str((self.directory / "input_check.json").relative_to(self.recipe)),
                ],
                cwd=self.recipe,
            )
            write_json(preflight_marker, {"completed_at": now(), "job": job})
        if not self.marker("training").exists():
            checkpoint = self.exp / "checkpoint.pth"
            if resumed and not checkpoint_exists(checkpoint):
                raise RuntimeError(
                    "Refusing to restart training without a saved checkpoint"
                )
            if resumed:
                # Validate a potentially interrupted save only on the allocated
                # compute node; the controller never imports Torch on login.
                run_command(
                    [
                        "python",
                        "-c",
                        "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu'); "
                        "assert all(k in c for k in ('model','reporter','optimizers'))",
                        str(checkpoint),
                    ],
                    cwd=self.recipe,
                )
            self.save_phase("training")
            self.stage(6, job)
            reason = self.normal_training_stop()
            if not checkpoint_exists(checkpoint) or not checkpoint_exists(
                self.exp / "valid.loss_clean.best.pth"
            ):
                raise RuntimeError(
                    "Training completed without a saved checkpoint and best model"
                )
            write_json(
                self.marker("training"),
                {"completed_at": now(), "job": job, "reason": reason},
            )
        if not self.marker("inference").exists():
            if not checkpoint_exists(self.exp / "valid.loss_clean.best.pth"):
                raise RuntimeError("Inference requires the saved best model")
            self.save_phase("inference")
            self.stage(7, job)
            count = self.verify_inference()
            write_json(
                self.marker("inference"),
                {"completed_at": now(), "job": job, "count": count},
            )
        self.save_phase("gpu_complete")

    def accounting(self, job):
        for attempt in range(7):
            result = subprocess.run(
                ["qacct", "-j", job], cwd=self.recipe, capture_output=True, text=True
            )
            if result.returncode == 0:
                try:
                    return parse_accounting(result.stdout)
                except (RuntimeError, ValueError):
                    pass
            if attempt < 6:
                time.sleep(5)
        raise RuntimeError(f"Accounting for job {job} is unavailable after 30 seconds")

    def control(self, job):
        with self.locked_state() as state:
            self.verify_source(state)
            gpu = next(
                entry
                for entry in state["jobs"]
                if entry["kind"] == "gpu" and entry["id"] == job
            )
            if gpu.get("handled"):
                return
            if state["current_gpu_job"] != job:
                raise RuntimeError(
                    "Controller is not associated with the current GPU job"
                )
            result = self.accounting(job)
            gpu["accounting"] = result
            phase = state["phase"]
            if result["failed"] == 0 and result["exit_status"] == 0:
                if (
                    not self.marker("training").exists()
                    or not self.marker("inference").exists()
                ):
                    raise RuntimeError(
                        "GPU job exited zero without verified training/inference markers"
                    )
                self.verify_inference()
                state["phase"] = "score_queued"
                self.qsub(state, "score")
            elif is_walltime(result):
                if phase == "training":
                    checkpoint = self.exp / "checkpoint.pth"
                    previous = state.get("checkpoint_before_job")
                    if not checkpoint_exists(checkpoint):
                        raise RuntimeError(
                            "Time limit reached before saving a checkpoint; no automatic restart"
                        )
                    if (
                        previous is not None
                        and checkpoint.stat().st_mtime_ns <= previous
                    ):
                        raise RuntimeError(
                            "Training made no checkpoint progress; no automatic restart"
                        )
                elif phase != "inference" or not self.marker("training").exists():
                    raise RuntimeError(
                        f"Time limit during {phase}; no automatic restart"
                    )
                gpu["continuation_reason"] = "scheduler_walltime"
                self.enqueue_gpu(state)
            else:
                state.update(
                    phase="failed", failure=f"GPU job {job} failed", failed_job=job
                )
                gpu["handled"] = True
                write_json(self.state_file, state)
                raise RuntimeError(
                    f"GPU job failed outside a time limit: {result}; no automatic retry"
                )
            gpu["handled"] = True

    def score(self):
        job = os.environ.get("JOB_ID")
        if not job:
            raise RuntimeError("CPU scoring requires a scheduler allocation (JOB_ID)")
        with self.locked_state() as state:
            self.verify_source(state)
            if not any(
                entry["kind"] == "score" and entry["id"] == job
                for entry in state["jobs"]
            ):
                raise RuntimeError("This scoring job was not submitted by the pipeline")
        self.verify_inference()
        self.save_phase("scoring")
        self.stage(8, job)
        expected = read_ids(self.test_scp)
        scoring = self.exp / "enhanced_tt/clean/scoring"
        results = {}
        for metric in METRICS:
            values = []
            for speaker in (1, 2):
                path = scoring / f"{metric}_spk{speaker}"
                if read_ids(path) != expected:
                    raise RuntimeError(f"Incomplete score utterance set: {path}")
                values.extend(
                    float(line.split()[1]) for line in path.read_text().splitlines()
                )
            summary = float((scoring / f"result_{metric.lower()}.txt").read_text())
            if not all(math.isfinite(value) for value in values) or not math.isfinite(
                summary
            ):
                raise RuntimeError(f"Nonfinite test scores: {metric}")
            if abs(summary - sum(values) / len(values)) > 0.00501:
                raise RuntimeError(
                    f"Score summary does not match individual scores: {metric}"
                )
            results[metric] = summary
        (self.exp / "RESULTS.md").write_text(
            f"# {self.variant}\n\n"
            f"WHAMR 8 kHz test: {len(expected)} utterances, clean references, "
            "valid.loss_clean.best.pth, PCM16 with 0.9 peak normalization.\n\n"
            "| Metric | Value |\n|---|---:|\n"
            + "".join(
                f"| {metric} | {value:.2f} |\n" for metric, value in results.items()
            )
        )
        write_json(
            self.marker("scoring"),
            {
                "completed_at": now(),
                "job": job,
                "count": len(expected),
                "results": results,
            },
        )
        self.save_phase("complete", results=results, completed_at=now())

    def record_failure(self, message):
        with self.locked_state() as state:
            if state and state.get("phase") != "complete":
                state.update(phase="failed", failure=message, failed_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "gpu", "control", "score"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--group", default="tga-shinoda")
    parser.add_argument("--gpu-job")
    args = parser.parse_args()
    pipeline = Pipeline(args.variant)
    try:
        if args.action == "submit":
            pipeline.submit(args.group)
        elif args.action == "gpu":
            pipeline.gpu()
        elif args.action == "control":
            if args.gpu_job is None:
                parser.error("control requires --gpu-job")
            pipeline.control(args.gpu_job)
        else:
            pipeline.score()
    except Exception as error:
        # A duplicate submission must never mark an existing active pipeline as
        # failed. Execution failures, however, remain visible in its state file.
        if args.action != "submit":
            pipeline.record_failure(f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
