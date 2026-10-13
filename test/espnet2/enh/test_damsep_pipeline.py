"""Scheduler safety and output validation for the automatic ablation workflow."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[3] / "egs2/whamr/damsep_clean"
SPEC = importlib.util.spec_from_file_location(
    "damsep_ablation_pipeline", RECIPE / "local/ablation_pipeline.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    monkeypatch.chdir(recipe)
    instance = PIPELINE.Pipeline("spmamba_clean", recipe)
    instance.config.parent.mkdir(parents=True)
    instance.config.write_text("seed: 0\n")
    instance.test_scp.parent.mkdir(parents=True)
    instance.test_scp.write_text("".join(f"u{i:04d} input.wav\n" for i in range(3000)))
    monkeypatch.setattr(PIPELINE, "commit", lambda: "abc123")
    calls = []
    numbers = iter(str(i) for i in range(100, 200))

    def command(arguments, **kwargs):
        calls.append(arguments)
        output = next(numbers) + "\n" if arguments[0] == "qsub" else ""
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(PIPELINE, "run_command", command)
    instance.submit("tga-shinoda")
    return instance, calls


def state(instance):
    return json.loads(instance.state_file.read_text())


def inference_outputs(instance):
    output = instance.exp / "enhanced_tt"
    (output / "clean").mkdir(parents=True)
    waveform = output / "clean/output.wav"
    waveform.write_bytes(b"RIFF")
    for speaker in (1, 2):
        (output / "clean" / f"spk{speaker}.scp").write_text(
            "".join(f"u{i:04d} {waveform}\n" for i in range(3000))
        )
    (instance.exp / "valid.loss.best.pth").write_bytes(b"best")
    PIPELINE.write_json(
        output / "inference.json",
        {
            "count": 3000,
            "clean_only": True,
            "model_file": str(instance.exp / "valid.loss.best.pth"),
        },
    )
    for marker in ("training", "inference"):
        PIPELINE.write_json(instance.marker(marker), {"status": "complete"})


def test_submission_installs_controller_before_releasing_gpu(pipeline):
    instance, calls = pipeline
    assert [call[0] for call in calls] == ["qsub", "qsub", "qrls"]
    assert "-h" in calls[0]
    assert calls[1][calls[1].index("-hold_jid") + 1] == "100"
    assert calls[2] == ["qrls", "100"]
    saved = state(instance)
    assert [job["id"] for job in saved["jobs"]] == ["100", "101"]
    assert saved["commit"] == "abc123"
    assert saved["expected_test_count"] == 3000
    assert saved["jobs"][0]["upper_points"] == 3.84
    assert saved["jobs"][1]["upper_points"] == 0.004


def test_duplicate_submission_preserves_running_pipeline(pipeline):
    instance, calls = pipeline
    before = state(instance)
    with pytest.raises(RuntimeError, match="already exists"):
        instance.submit("tga-shinoda")
    after = state(instance)
    assert after["jobs"] == before["jobs"]
    assert after["phase"] == before["phase"]
    assert len(calls) == 3


def test_controller_submission_failure_keeps_gpu_user_held(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    instance = PIPELINE.Pipeline("feature_dereverb_clean", recipe)
    instance.config.parent.mkdir(parents=True)
    instance.config.write_text("seed: 0\n")
    instance.test_scp.parent.mkdir(parents=True)
    instance.test_scp.write_text("".join(f"u{i} x.wav\n" for i in range(3000)))
    monkeypatch.setattr(PIPELINE, "commit", lambda: "abc123")
    calls = []

    def fail_controller(arguments, **kwargs):
        calls.append(arguments)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout="123\n", stderr="")

    monkeypatch.setattr(PIPELINE, "run_command", fail_controller)
    with pytest.raises(subprocess.CalledProcessError):
        instance.submit("tga-shinoda")
    assert len(calls) == 2
    assert "-h" in calls[0]
    saved = state(instance)
    assert saved["phase"] == "submission_failed"
    assert saved["jobs"][0]["id"] == "123"


def test_walltime_continues_only_with_checkpoint_progress(pipeline, monkeypatch):
    instance, calls = pipeline
    checkpoint = instance.exp / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    instance.save_phase("training")
    result = {"failed": 44, "exit_status": 137, "ru_wallclock": 86394}
    monkeypatch.setattr(instance, "accounting", lambda job: result)
    instance.control("100")
    saved = state(instance)
    assert saved["current_gpu_job"] == "102"
    assert saved["jobs"][0]["continuation_reason"] == "scheduler_walltime"
    assert saved["jobs"][0]["handled"]
    assert [call[0] for call in calls[-3:]] == ["qsub", "qsub", "qrls"]
    assert calls[-2][calls[-2].index("-hold_jid") + 1] == "102"
    # No new saved epoch during the second chunk: prevent an expensive retry loop.
    instance.save_phase("training")
    with pytest.raises(RuntimeError, match="no checkpoint progress"):
        instance.control("102")
    assert len(calls) == 6


def test_walltime_without_checkpoint_never_restarts(pipeline, monkeypatch):
    instance, calls = pipeline
    instance.save_phase("training")
    monkeypatch.setattr(
        instance,
        "accounting",
        lambda job: {"failed": 44, "exit_status": 137, "ru_wallclock": 86398},
    )
    with pytest.raises(RuntimeError, match="before saving a checkpoint"):
        instance.control("100")
    assert len(calls) == 3


@pytest.mark.parametrize(
    "failed,exit_status,wallclock", [(44, 137, 120), (0, 1, 86390)]
)
def test_non_walltime_failure_never_retries(
    pipeline, monkeypatch, failed, exit_status, wallclock
):
    instance, calls = pipeline
    instance.save_phase("training")
    monkeypatch.setattr(
        instance,
        "accounting",
        lambda job: {
            "failed": failed,
            "exit_status": exit_status,
            "ru_wallclock": wallclock,
        },
    )
    with pytest.raises(RuntimeError, match="outside a time limit"):
        instance.control("100")
    assert state(instance)["phase"] == "failed"
    assert state(instance)["jobs"][0]["accounting"]["exit_status"] == exit_status
    assert len(calls) == 3


def test_controller_success_requires_verified_outputs(pipeline, monkeypatch):
    instance, calls = pipeline
    monkeypatch.setattr(
        instance,
        "accounting",
        lambda job: {"failed": 0, "exit_status": 0, "ru_wallclock": 500},
    )
    with pytest.raises(RuntimeError, match="without verified"):
        instance.control("100")
    assert len(calls) == 3
    inference_outputs(instance)
    instance.control("100")
    saved = state(instance)
    assert saved["phase"] == "score_queued"
    assert saved["jobs"][-1]["kind"] == "score"
    assert saved["jobs"][-1]["resource"] == "cpu_8"
    assert saved["jobs"][-1]["upper_points"] == 0.048
    instance.control("100")
    assert len(calls) == 4


def test_inference_exact_ids_and_saved_best_model_required(pipeline):
    instance, _ = pipeline
    inference_outputs(instance)
    assert instance.verify_inference() == 3000
    output = instance.exp / "enhanced_tt"
    metadata = json.loads((output / "inference.json").read_text())
    metadata["model_file"] = str(instance.exp / "latest.pth")
    PIPELINE.write_json(output / "inference.json", metadata)
    with pytest.raises(RuntimeError, match="best.pth"):
        instance.verify_inference()
    metadata["model_file"] = str(instance.exp / "valid.loss.best.pth")
    PIPELINE.write_json(output / "inference.json", metadata)
    speaker = output / "clean/spk2.scp"
    speaker.write_text(speaker.read_text().replace("u2999", "wrong"))
    with pytest.raises(RuntimeError, match="utterance set"):
        instance.verify_inference()


def test_invalid_gradient_stop_does_not_count_as_training_completion(pipeline):
    instance, _ = pipeline
    log = instance.exp / "train.1.log"
    log.write_text(
        "The gradients at all steps are invalid. This training was stopped at 3epoch\n"
    )
    with pytest.raises(RuntimeError, match="normal ESPnet stop"):
        instance.normal_training_stop()
    log.write_text(
        "[Early stopping] valid.loss has not been improved 5 epochs continuously. The training was stopped at 15epoch\n"
    )
    assert instance.normal_training_stop() == "early_stopping"
    log.write_text("The training was finished at 500 epochs\n")
    assert instance.normal_training_stop() == "max_epoch"


def test_native_cpu_scoring_validates_all_samples_and_writes_results(
    pipeline, monkeypatch
):
    instance, _ = pipeline
    inference_outputs(instance)
    with instance.locked_state() as saved:
        saved["jobs"].append({"kind": "score", "id": "200"})
    monkeypatch.setenv("JOB_ID", "200")
    scoring = instance.exp / "enhanced_tt/clean/scoring"
    scoring.mkdir(parents=True)
    for metric in PIPELINE.METRICS:
        for speaker in (1, 2):
            (scoring / f"{metric}_spk{speaker}").write_text(
                "".join(f"u{i:04d} 12.34\n" for i in range(3000))
            )
        (scoring / f"result_{metric.lower()}.txt").write_text("12.34\n")
    stages = []
    monkeypatch.setattr(
        instance, "stage", lambda stage, job: stages.append((stage, job))
    )
    instance.score()
    assert stages == [(8, "200")]
    assert state(instance)["phase"] == "complete"
    assert state(instance)["results"]["SI_SNR"] == 12.34
    assert "3000 utterances" in (instance.exp / "RESULTS.md").read_text()
    (scoring / "SI_SNR_spk1").write_text(
        (scoring / "SI_SNR_spk1").read_text().replace("12.34", "nan", 1)
    )
    with pytest.raises(RuntimeError, match="Nonfinite"):
        instance.score()


def test_accounting_parser_distinguishes_oom_and_time_limit():
    parsed = PIPELINE.parse_accounting(
        "jobnumber 456\nfailed 44 : died through signal\nexit_status 137\nru_wallclock 86399.9\nhostname r1n1\n"
    )
    assert PIPELINE.is_walltime(parsed)
    parsed["ru_wallclock"] = 10
    assert not PIPELINE.is_walltime(parsed)
