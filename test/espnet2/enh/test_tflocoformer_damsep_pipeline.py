"""Check continuation boundaries and complete A/B inference requirements."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "egs2/whamr/damsep_clean/local/tflocoformer_damsep_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("tfd_pipeline", PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def test_scheduler_fallback_when_batch_path_omits_uge(tmp_path, monkeypatch):
    binary = tmp_path / "qacct"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(PIPELINE.shutil, "which", lambda name: None)
    monkeypatch.setattr(PIPELINE, "SCHEDULER_DIR", tmp_path)
    assert PIPELINE.scheduler_program("qacct") == str(binary)


@pytest.mark.parametrize("invalid", [None, "active", "config", "model", "checkpoint"])
def test_controller_recovery_preserves_experiment(tmp_path, monkeypatch, invalid):
    pipeline = PIPELINE.Pipeline("clean_reverb", tmp_path)
    pipeline.config.parent.mkdir(parents=True)
    pipeline.config.write_text("unchanged configuration")
    pipeline.exp.mkdir(parents=True)
    if invalid != "checkpoint":
        (pipeline.exp / "checkpoint.pth").write_text("saved checkpoint")
    state = dict(
        phase="failed",
        failure="FileNotFoundError: qacct",
        commit="old",
        config_sha256=hashlib.sha256(pipeline.config.read_bytes()).hexdigest(),
        current_gpu_job="10",
        checkpoint_before_job=None,
        jobs=[dict(id="10", kind="gpu"), dict(id="11", kind="control")],
    )
    PIPELINE.write_json(pipeline.state_file, state)
    if invalid == "config":
        pipeline.config.write_text("changed configuration")
    monkeypatch.setattr(PIPELINE, "commit", lambda: "new")
    monkeypatch.setattr(PIPELINE, "scheduler_program", lambda name: name)
    monkeypatch.setattr(
        PIPELINE.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, "job_number: 10" if invalid == "active" else "", ""
        ),
    )
    changed = (
        "espnet2/enh/damsep/tflocoformer.py"
        if invalid == "model"
        else "egs2/whamr/damsep_clean/local/tflocoformer_damsep_pipeline.py"
    )
    monkeypatch.setattr(
        PIPELINE,
        "run_command",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, changed, ""),
    )
    monkeypatch.setattr(
        pipeline,
        "accounting",
        lambda job: dict(failed=44, exit_status=137, ru_wallclock=86400),
    )
    calls = []

    def enqueue(updated):
        calls.append(updated.copy())
        updated.update(phase="gpu_queued", current_gpu_job="12")

    monkeypatch.setattr(pipeline, "enqueue_gpu", enqueue)
    if invalid:
        with pytest.raises(RuntimeError):
            pipeline.recover()
        assert not calls
    else:
        pipeline.recover()
        recovered = json.loads(pipeline.state_file.read_text())
        assert recovered["commit"] == "new" and recovered["current_gpu_job"] == "12"
        assert recovered["jobs"][0]["handled"]
        assert (
            recovered["recovery_history"][0]["previous_failure"]
            == "FileNotFoundError: qacct"
        )
        assert "failure" not in recovered


def test_only_scheduler_confirmed_walltime_is_continued():
    assert PIPELINE.is_walltime(dict(failed=44, exit_status=137, ru_wallclock=86400))
    for record in [
        dict(failed=0, exit_status=137, ru_wallclock=86400),
        dict(failed=44, exit_status=137, ru_wallclock=100),
        dict(failed=44, exit_status=1, ru_wallclock=86400),
    ]:
        assert not PIPELINE.is_walltime(record)


@pytest.mark.parametrize("variant", ["clean_reverb", "three_loss"])
def test_pipeline_uses_noisy_data_and_clean_best_model(tmp_path, variant):
    p = PIPELINE.Pipeline(variant, tmp_path)
    assert "mix_both_reverb" in str(p.test_scp)
    command = p.command(6)
    assert command[1] == "run_tflocoformer_damsep.sh"
    assert command[command.index("--batch_size") + 1] == "4"
    assert (
        command[command.index("--inference_model") + 1] == "valid.loss_clean.best.pth"
    )
    p.test_scp.parent.mkdir(parents=True)
    p.test_scp.write_text("u unused\n")
    p.exp.mkdir(parents=True)
    best = p.exp / "1epoch.pth"
    best.write_text("model")
    (p.exp / "valid.loss_clean.best.pth").symlink_to(best.name)
    output = p.exp / "enhanced_tt"
    output.mkdir()
    for branch in ["clean", "reverb"]:
        (output / branch).mkdir()
        for speaker in [1, 2]:
            wav = output / branch / f"speaker{speaker}.wav"
            wav.write_text("waveform")
            (output / branch / f"spk{speaker}.scp").write_text(f"u {wav}\n")
    (output / "inference.json").write_text(
        json.dumps(
            dict(
                count=1,
                clean_only=False,
                ctf_enabled=variant == "three_loss",
                model_file=str(best),
            )
        )
    )
    if variant == "three_loss":
        (output / "ctf.scp").write_text("u unused\n")
    assert p.verify_inference() == 1
    (output / "reverb/spk2.scp").write_text("other unused\n")
    with pytest.raises(RuntimeError, match="Incomplete inference"):
        p.verify_inference()
