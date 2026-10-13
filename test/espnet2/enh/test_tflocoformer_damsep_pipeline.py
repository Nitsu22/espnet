"""Check continuation boundaries and complete A/B inference requirements."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "egs2/whamr/damsep_clean/local/tflocoformer_damsep_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("tfd_pipeline", PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


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
