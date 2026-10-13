"""Verify the CTF ablation, baseline parity, source assignment and native I/O."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from espnet2.enh.damsep.tflocoformer import ESPnetTFLocoformerDAMSEPModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "egs2/whamr/damsep_clean"


@pytest.fixture
def reference_mamba(monkeypatch):
    pytest.importorskip("mamba_ssm")
    from mamba_ssm.modules import mamba_simple
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    monkeypatch.setenv("LOOK2HEAR_USE_TORCH_RMSNORM", "1")
    monkeypatch.setattr(mamba_simple, "causal_conv1d_fn", None)
    monkeypatch.setattr(mamba_simple, "selective_scan_fn", selective_scan_ref)
    init = mamba_simple.Mamba.__init__

    def reference_init(self, *args, **kwargs):
        kwargs["use_fast_path"] = False
        init(self, *args, **kwargs)

    monkeypatch.setattr(mamba_simple.Mamba, "__init__", reference_init)


def model(weight):
    return ESPnetTFLocoformerDAMSEPModel(
        separator_conf=dict(
            n_layers=1,
            emb_dim=16,
            ffn_hidden_dim=[16, 16],
            conv1d_kernel=4,
            attention_dim=16,
        ),
        loss_w_reconstruction=weight,
    )


def test_pair_initialization_ctf_skip_and_shared_output(reference_mamba):
    torch.manual_seed(0)
    a = model(0).eval()
    torch.manual_seed(0)
    b = model(0.5).eval()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def forbidden(*args):
        raise AssertionError("Disabled CTF branch executed")

    handles = [
        module.register_forward_pre_hook(forbidden)
        for module in (
            a.network.recrir.compress_CTF,
            a.network.recrir.weight_layer,
            a.network.recrir.decoder_CTF,
            *a.network.recrir.ctf_layers,
        )
    ]
    mixture = torch.randn(2, 640)
    output_a = a.separate(mixture)
    output_b = b.separate(mixture)
    for field in ("x_sep", "x_derev"):
        torch.testing.assert_close(output_a[field], output_b[field])
    assert "rir" not in output_a and output_b["rir"].shape == (4, 2, 257, 60)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_variable_lengths_backward_and_resume(reference_mamba, weight):
    torch.manual_seed(6)
    m = model(weight).eval()
    fields = (
        "speech_mix",
        "speech_ref1",
        "speech_ref2",
        "speech_reverb1",
        "speech_reverb2",
    )
    values = {field: torch.randn(2, 768) for field in fields}
    values["speech_mix_lengths"] = torch.tensor([640, 768])
    loss, stats, count = m(**values)
    assert count == 2 and torch.isfinite(loss)
    single = []
    for i, length in enumerate((640, 768)):
        item = {field: values[field][i : i + 1, :length] for field in fields}
        single.append(m(**item, speech_mix_lengths=torch.tensor([length]))[0])
    torch.testing.assert_close(loss.squeeze(), torch.stack(single).mean())
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in m.parameters()
        if p.requires_grad
    )
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    assert ("loss_reconstruction" in stats) == bool(weight)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.001)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
    opt.step()
    restored = model(weight).eval()
    restored.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(**values)[0], restored(**values)[0])


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_same_pit_for_clean_reverb_and_ctf(reference_mamba, weight):
    m = model(weight).eval()
    clean = torch.randn(2, 2, 640)
    reverb = clean.clone()
    output = dict(x_derev=clean + 0.02 * torch.randn_like(clean), x_sep=reverb.clone())
    if weight:
        output["rir"] = torch.zeros(4, 2, 257, 60)
        output["rir"][:, 0, :, 0] = 1
    loss, stats = m._loss(output, clean, reverb)
    expected = SISNRLoss(eps=1e-7)(clean, output["x_derev"]).mean()
    torch.testing.assert_close(stats["loss_clean"], expected)
    swapped = {key: value.flip(1) for key, value in output.items() if key != "rir"}
    if weight:
        swapped["rir"] = (
            output["rir"].reshape(2, 2, 2, 257, 60).flip(1).reshape(4, 2, 257, 60)
        )
    loss_swap, stats_swap = m._loss(swapped, clean, reverb)
    torch.testing.assert_close(loss, loss_swap)
    assert stats_swap["pit_swap_ratio"] == 1
    assert stats["loss_reverb"] == 0
    if weight:
        assert stats["loss_reconstruction"] == 0


def test_configs_match_baseline_and_only_reconstruction_differs():
    a = yaml.safe_load(
        (
            RECIPE / "conf/tuning/train_tflocoformer_damsep_whamr_8k_clean_reverb.yaml"
        ).read_text()
    )
    b = yaml.safe_load(
        (
            RECIPE / "conf/tuning/train_tflocoformer_damsep_whamr_8k_three_loss.yaml"
        ).read_text()
    )
    baseline = yaml.safe_load(
        (
            ROOT
            / "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_whamr_8k.yaml"
        ).read_text()
    )
    assert a["model_conf"]["loss_w_reconstruction"] == 0
    assert b["model_conf"]["loss_w_reconstruction"] == 0.5
    b["model_conf"]["loss_w_reconstruction"] = 0
    assert a == b
    assert a["model_conf"]["separator_conf"] == baseline["separator_conf"]
    for key in (
        "seed",
        "max_epoch",
        "use_amp",
        "batch_size",
        "optim",
        "optim_conf",
        "scheduler",
        "scheduler_conf",
        "patience",
    ):
        assert a[key] == baseline[key]
    assert a["early_stopping_criterion"] == ["valid", "loss_clean", "min"]
    assert a["val_scheduler_criterion"] == ["valid", "loss_clean"]


def test_tf_initialization_exactly_matches_baseline(reference_mamba):
    from espnet2.tasks.enh import EnhancementTask

    config = (
        ROOT
        / "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_whamr_8k.yaml"
    )
    options = EnhancementTask.get_parser().parse_args(
        ["--config", str(config), "--output_dir", "unused"]
    )
    torch.manual_seed(0)
    baseline = EnhancementTask.build_model(options)
    torch.manual_seed(0)
    extended = ESPnetTFLocoformerDAMSEPModel(loss_w_reconstruction=0)
    for key, value in baseline.separator.state_dict().items():
        torch.testing.assert_close(
            value, extended.network.separator.state_dict()[key], rtol=0, atol=0
        )


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_native_trainer_resume_and_inference(reference_mamba, tmp_path, weight):
    import argparse
    import json

    from espnet2.bin.damsep_inference import inference
    from espnet2.tasks.damsep import DAMSEPTask

    fields = [
        "speech_mix",
        "speech_ref1",
        "speech_ref2",
        "speech_reverb1",
        "speech_reverb2",
    ]
    rng = np.random.default_rng(7)
    source = [rng.normal(0, 0.02, 640) for _ in range(3)]
    clean1, clean2, noise = source
    rev1 = np.convolve(clean1, [0.8, 0.2])[:640]
    rev2 = np.convolve(clean2, [0.8, 0.2])[:640]
    data_args = []
    for field, audio in zip(fields, [rev1 + rev2 + noise, clean1, clean2, rev1, rev2]):
        wav = tmp_path / f"{field}.wav"
        sf.write(wav, audio, 8000, subtype="FLOAT")
        scp = tmp_path / f"{field}.scp"
        scp.write_text(f"u {wav}\n")
        for split in ["train", "valid"]:
            data_args += [
                f"--{split}_data_path_and_name_and_type",
                f"{scp},{field},sound",
            ]
    shape = tmp_path / "shape"
    shape.write_text("u 640\n")
    config = tmp_path / "train.yaml"
    production = yaml.safe_load(
        (
            RECIPE / "conf/tuning/train_tflocoformer_damsep_whamr_8k_clean_reverb.yaml"
        ).read_text()
    )
    production["model_conf"].update(
        separator_conf=dict(
            n_layers=1,
            emb_dim=16,
            ffn_hidden_dim=[16, 16],
            conv1d_kernel=4,
            attention_dim=16,
        ),
        loss_w_reconstruction=weight,
    )
    production.update(num_workers=0, batch_size=1)
    production["scheduler_conf"]["warmup_steps"] = 1
    config.write_text(yaml.safe_dump(production))
    exp = tmp_path / "exp"
    cmd = [
        "--config",
        str(config),
        "--output_dir",
        str(exp),
        "--ngpu",
        "0",
        "--use_tensorboard",
        "false",
        "--log_level",
        "WARNING",
        "--train_shape_file",
        str(shape),
        "--valid_shape_file",
        str(shape),
    ] + data_args
    DAMSEPTask.main(cmd=cmd + ["--max_epoch", "1"])
    DAMSEPTask.main(cmd=cmd + ["--max_epoch", "2", "--resume", "true"])
    assert (
        torch.load(exp / "checkpoint.pth", map_location="cpu")["reporter"]["epoch"] == 2
    )
    dest = tmp_path / "inference"
    inference(
        argparse.Namespace(
            train_config=str(exp / "config.yaml"),
            model_file=str(exp / "valid.loss_clean.best.pth"),
            wav_scp=tmp_path / "speech_mix.scp",
            output_dir=dest,
            key_file=None,
            device="cpu",
            normalize_output_wav=True,
        )
    )
    assert (dest / "clean/spk1.scp").is_file() and (dest / "reverb/spk1.scp").is_file()
    assert (dest / "ctf.scp").exists() == bool(weight)
    assert json.loads((dest / "inference.json").read_text())["ctf_enabled"] == bool(
        weight
    )


def test_noisy_data_validation_and_shape_reuse(tmp_path, monkeypatch):
    path = RECIPE / "local/check_tflocoformer_damsep.py"
    import sys

    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("tfd_check", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    for split, phase in [("tr", "train"), ("cv", "valid"), ("tt", None)]:
        wave = [rng.normal(0, 0.1, (640, 2)) for _ in range(5)]
        c1, c2, r1, r2, noise = wave
        noisy = r1 + r2 + noise
        values = {
            "wav": noisy,
            "spk1": c1,
            "spk2": c2,
            "spk1_reverb": r1,
            "spk2_reverb": r2,
            "noise1": noise,
        }
        for dump, entries in [
            ("dump_whamr", {k: values[k] for k in ("wav", "spk1", "spk2", "noise1")}),
            ("dump_reverb", {k: values[k] for k in ("spk1_reverb", "spk2_reverb")}),
            ("dump_clean", dict(wav=r1 + r2, spk1=c1, spk2=c2)),
        ]:
            subset = "mix_clean_reverb" if dump == "dump_clean" else "mix_both_reverb"
            folder = tmp_path / dump / "raw" / f"{split}_{subset}_min_8k"
            folder.mkdir(parents=True)
            for name, audio in entries.items():
                wav = folder / f"{name}.wav"
                sf.write(wav, audio, 8000, subtype="FLOAT")
                (folder / f"{name}.scp").write_text(f"u {wav}\n")
        if phase:
            folder = tmp_path / "stats" / phase
            folder.mkdir(parents=True)
            for field in checker.FIELDS:
                (folder / f"{field}_shape").write_text("u 640,2\n")
    report, _, _ = checker.check_data(tmp_path / "stats", dict(tr=1, cv=1, tt=1))
    assert report["splits"]["tr"]["noise_sum_checked"]
    corrupt = tmp_path / "dump_whamr/raw/tr_mix_both_reverb_min_8k/wav.wav"
    sf.write(corrupt, np.zeros((640, 2)), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="source semantics"):
        checker.check_data(tmp_path / "stats", dict(tr=1, cv=1, tt=1))
