"""Check shared PIT, complex convolution, the real network and zero-copy data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from espnet2.enh.damsep.espnet_model import (
    ESPnetDAMSEPModel,
    complex_convolve,
    negative_sdr,
)
from espnet2.train.preprocessor_rec_rir_pit import RecRIRPITPreprocessor


@pytest.fixture
def cpu_model(monkeypatch):
    pytest.importorskip("mamba_ssm")
    from mamba_ssm.modules import mamba_simple
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    # Use the official differentiable PyTorch computation, not a dummy mixer.
    # CUDA production modules and their weights are otherwise unchanged.
    monkeypatch.setenv("LOOK2HEAR_USE_TORCH_RMSNORM", "1")
    monkeypatch.setattr(mamba_simple, "causal_conv1d_fn", None)
    monkeypatch.setattr(mamba_simple, "selective_scan_fn", selective_scan_ref)
    original_init = mamba_simple.Mamba.__init__

    def reference_init(self, *args, **kwargs):
        kwargs["use_fast_path"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(mamba_simple.Mamba, "__init__", reference_init)
    model = ESPnetDAMSEPModel(network_conf=dict(n_layers=1, emb_dim=8, emb_ks=2))
    for module in model.modules():
        if isinstance(module, mamba_simple.Mamba):
            module.use_fast_path = False
    return model


def test_complex_convolution_batch_speaker_alignment():
    torch.manual_seed(3)
    signal = torch.randn(2, 2, 3, 7, dtype=torch.complex64, requires_grad=True)
    ctf = torch.randn(2, 2, 3, 4, dtype=torch.complex64, requires_grad=True)
    expected = sum(
        F.pad(signal[..., : 7 - k], (k, 0)) * ctf[..., k, None] for k in range(4)
    )
    actual = complex_convolve(signal, ctf)
    torch.testing.assert_close(actual, expected)
    actual.abs().sum().backward()
    assert torch.isfinite(signal.grad).all() and torch.isfinite(ctf.grad).all()
    impulse = torch.zeros_like(ctf)
    impulse[..., 0] = 1
    torch.testing.assert_close(complex_convolve(signal, impulse), signal)


def test_negative_snr_differs_from_scale_invariant():
    target = torch.randn(2, 2, 100)
    estimate = 2 * target
    assert negative_sdr(estimate, target).abs().max() < 1e-5
    assert (negative_sdr(estimate, target, True) < -80).all()


def test_shared_pit_reorders_all_three_outputs(cpu_model):
    torch.manual_seed(5)
    clean = torch.randn(2, 2, 640)
    # Identity CTF exactly reconstructs the reverberant target.
    output = {
        "x_derev": clean + 0.01 * torch.randn_like(clean),
        "x_sep": clean.clone(),
        "rir": torch.zeros(4, 2, 257, 60),
    }
    output["rir"][:, 0, :, 0] = 1
    loss, stats = cpu_model._loss(output, clean, clean)
    swapped = {
        k: (
            v.flip(1)
            if k != "rir"
            else v.reshape(2, 2, 2, 257, 60).flip(1).reshape_as(v)
        )
        for k, v in output.items()
    }
    other_loss, other_stats = cpu_model._loss(swapped, clean, clean)
    torch.testing.assert_close(loss, other_loss)
    assert stats["pit_swap_ratio"] == 0 and other_stats["pit_swap_ratio"] == 1
    assert stats["loss_reconstruction"] < 1e-5
    # A wrong reverb permutation must not choose a separate matching.
    swapped["x_sep"] = output["x_sep"]
    _, bad = cpu_model._loss(swapped, clean, clean)
    assert bad["loss_reverb"] > 1


def test_network_outputs_and_joint_backward(cpu_model):
    torch.manual_seed(7)
    mixture = torch.randn(1, 640) * 0.05
    output = cpu_model.network(mixture)
    assert output["x_sep"].shape == output["x_derev"].shape == (1, 2, 640)
    assert output["rir"].shape == (2, 2, 257, 60)
    clean = torch.randn(1, 2, 640) * 0.02
    reverb = torch.randn_like(clean) * 0.02
    loss, stats = cpu_model._loss(output, clean, reverb)
    assert torch.isfinite(loss) and all(torch.isfinite(x) for x in stats.values())
    loss.backward()
    missing = [
        name
        for name, p in cpu_model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in cpu_model.parameters())
    for prefix in (
        "conv",
        "deconv",
        "recrir.decoder_rev",
        "recrir.decoder_CTF",
        "recrir.compress_CTF",
    ):
        assert any(
            p.grad.abs().sum() > 0
            for name, p in cpu_model.network.named_parameters()
            if name.startswith(prefix)
        )


def test_padding_is_excluded_from_network_and_loss(cpu_model):
    cpu_model.eval()
    signals = [torch.randn(2, 768) * 0.01 for _ in range(5)]
    lengths = torch.tensor([640, 768])
    names = (
        "speech_mix",
        "speech_ref1",
        "speech_ref2",
        "speech_reverb1",
        "speech_reverb2",
    )
    with torch.no_grad():
        batch = dict(zip(names, signals), speech_mix_lengths=lengths)
        actual, _, _ = cpu_model(**batch)
        singles = []
        for i, n in enumerate(lengths):
            b = {name: signal[i : i + 1, :n] for name, signal in zip(names, signals)}
            b["speech_mix_lengths"] = lengths[i : i + 1]
            singles.append(cpu_model(**b)[0])
        torch.testing.assert_close(actual, sum(singles) / 2)
        for s in signals:
            s[0, 640:] = 10000
        other, _, _ = cpu_model(**batch)
        torch.testing.assert_close(actual, other)


def test_common_crop_and_left_channel():
    proc = RecRIRPITPreprocessor(
        train=True, speech_direct_prefix="speech_ref", speech_segment=320
    )
    base = np.arange(640, dtype=np.float32)
    names = (
        "speech_mix",
        "speech_ref1",
        "speech_ref2",
        "speech_reverb1",
        "speech_reverb2",
    )
    data = {n: np.stack([base + i, base * 0 + 999], -1) for i, n in enumerate(names)}
    result = proc("example", data)
    assert result["speech_mix"].shape == (320,)
    for i, n in enumerate(names):
        np.testing.assert_allclose(result[n] - i, result["speech_mix"])


def test_batched_network_and_released_configuration(cpu_model):
    """Check the flattened batch/speaker path and the actual six-block model."""
    torch.manual_seed(11)
    mixture = torch.randn(2, 641) * 0.02
    cpu_model.eval()
    with torch.no_grad():
        batched = cpu_model.separate(mixture)
        single = [cpu_model.separate(x[None]) for x in mixture]
        for key in ("x_sep", "x_derev", "rir"):
            torch.testing.assert_close(
                batched[key], torch.cat([s[key] for s in single]), atol=1e-5, rtol=1e-4
            )
        released = ESPnetDAMSEPModel().eval()
        output = released.separate(mixture[:1])
        assert len(released.network.blocks) == 6
        assert output["x_sep"].shape == output["x_derev"].shape == (1, 2, 641)
        assert output["rir"].shape == (2, 2, 257, 60)
        assert all(torch.isfinite(x).all() for x in output.values())


def _load_preparer():
    path = (
        Path(__file__).resolve().parents[3]
        / "egs2/whamr/damsep_clean/local/prepare_nf_dump.py"
    )
    spec = importlib.util.spec_from_file_location("damsep_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dump_reuses_baseline_and_rejects_bad_reverb(tmp_path):
    preparer = _load_preparer()
    source, audio_root = tmp_path / "baseline", tmp_path / "whamr"
    rng = np.random.RandomState(1)
    for split in ("tr", "cv", "tt"):
        folder = source / "dump_clean/raw" / f"{split}_mix_clean_reverb_min_8k"
        folder.mkdir(parents=True)
        manifest = source / "data" / folder.name
        manifest.mkdir(parents=True)
        clean = [rng.randn(640, 2).astype(np.float32) * 0.01 for _ in (1, 2)]
        rev = [np.roll(x, 3, axis=0) for x in clean]
        for speaker in (1, 2):
            for kind, values in (
                ("anechoic", clean[speaker - 1]),
                ("reverb", rev[speaker - 1]),
            ):
                dest = audio_root / split / f"s{speaker}_{kind}" / "example.wav"
                dest.parent.mkdir(parents=True)
                sf.write(dest, values, 8000, subtype="FLOAT")
            (manifest / f"spk{speaker}_reverb.scp").write_text(
                "example /old/root/example.wav\n"
            )
        for name, values in (("wav", sum(rev)), ("spk1", clean[0]), ("spk2", clean[1])):
            sf.write(folder / f"{name}.wav", values, 8000, subtype="PCM_16")
            relative = (folder / f"{name}.wav").relative_to(source)
            (folder / f"{name}.scp").write_text(f"example {relative}\n")
    output = tmp_path / "dump"
    preparer.prepare(source, audio_root, output, workers=1)
    report = json.loads((output / "preparation.json").read_text())
    assert all(v["count"] == 1 for v in report["splits"].values())
    assert not list(output.rglob("*.wav"))
    reverb = audio_root / "tr/s1_reverb/example.wav"
    sf.write(reverb, np.zeros((640, 2)), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="sum of reverb"):
        preparer.prepare(source, audio_root, tmp_path / "invalid", workers=1)
    assert not (tmp_path / "invalid").exists()


def test_native_training_resume_inference_and_scoring(cpu_model, tmp_path):
    """Exercise the complete trainer/checkpoint/adapter path on real waveforms."""
    import argparse

    import yaml

    from espnet2.bin.damsep_inference import inference
    from espnet2.bin.enh_scoring import main as score_main
    from espnet2.tasks.damsep import DAMSEPTask

    rng = np.random.RandomState(19)
    clean = [rng.randn(640).astype(np.float32) * 0.02 for _ in (1, 2)]
    reverb = [
        np.convolve(x, [0.8, 0.2], mode="full")[:640].astype(np.float32) for x in clean
    ]
    fields = dict(
        speech_mix=sum(reverb),
        speech_ref1=clean[0],
        speech_ref2=clean[1],
        speech_reverb1=reverb[0],
        speech_reverb2=reverb[1],
    )
    data_args = []
    for field, samples in fields.items():
        wav = tmp_path / f"{field}.wav"
        sf.write(wav, samples, 8000, subtype="FLOAT")
        scp = tmp_path / f"{field}.scp"
        scp.write_text(f"example {wav}\n")
        for split in ("train", "valid"):
            data_args.extend(
                [f"--{split}_data_path_and_name_and_type", f"{scp},{field},sound"]
            )
    shape = tmp_path / "shape"
    shape.write_text("example 640\n")
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                model_conf=dict(network_conf=dict(n_layers=1, emb_dim=8, emb_ks=2)),
                preprocessor_conf=dict(speech_segment=640, force_single_channel=True),
                num_workers=0,
                batch_type="unsorted",
                batch_size=1,
                valid_batch_size=1,
                optim="adam",
                optim_conf=dict(lr=0.001),
                scheduler="reducelronplateau",
                scheduler_conf=dict(patience=5, factor=0.5),
                grad_clip=5,
                best_model_criterion=[["valid", "loss", "min"]],
                keep_nbest_models=1,
            )
        )
    )
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
    assert (exp / "valid.loss.best.pth").exists() and (exp / "checkpoint.pth").exists()
    DAMSEPTask.main(cmd=cmd + ["--max_epoch", "2", "--resume", "true"])
    assert (
        torch.load(exp / "checkpoint.pth", map_location="cpu")["reporter"]["epoch"] == 2
    )
    dest = tmp_path / "inference"
    inference(
        argparse.Namespace(
            train_config=str(exp / "config.yaml"),
            model_file=str(exp / "valid.loss.best.pth"),
            wav_scp=tmp_path / "speech_mix.scp",
            output_dir=dest,
            key_file=None,
            device="cpu",
            normalize_output_wav=True,
        )
    )
    with np.load(dest / "ctf/example.npz") as data:
        assert data["ctf"].shape == (2, 257, 60) and np.isfinite(data["ctf"]).all()
    # Score the clean branch with the unchanged baseline scorer.
    score_main(
        [
            "--output_dir",
            str(tmp_path / "scores"),
            "--key_file",
            str(shape),
            "--ref_scp",
            str(tmp_path / "speech_ref1.scp"),
            "--ref_scp",
            str(tmp_path / "speech_ref2.scp"),
            "--inf_scp",
            str(dest / "clean/spk1.scp"),
            "--inf_scp",
            str(dest / "clean/spk2.scp"),
            "--ref_channel",
            "0",
        ]
    )
    for metric in ("SI_SNR", "SDR"):
        value = float((tmp_path / "scores" / f"{metric}_spk1").read_text().split()[1])
        assert np.isfinite(value)
