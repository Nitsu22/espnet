"""Joint CTF objective checks; runnable with unittest without pytest."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.enh import EnhancementTask
from espnet2.train.preprocessor import EnhPreprocessor
from espnet2.train.preprocessor_enh_ctf import EnhCTFPreprocessor

ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "egs2/whamr/enh_rir/conf/tuning"
    / "train_enh_tflocoformer_small_nocashe_ctf_joint_4gpu.yaml"
)


def build(tiny=True):
    args = EnhancementTask.get_parser().parse_args(
        [
            "--config",
            str(CONFIG),
            "--output_dir",
            "/tmp/ctf_joint_test",
        ]
    )
    if tiny:
        args.separator_conf.update(
            n_layers=1,
            emb_dim=8,
            num_groups=2,
            n_heads=2,
            attention_dim=8,
            ffn_hidden_dim=[8, 8],
            conv1d_kernel=2,
            ctf_taps=3,
        )
        args.encoder_conf = dict(n_fft=32, hop_length=8)
        args.decoder_conf = dict(n_fft=32, hop_length=8)
    return EnhancementTask.build_model(args), args


class TestJointCTF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model, self.args = build()

    def test_causal_convolution_and_gradients(self):
        signal = torch.randn(1, 2, 3, 7, dtype=torch.cfloat, requires_grad=True)
        filt = torch.randn(1, 2, 3, 4, dtype=torch.cfloat, requires_grad=True)
        actual = self.model.convolve_ctf(signal, filt)
        # Independent polynomial multiplication, including the full late tail.
        expected = torch.zeros(1, 2, 3, 10, dtype=torch.cfloat)
        for t in range(7):
            for lag in range(4):
                expected[..., t + lag] += signal[..., t] * filt[..., lag]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        gs, gf = torch.autograd.grad(actual[..., -1].abs().sum(), (signal, filt))
        self.assertGreater(gs[..., -1].abs().sum().item(), 0)
        self.assertGreater(gf[..., -1].abs().sum().item(), 0)

    def test_speech_branch_is_identical_to_baseline(self):
        # Removing only new head weights permits exact baseline state loading.
        from espnet2.enh.separator.tflocoformer_separator_nocashe import (
            TFLocoformerSeparator,
        )

        config = dict(self.args.separator_conf)
        config.pop("ctf_taps")
        baseline = TFLocoformerSeparator(17, **config)
        state = {
            k: v
            for k, v in self.model.separator.state_dict().items()
            if not k.startswith("ctf_")
        }
        baseline.load_state_dict(state)
        spectrum = torch.randn(2, 13, 17, dtype=torch.cfloat)
        lengths = torch.tensor([13, 10])
        reference = baseline(spectrum, lengths)[0]
        outputs, _, others = self.model.separator(spectrum, lengths)
        for a, b in zip(outputs, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(tuple(others["ctf"].shape), (2, 2, 17, 3))

    def test_padding_does_not_enter_ctf_pool(self):
        features = torch.randn(2, 8, 9, 17)
        lengths = torch.tensor([9, 5])
        expected = self.model.separator.estimate_ctf(features, lengths)
        features[1, :, 5:] = 10000
        actual = self.model.separator.estimate_ctf(features, lengths)
        torch.testing.assert_close(actual, expected)

    def test_identity_ctf_waveforms_and_backward(self):
        lengths = torch.tensor([256, 208])
        speech = [torch.randn(2, 256, requires_grad=True) for _ in range(2)]
        ctf = torch.zeros(2, 2, 17, 3, dtype=torch.cfloat)
        ctf[..., 0] = 1
        ctf.requires_grad_()
        output = self.model.reconstruct_reverb(speech, lengths, ctf)
        for a, b in zip(output, speech):
            for i, n in enumerate(lengths):
                torch.testing.assert_close(a[i, :n], b[i, :n], atol=1e-5, rtol=1e-5)
        # Probe is independent of the identity output, so SI-SNR has a gradient.
        loss = sum(
            self.model.loss_wrappers[0].criterion(torch.randn_like(x), x).mean()
            for x in output
        )
        grads = torch.autograd.grad(loss, [*speech, ctf])
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_joint_pit_is_not_independent_pit(self):
        refs = [torch.randn(1, 256) for _ in range(2)]
        lengths = torch.tensor([256])
        starts = torch.tensor([0])
        result = self.model.joint_loss(refs, refs[::-1], refs, refs, lengths, starts)
        total, clean, reverb, _, clean_pit = result
        torch.testing.assert_close(total, clean + reverb)
        perfect = torch.stack(
            [self.model.loss_wrappers[0].criterion(x, x) for x in refs]
        ).mean()
        torch.testing.assert_close(clean_pit, perfect)
        self.assertGreater(total.item(), 2 * perfect.item() + 10)
        # References are speaker pairs; swapping both must preserve the loss.
        swapped = self.model.joint_loss(
            refs, refs[::-1], refs[::-1], refs[::-1], lengths, starts
        )
        # With tied totals, the clean/reverb components can swap while their
        # sum remains invariant; there is no unique best permutation here.
        torch.testing.assert_close(result[0], swapped[0])

    def test_model_update_checkpoint_and_inference(self):
        lengths = torch.tensor([256, 208])
        inputs = dict(
            speech_mix=torch.randn(2, 256),
            speech_mix_lengths=lengths,
            reverb_crop_start=torch.tensor([[0], [10]]),
        )
        for s in (1, 2):
            inputs[f"speech_ref{s}"] = torch.randn(2, 256)
            inputs[f"speech_reverb{s}"] = torch.randn(2, 256)
        loss, stats, weight = self.model(**inputs)
        torch.testing.assert_close(loss, stats["loss_clean"] + stats["loss_reverb"])
        self.assertEqual(weight.item(), 2)
        loss.backward()
        for name in ("ctf_head.2.weight", "deconv.weight", "conv.0.weight"):
            grad = dict(self.model.separator.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        old = self.model.separator.ctf_head[2].weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(old, self.model.separator.ctf_head[2].weight))
        with tempfile.TemporaryDirectory() as directory:
            import yaml

            config = Path(directory) / "config.yaml"
            config.write_text(yaml.safe_dump(vars(self.args)))
            checkpoint = Path(directory) / "model.pth"
            torch.save(self.model.state_dict(), checkpoint)
            from espnet2.bin.enh_inference import SeparateSpeech

            separator = SeparateSpeech(config, checkpoint)
            output = separator(torch.randn(1, 256).numpy())
            self.assertEqual(len(output), 2)
            self.assertTrue(
                all(np.isfinite(x).all() and x.shape[-1] == 256 for x in output)
            )

    def test_reference_padding_and_crop_context_are_excluded(self):
        lengths = torch.tensor([180])
        starts = torch.tensor([40])
        cp = [torch.randn(1, 256) for _ in range(2)]
        rp = [torch.randn(1, 256) for _ in range(2)]
        cr = [torch.randn(1, 256) for _ in range(2)]
        rr = [torch.randn(1, 256) for _ in range(2)]
        expected = self.model.joint_loss(cp, rp, cr, rr, lengths, starts)[0]
        for x in cr:
            x[:, 180:] = 1000
        for x in rr:
            x[:, :40] = 1000
            x[:, 180:] = 1000
        torch.testing.assert_close(
            self.model.joint_loss(cp, rp, cr, rr, lengths, starts)[0], expected
        )

    def test_preprocessing_keeps_baseline_crop_and_reverb_alignment(self):
        kwargs = dict(
            num_spk=2, sample_rate=8000, speech_segment=32000, force_single_channel=True
        )
        baseline = EnhPreprocessor(train=True, **kwargs)
        joint = EnhCTFPreprocessor(train=True, **kwargs)
        x = np.linspace(0.01, 0.1, 40000, dtype=np.float32)
        signals = dict(
            speech_mix=np.stack([x, -x], 1),
            speech_ref1=np.stack([x * 0.2, -x], 1),
            speech_ref2=np.stack([x * 0.3, -x], 1),
        )
        np.random.seed(0)
        expected = baseline("test", copy.deepcopy(signals))
        signals.update(
            speech_reverb1=np.stack([x * 0.4, -x], 1),
            speech_reverb2=np.stack([x * 0.5, -x], 1),
        )
        np.random.seed(0)
        actual = joint("test", signals)
        for name, value in expected.items():
            np.testing.assert_array_equal(actual[name], value)
        start = int(actual["reverb_crop_start"][0])
        np.testing.assert_array_equal(
            actual["speech_reverb1"], x[start : start + 32000] * 0.4
        )
        self.assertIsInstance(
            EnhancementTask.build_preprocess_fn(self.args, True), EnhCTFPreprocessor
        )

    def test_production_configuration_forward_backward(self):
        model, args = build(tiny=False)
        lengths = torch.tensor([1024])
        inputs = dict(speech_mix=torch.randn(1, 1024), speech_mix_lengths=lengths)
        for s in (1, 2):
            inputs[f"speech_ref{s}"] = torch.randn(1, 1024)
            inputs[f"speech_reverb{s}"] = torch.randn(1, 1024)
        loss, _, _ = model(**inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.separator.parameters()
                if p.requires_grad
            )
        )


class TestPairedDump(unittest.TestCase):
    def test_verified_baseline_reuse_and_mismatch_rejection(self):
        path = ROOT / "egs2/whamr/enh_rir/local/prepare_ctf_joint_dump.py"
        spec = importlib.util.spec_from_file_location("prepare_joint", path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "enh1"
            source = baseline / "data"
            for split, (shape_split, _) in adapter.SPLITS.items():
                name = f"{split}_mix_both_reverb_min_8k"
                uid = "paired_utterance"
                src = source / name
                dst = baseline / "dump/raw" / name
                src.mkdir(parents=True)
                dst.mkdir(parents=True)
                clean1, clean2, rv1, rv2, noise = [
                    np.random.randn(256).astype("float32") * 0.01 for _ in range(5)
                ]
                # Baseline formatting clips PCM16; source FLOAT teachers do not.
                clean1[0] += 2.0
                rv1[0] += 2.0
                signals = {
                    "wav.scp": rv1 + rv2 + noise,
                    "spk1.scp": clean1,
                    "spk2.scp": clean2,
                    "spk1_reverb.scp": rv1,
                    "spk2_reverb.scp": rv2,
                    "noise1.scp": noise,
                }
                for file, signal in signals.items():
                    wav = src / (file + ".wav")
                    sf.write(wav, signal, 8000, subtype="FLOAT")
                    (src / file).write_text(f"{uid} {wav}\n")
                    if file in ("wav.scp", "spk1.scp", "spk2.scp"):
                        old = dst / (file + ".wav")
                        sf.write(old, signal, 8000, subtype="PCM_16")
                        (dst / file).write_text(f"{uid} {old}\n")
                if shape_split != "test":
                    stats = baseline / "exp/enh_stats_8k" / shape_split
                    stats.mkdir(parents=True)
                    for file in (
                        "speech_mix_shape",
                        "speech_ref1_shape",
                        "speech_ref2_shape",
                    ):
                        (stats / file).write_text(f"{uid} 256\n")
            output = root / "dump_joint"
            report = adapter.prepare(baseline, source, output, limit=1)
            self.assertEqual(report["splits"]["tr"]["count"], 1)
            self.assertGreater(
                report["splits"]["tr"]["source_samples_clipped_in_baseline"], 0
            )
            indexed = adapter.read_index(
                output / "raw/tr_mix_both_reverb_min_8k/wav.scp"
            )
            self.assertEqual(
                indexed[uid],
                str(baseline / "dump/raw/tr_mix_both_reverb_min_8k/wav.scp.wav"),
            )
            bad = source / "tr_mix_both_reverb_min_8k/spk1.scp.wav"
            sf.write(bad, np.ones(256) * 0.2, 8000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "not the baseline dataset"):
                adapter.prepare(baseline, source, root / "bad_dump", limit=1)
            self.assertFalse((root / "bad_dump").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
