"""CPU checks for oracle CTF prefixes, gradients, padding, and real inference."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from espnet2.enh.separator.tflocoformer_separator_nocashe import (
    TFLocoformerSeparator as BaselineSeparator,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_ctf_concat import (
    TFLocoformerSeparator,
)
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask


ROOT = Path(__file__).resolve().parents[3]
BASELINE = (
    ROOT
    / "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_8k.yaml"
)
SMALL_TRUNK = dict(
    num_spk=2,
    n_layers=1,
    emb_dim=8,
    norm_type="rmsgroupnorm",
    num_groups=2,
    n_heads=2,
    attention_dim=8,
    ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
    ffn_hidden_dim=[8, 8],
    conv1d_kernel=3,
    dropout=0.0,
)


def build_model(tiny=True):
    args = EnhancementRIRCTFTask.get_parser().parse_args(
        ["--config", str(BASELINE), "--output_dir", "/tmp/oracle_ctf_test"]
    )
    args.separator = "tflocoformer_nocashe_ctf_concat"
    if tiny:
        args.separator_conf = dict(SMALL_TRUNK, ctf_taps=3)
        args.encoder_conf = dict(n_fft=32, hop_length=8)
        args.decoder_conf = dict(n_fft=32, hop_length=8)
    else:
        args.separator_conf = dict(args.separator_conf, ctf_taps=60)
    args.model_conf = dict(normalize_variance=True, rir_condition_type="rir_ctf")
    return EnhancementRIRCTFTask.build_model(args), args


class OracleCTFConcatTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.separator = TFLocoformerSeparator(17, ctf_taps=3, **SMALL_TRUNK)
        self.spectrum = torch.randn(2, 9, 17, dtype=torch.cfloat)
        self.ctf = torch.randn(2, 3, 2, 17, 2)
        self.condition = {
            "rir_ctf": self.ctf,
            "rir_ctf_lengths": torch.tensor([3, 3]),
        }

    def test_baseline_trunk_initialization_is_preserved(self):
        torch.manual_seed(100)
        baseline = BaselineSeparator(17, **SMALL_TRUNK)
        torch.manual_seed(100)
        conditioned = TFLocoformerSeparator(17, ctf_taps=3, **SMALL_TRUNK)
        base_state, state = baseline.state_dict(), conditioned.state_dict()
        self.assertEqual(
            set(state) - set(base_state),
            {"ctf_projection.weight", "ctf_projection.bias"},
        )
        for name, weight in base_state.items():
            torch.testing.assert_close(state[name], weight, rtol=0, atol=0)

    def test_actual_task_xavier_initialization_matches_dai8(self):
        # Model constructors consume randomness before Task.initialize. Compare
        # the complete build path, including the native DAI-8 four-block model.
        for tiny in (True, False):
            with self.subTest(tiny=tiny):
                torch.manual_seed(0)
                conditioned, args = build_model(tiny=tiny)
                baseline_args = copy.deepcopy(args)
                baseline_args.separator = "tflocoformer_nocashe"
                baseline_args.separator_conf.pop("ctf_taps")
                baseline_args.model_conf.pop("rir_condition_type")
                torch.manual_seed(0)
                baseline = EnhancementTask.build_model(baseline_args)
                state = conditioned.state_dict()
                for name, weight in baseline.state_dict().items():
                    torch.testing.assert_close(state[name], weight, rtol=0, atol=0)

    def test_prefix_order_and_crop_before_output_convolution(self):
        self.separator.eval()
        captured = {}

        def capture_prefix(module, inputs):
            captured["prefix"] = inputs[0].detach().clone()

        def capture_deconv(module, inputs):
            captured["deconv_frames"] = inputs[0].shape[2]

        handles = [
            self.separator.blocks[0].register_forward_pre_hook(capture_prefix),
            self.separator.deconv.register_forward_pre_hook(capture_deconv),
        ]
        try:
            outputs, lengths, _ = self.separator(
                self.spectrum, torch.tensor([9, 9]), self.condition
            )
        finally:
            for handle in handles:
                handle.remove()
        speech = torch.stack((self.spectrum.real, self.spectrum.imag), dim=1)
        prompts = [
            self.separator.ctf_projection(self.ctf[:, :, i].permute(0, 3, 1, 2))
            for i in range(2)
        ]
        expected = torch.cat([*prompts, self.separator.conv(speech)], dim=2)
        torch.testing.assert_close(captured["prefix"], expected, rtol=2e-5, atol=2e-7)
        self.assertEqual(captured["deconv_frames"], 9)
        self.assertTrue(all(output.shape == (2, 9, 17) for output in outputs))
        torch.testing.assert_close(lengths, torch.tensor([9, 9]))

    def test_condition_gradients_and_effect_on_audio(self):
        self.separator.eval()
        condition = self.ctf.clone().requires_grad_()
        outputs = self.separator(
            self.spectrum, torch.tensor([9, 9]), {"rir_ctf": condition}
        )[0]
        sum(output.abs().square().mean() for output in outputs).backward()
        for gradient in (
            condition.grad,
            self.separator.ctf_projection.weight.grad,
            self.separator.conv[0].weight.grad,
            self.separator.deconv.weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        changed = self.separator(
            self.spectrum,
            torch.tensor([9, 9]),
            {"rir_ctf": torch.zeros_like(condition)},
        )[0]
        self.assertGreater(
            sum((a - b).abs().sum().item() for a, b in zip(outputs, changed)), 0
        )

    def test_padding_is_excluded_and_matches_individual_forward(self):
        self.separator.eval()
        spectrum = self.spectrum.clone().requires_grad_()
        lengths = torch.tensor([9, 5])
        outputs = self.separator(spectrum, lengths, self.condition)[0]
        altered = self.spectrum.clone()
        altered[1, 5:] = 10000 + 10000j
        changed = self.separator(altered, lengths, self.condition)[0]
        individual = self.separator(
            self.spectrum[1:2, :5], torch.tensor([5]), {"rir_ctf": self.ctf[1:2]}
        )[0]
        for output, other, single in zip(outputs, changed, individual):
            torch.testing.assert_close(output, other, rtol=0, atol=0)
            torch.testing.assert_close(output[1:2, :5], single, rtol=0, atol=0)
            self.assertEqual(output[1, 5:].abs().sum().item(), 0)
        sum(output.abs().sum() for output in outputs).backward()
        self.assertEqual(spectrum.grad[1, 5:].abs().sum().item(), 0)
        self.assertGreater(spectrum.grad[1, :5].abs().sum().item(), 0)

    def test_training_swaps_per_utterance_and_evaluation_is_deterministic(self):
        self.separator.train()
        with mock.patch("torch.rand", return_value=torch.tensor([0.1, 0.9])):
            swapped = self.separator._get_ctf(self.condition, 2, 17, self.spectrum.real)
        torch.testing.assert_close(swapped[0], self.ctf[0].flip(1), rtol=0, atol=0)
        torch.testing.assert_close(swapped[1], self.ctf[1], rtol=0, atol=0)
        self.separator.eval()
        random_state = torch.get_rng_state().clone()
        first = self.separator(self.spectrum, torch.tensor([9, 9]), self.condition)[0]
        second = self.separator(self.spectrum, torch.tensor([9, 9]), self.condition)[0]
        torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Time concatenation is order-sensitive; both source permutations must
        # still be finite valid inputs, while training augments this ambiguity.
        reverse = self.separator(
            self.spectrum, torch.tensor([9, 9]), {"rir_ctf": self.ctf.flip(2)}
        )[0]
        self.assertTrue(all(torch.isfinite(output).all() for output in reverse))

    def test_malformed_conditions_and_lengths_fail(self):
        for ctf in (
            self.ctf[:, :2],
            self.ctf[..., :16, :],
            self.ctf[..., 0],
            self.ctf.long(),
        ):
            with self.subTest(shape=tuple(ctf.shape), dtype=ctf.dtype):
                with self.assertRaises(ValueError):
                    self.separator(
                        self.spectrum, torch.tensor([9, 9]), {"rir_ctf": ctf}
                    )
        bad = self.ctf.clone()
        bad[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.separator(self.spectrum, torch.tensor([9, 9]), {"rir_ctf": bad})
        with self.assertRaisesRegex(ValueError, "fixed ctf_taps"):
            self.separator(
                self.spectrum,
                torch.tensor([9, 9]),
                dict(self.condition, rir_ctf_lengths=torch.tensor([3, 2])),
            )
        for lengths in (
            torch.tensor([9, 0]),
            torch.tensor([9, 10]),
            torch.tensor([9.0, 9.0]),
        ):
            with self.assertRaises(ValueError):
                self.separator(self.spectrum, lengths, self.condition)
        with self.assertRaisesRegex(ValueError, "required"):
            self.separator(self.spectrum, torch.tensor([9, 9]))

    def test_real_task_pit_training_and_checkpoint_inference(self):
        model, args = build_model()
        speech = torch.randn(2, 256)
        condition = self.ctf.clone().requires_grad_()
        inputs = dict(
            speech_mix=speech,
            speech_mix_lengths=torch.tensor([256, 208]),
            speech_ref1=torch.randn_like(speech),
            speech_ref2=torch.randn_like(speech),
            rir_ctf=condition,
            rir_ctf_lengths=torch.tensor([3, 3]),
        )
        self.assertEqual(model.loss_wrappers[0].__class__.__name__, "PITSolver")
        loss, stats, weight = model(**inputs)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(weight.item(), 2)
        self.assertIn("si_snr_loss", stats)
        loss.backward()
        self.assertTrue(torch.isfinite(condition.grad).all())
        self.assertGreater(condition.grad.abs().sum().item(), 0)
        old = model.separator.ctf_projection.weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        self.assertFalse(torch.equal(old, model.separator.ctf_projection.weight))

        # The existing RIR inference entry point substitutes globals in its base
        # module. Restore them afterwards to isolate other enhancement tests.
        import espnet2.bin.enh_inference as base_inference

        original_task = base_inference.EnhancementTask
        original_inference = base_inference.SeparateSpeech
        try:
            from espnet2.bin.enh_rir_ctf_inference import SeparateSpeech

            base_inference.EnhancementTask = EnhancementRIRCTFTask
            with tempfile.TemporaryDirectory(prefix="oracle_ctf_concat_") as directory:
                config = Path(directory) / "config.yaml"
                checkpoint = Path(directory) / "model.pth"
                config.write_text(yaml.safe_dump(vars(args)))
                torch.save(model.state_dict(), checkpoint)
                inference = SeparateSpeech(
                    config, checkpoint, normalize_output_wav=False
                )
                waveform, ctf = speech[:1].numpy(), self.ctf[:1].numpy()
                first = inference(waveform, ctf)
                second = inference(waveform, ctf)
                self.assertEqual(len(first), 2)
                for a, b in zip(first, second):
                    self.assertEqual(a.shape, (1, 256))
                    self.assertTrue(np.isfinite(a).all())
                    np.testing.assert_array_equal(a, b)
                # Checkpoint restores the shared projection, not just the trunk.
                torch.testing.assert_close(
                    inference.enh_model.separator.ctf_projection.weight,
                    model.separator.ctf_projection.weight,
                    rtol=0,
                    atol=0,
                )
        finally:
            base_inference.EnhancementTask = original_task
            base_inference.SeparateSpeech = original_inference

    def test_preprocessor_leaves_ctf_taps_and_speaker_axes_intact(self):
        _, args = build_model()
        args = copy.deepcopy(args)
        args.speech_segment = 8000
        preprocess = EnhancementRIRCTFTask.build_preprocess_fn(args, train=True)
        ctf = self.ctf[0].numpy()
        sample = {
            "speech_mix": np.random.randn(16000).astype(np.float32),
            "speech_ref1": np.random.randn(16000).astype(np.float32),
            "speech_ref2": np.random.randn(16000).astype(np.float32),
            "rir_ctf": ctf.copy(),
        }
        result = preprocess("test_utterance", sample)
        self.assertEqual(result["speech_mix"].shape, (8000,))
        np.testing.assert_array_equal(result["rir_ctf"], ctf)


if __name__ == "__main__":
    unittest.main()
