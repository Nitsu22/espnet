"""Constant delimiters preserve speech frames, initialization and CTF gradients."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from espnet2.enh.separator.tflocoformer_separator_nocashe_ctf_concat import (
    TFLocoformerSeparator,
)
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask
from test_oracle_ctf_concat import SMALL_TRUNK, build_model


class OracleCTFGlueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.spectrum = torch.randn(2, 9, 17, dtype=torch.cfloat)
        self.ctf = torch.randn(2, 3, 2, 17, 2)

    def separator(self, **kwargs):
        return TFLocoformerSeparator(17, ctf_taps=3, **SMALL_TRUNK, **kwargs).eval()

    def test_exact_post_projection_glue_and_crop_for_zero_and_nonzero_values(self):
        for value in (0.0, 5.0):
            model = self.separator(glue_frames=4, glue_value=value)
            captured = {}
            def prefix(module, inputs):
                captured["prefix"] = inputs[0].detach().clone()
            def speech_only(module, inputs):
                captured["speech_frames"] = inputs[0].shape[2]
            handles = [model.blocks[0].register_forward_pre_hook(prefix),
                       model.deconv.register_forward_pre_hook(speech_only)]
            try:
                waves, lengths, _ = model(self.spectrum, torch.tensor([9, 9]),
                                          {"rir_ctf": self.ctf})
            finally:
                for handle in handles:
                    handle.remove()
            speech = model.conv(torch.stack((self.spectrum.real, self.spectrum.imag), dim=1))
            prompts = [model.ctf_projection(self.ctf[:, :, s].permute(0, 3, 1, 2)) for s in (0, 1)]
            glue = speech.new_full((2, 8, 4, 17), value)
            expected = torch.cat((prompts[0], glue, prompts[1], glue, speech), dim=2)
            torch.testing.assert_close(captured["prefix"], expected)
            self.assertEqual(captured["prefix"].shape[2], 23)
            self.assertEqual(captured["speech_frames"], 9)
            self.assertTrue(all(w.shape == (2, 9, 17) for w in waves))
            torch.testing.assert_close(lengths, torch.tensor([9, 9]))

    def test_default_disabled_is_bitwise_compatible_and_adds_no_parameters(self):
        model = self.separator()
        explicit = self.separator(glue_frames=0, glue_value=5.0)
        explicit.load_state_dict(model.state_dict(), strict=True)
        first = model(self.spectrum, torch.tensor([9, 9]), {"rir_ctf": self.ctf})[0]
        second = explicit(self.spectrum, torch.tensor([9, 9]), {"rir_ctf": self.ctf})[0]
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Native S task initialization must remain identical, including Xavier.
        torch.manual_seed(0)
        original, args = build_model(tiny=False)
        args = copy.deepcopy(args)
        args.separator_conf.update(glue_frames=4, glue_value=0.0)
        torch.manual_seed(0)
        glued = EnhancementRIRCTFTask.build_model(args)
        self.assertEqual(sum(p.numel() for p in glued.parameters()), 5125540)
        self.assertEqual(set(original.state_dict()), set(glued.state_dict()))
        for name, weight in original.state_dict().items():
            torch.testing.assert_close(weight, glued.state_dict()[name], rtol=0, atol=0)

    def test_padding_exclusion_and_finite_condition_gradients(self):
        model = self.separator(glue_frames=4)
        spectrum = self.spectrum.clone().requires_grad_()
        condition = self.ctf.clone().requires_grad_()
        outputs = model(spectrum, torch.tensor([9, 5]), {"rir_ctf": condition})[0]
        changed = self.spectrum.clone()
        changed[1, 5:] = 10000 + 10000j
        padded = model(changed, torch.tensor([9, 5]), {"rir_ctf": condition})[0]
        single = model(self.spectrum[1:2, :5], torch.tensor([5]),
                       {"rir_ctf": condition[1:2]})[0]
        for output, other, individual in zip(outputs, padded, single):
            torch.testing.assert_close(output, other, rtol=0, atol=0)
            torch.testing.assert_close(output[1:2, :5], individual, rtol=0, atol=0)
            self.assertEqual(output[1, 5:].abs().sum().item(), 0)
        sum(w.abs().square().mean() for w in outputs).backward()
        self.assertEqual(spectrum.grad[1, 5:].abs().sum().item(), 0)
        for gradient in (condition.grad, model.ctf_projection.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_native_loss_backward_and_checkpoint_inference_restore_glue(self):
        _, args = build_model()
        args.separator_conf.update(glue_frames=4, glue_value=0.0)
        model = EnhancementRIRCTFTask.build_model(args)
        speech = torch.randn(2, 256)
        loss, _, _ = model(speech_mix=speech, speech_mix_lengths=torch.tensor([256, 208]),
                           speech_ref1=torch.randn_like(speech), speech_ref2=torch.randn_like(speech),
                           rir_ctf=self.ctf, rir_ctf_lengths=torch.tensor([3, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.separator.ctf_projection.weight.grad.abs().sum().item(), 0)
        import espnet2.bin.enh_inference as base
        original_task, original_inference = base.EnhancementTask, base.SeparateSpeech
        try:
            from espnet2.bin.enh_rir_ctf_inference import SeparateSpeech
            base.EnhancementTask = EnhancementRIRCTFTask
            with tempfile.TemporaryDirectory(prefix="oracle_glue_") as directory:
                config, weights = Path(directory) / "config.yaml", Path(directory) / "model.pth"
                config.write_text(yaml.safe_dump(vars(args)))
                torch.save(model.state_dict(), weights)
                inference = SeparateSpeech(config, weights, normalize_output_wav=True)
                self.assertEqual(inference.enh_model.separator.glue_frames, 4)
                self.assertEqual(inference.enh_model.separator.glue_value, 0.0)
                waves = inference(speech[:1].numpy(), self.ctf[:1].numpy())
                self.assertEqual(len(waves), 2)
                self.assertTrue(all(w.shape == (1, 256) and np.isfinite(w).all() for w in waves))
        finally:
            base.EnhancementTask, base.SeparateSpeech = original_task, original_inference

    def test_malformed_glue_configuration_fails(self):
        for frames in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "glue_frames"):
                self.separator(glue_frames=frames)
        for value in (float("nan"), float("inf"), 1e300, True, "0"):
            with self.assertRaisesRegex(ValueError, "glue_value"):
                self.separator(glue_value=value)
