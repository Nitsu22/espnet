"""Split-S checks, also runnable with unittest without installing pytest."""

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator
from espnet2.enh.separator.tflocoformer_separator_split import (
    FeatureFusion,
    FeatureSplit,
    TFLocoformerSplitSeparator,
)
from espnet2.torch_utils.initialize import initialize

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "egs2/whamr/enh8_split"


class TestTFLocoformerSplit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def small_separator(self):
        return TFLocoformerSplitSeparator(
            9,
            emb_dim=8,
            norm_type="rmsgroupnorm",
            num_groups=2,
            attention_dim=8,
            n_heads=2,
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=[8, 8],
            conv1d_kernel=4,
        )

    def test_split_and_fusion_preserve_example_pairs(self):
        split = FeatureSplit(4)
        fusion = FeatureFusion(4)
        features = torch.randn(3, 4, 5, 7, requires_grad=True)
        branches = split(features)
        self.assertEqual(tuple(branches.shape), (6, 4, 5, 7))
        for example in range(3):
            torch.testing.assert_close(
                branches[2 * example : 2 * example + 2],
                split(features[example : example + 1]),
            )
        self.assertGreater((branches[0] - branches[1]).abs().sum().item(), 0)
        # Select branch 0 or 1 independently: fusion must pair branches from
        # the same example, rather than adjacent examples or split halves.
        with torch.no_grad():
            fusion.projection.bias.zero_()
            for branch in range(2):
                fusion.projection.weight.zero_()
                for channel in range(4):
                    fusion.projection.weight[channel, branch * 4 + channel, 0, 0] = 1
                torch.testing.assert_close(
                    fusion(branches, features), features + branches[branch::2]
                )
            fusion.projection.weight.zero_()
            torch.testing.assert_close(fusion(branches, features), features)
        with self.assertRaises(ValueError):
            fusion(branches[:-1], features)

    def test_block_order_batch_isolation_and_gradients(self):
        model = self.small_separator().double().eval()
        initialize(model, "xavier_uniform")
        seen = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, args: seen.append(tuple(args[0].shape))
            )
            for block in model.blocks
        ]
        x = torch.randn(3, 7, 9, dtype=torch.complex128, requires_grad=True)
        lengths = torch.tensor([7, 7, 7])
        outputs, actual_lengths, others = model(x, lengths)
        for handle in handles:
            handle.remove()
        self.assertEqual(seen, [(3, 8, 7, 9), (6, 8, 7, 9)] * 2)
        torch.testing.assert_close(actual_lengths, lengths)
        self.assertEqual(others, {})
        for output in outputs:
            self.assertEqual(tuple(output.shape), (3, 7, 9))
            self.assertTrue(torch.isfinite(output).all())
        for example in range(3):
            individual, _, _ = model(x[example : example + 1], lengths[:1])
            for together, alone in zip(outputs, individual):
                torch.testing.assert_close(together[example : example + 1], alone)
        isolated_gradient = torch.autograd.grad(
            outputs[0][0].abs().square().mean(), x, retain_graph=True
        )[0]
        self.assertEqual(isolated_gradient[1:].abs().sum().item(), 0)
        sum(output.abs().square().mean() for output in outputs).backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_mono_input_and_checkpoint(self):
        model = self.small_separator().eval()
        x = torch.randn(2, 5, 9, dtype=torch.complex64)
        lengths = torch.tensor([5, 3])
        expected, _, _ = model(x, lengths)
        clone = copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(), strict=True)
        actual, actual_lengths, _ = clone(x.unsqueeze(2), lengths)
        for original, restored in zip(expected, actual):
            torch.testing.assert_close(original, restored)
        torch.testing.assert_close(actual_lengths, lengths)
        with self.assertRaises(ValueError):
            model(x.unsqueeze(2).expand(-1, -1, 2, -1), lengths)
        with self.assertRaises(ValueError):
            TFLocoformerSplitSeparator(9, n_layers=6)

    def test_baseline_config_and_original_blocks(self):
        for condition in ("whamr", "nf"):
            filename = f"train_enh_tflocoformer_split_s_{condition}_8k.yaml"
            config = yaml.safe_load((RECIPE / "conf/tuning" / filename).read_text())
            baseline_file = (
                RECIPE.parent
                / "enh7_baseline/conf/tuning"
                / f"train_enh_tflocoformer_s_{condition}_8k.yaml"
            )
            baseline_config = yaml.safe_load(baseline_file.read_text())
            self.assertEqual(config.pop("separator"), "tflocoformer_split")
            baseline_config.pop("separator")
            self.assertEqual(config, baseline_config)
        baseline = TFLocoformerSeparator(129, **config["separator_conf"])
        split = TFLocoformerSplitSeparator(129, **config["separator_conf"])
        for name, parameter in baseline.named_parameters():
            self.assertEqual(
                parameter.shape, dict(split.named_parameters())[name].shape
            )
        self.assertEqual(len({id(block) for block in split.blocks}), 4)
        self.assertIsNot(split.splits[0], split.splits[1])
        self.assertIsNot(split.fusions[0], split.fusions[1])
        extra = sum(p.numel() for p in split.parameters()) - sum(
            p.numel() for p in baseline.parameters()
        )
        self.assertEqual(extra, 185664)

    def test_espnet_waveform_update_and_reload(self):
        from espnet2.tasks.enh import EnhancementTask

        config = RECIPE / "conf/tuning/train_enh_tflocoformer_split_s_whamr_8k.yaml"
        with tempfile.TemporaryDirectory() as directory:
            args = EnhancementTask.get_parser().parse_args(
                ["--config", str(config), "--output_dir", directory]
            )
            model = EnhancementTask.build_model(args)
            self.assertIsInstance(model.separator, TFLocoformerSplitSeparator)
            refs = [torch.randn(2, 512) for _ in range(2)]
            mixture = refs[0] + refs[1] + 0.01 * torch.randn(2, 512)
            lengths = torch.tensor([512, 512])
            loss, _, _ = model(
                speech_mix=mixture,
                speech_mix_lengths=lengths,
                speech_ref1=refs[0],
                speech_ref2=refs[1],
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.optim.AdamW(model.parameters(), **args.optim_conf).step()
            config_file = Path(directory) / "config.yaml"
            config_file.write_text(yaml.safe_dump(vars(args)))
            checkpoint = Path(directory) / "model.pth"
            torch.save(model.state_dict(), checkpoint)
            restored, _ = EnhancementTask.build_model_from_file(
                str(config_file), str(checkpoint), device="cpu"
            )
            model.eval()
            restored.eval()
            with torch.no_grad():
                encoded, frames = model.encoder(mixture, lengths)
                expected, _, _ = model.separator(encoded, frames)
                actual, _, _ = restored.separator(encoded, frames)
                for original, reloaded in zip(expected, actual):
                    torch.testing.assert_close(original, reloaded)
                    audio, audio_lengths = restored.decoder(reloaded, lengths)
                    self.assertEqual(audio.shape, mixture.shape)
                    self.assertTrue(torch.isfinite(audio).all())
                    torch.testing.assert_close(audio_lengths, lengths)

    def test_recipe_routes_conditions_and_keeps_outputs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "utils").symlink_to(RECIPE / "utils")
            for dump in ("dump", "dump_clean"):
                (work / f"{dump}_target/raw").mkdir(parents=True)
                (work / dump).symlink_to(f"{dump}_target")
            (work / "enh.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            runner = RECIPE / "run_tflocoformer_split_s.sh"
            for condition, tag, subset in (
                ("whamr", "whamr", "mix_both_reverb_min_8k"),
                ("nf_whamr", "nf", "mix_clean_reverb_min_8k"),
            ):
                process = subprocess.run(
                    [
                        "bash",
                        str(runner),
                        "--condition",
                        condition,
                        "--stage",
                        "7",
                        "--stop_stage",
                        "7",
                    ],
                    cwd=work,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                arguments = process.stdout.splitlines()
                self.assertIn(f"tr_{subset}", arguments)
                self.assertIn(
                    f"conf/tuning/train_enh_tflocoformer_split_s_{tag}_8k.yaml",
                    arguments,
                )
                self.assertIn(
                    f"exp/enh_train_tflocoformer_split_s_{tag}_8k_1gpu_batch4",
                    arguments,
                )
            rejected = subprocess.run(
                ["bash", str(runner), "--stage", "5"],
                cwd=work,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_shared_recipe_bootstrap(self):
        process = subprocess.run(
            ["bash", "./enh.sh", "--help"],
            cwd=RECIPE,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Usage:", process.stdout + process.stderr)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
