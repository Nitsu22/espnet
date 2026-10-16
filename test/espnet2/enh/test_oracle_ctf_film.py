"""Fair initialization, identity FiLM, conditioning gradients and inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator as Baseline
from espnet2.enh.separator.tflocoformer_separator_nocashe_ctf_film import TFLocoformerSeparator as FiLM
from espnet2.torch_utils.initialize import initialize

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh9_condition/local"
sys.path.insert(0, str(LOCAL))
import oracle_ctf_pipeline as pipeline
from check_oracle_ctf_readiness import verify_readiness
from unittest.mock import patch

SMALL = dict(num_spk=2, n_layers=1, emb_dim=8, num_groups=2, norm_type="rmsgroupnorm",
             n_heads=2, attention_dim=8, ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
             ffn_hidden_dim=[8, 8], conv1d_kernel=3, dropout=0.0)


class TestOracleFiLM(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.baseline = Baseline(17, **SMALL)
        initialize(self.baseline, "xavier_uniform")
        torch.manual_seed(0)
        self.film = FiLM(17, ctf_taps=3, random_ctf_swap=False, **SMALL)
        initialize(self.film, "xavier_uniform")
        self.speech = torch.randn(2, 9, 17, dtype=torch.complex64)
        self.ctf = torch.randn(2, 3, 2, 17, 2)

    def test_shared_initialization_and_identity_matches_baseline(self):
        self.baseline.eval()
        self.film.eval()
        for name, tensor in self.baseline.state_dict().items():
            torch.testing.assert_close(tensor, self.film.state_dict()[name], rtol=0, atol=0)
        baseline = self.baseline(self.speech, torch.tensor([9, 9]))[0]
        film = self.film(self.speech, torch.tensor([9, 9]), {"rir_ctf": self.ctf})[0]
        for a, b in zip(baseline, film):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(self.film.rir_films[0].gamma_proj.bias, torch.ones(8)))
        self.assertEqual(self.film.rir_films[0].beta_proj.weight.abs().sum(), 0)

    def test_two_updates_reach_ctf_encoder_and_condition_changes_outputs(self):
        self.film.train()
        optimizer = torch.optim.AdamW(self.film.parameters(), lr=1e-3)
        target = torch.randn_like(self.speech)
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            waves = self.film(self.speech, torch.tensor([9, 9]), {"rir_ctf": self.ctf})[0]
            loss = sum((wave - target).abs().square().mean() for wave in waves)
            loss.backward()
            for parameter in self.film.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            if update == 1:
                self.assertGreater(self.film.ctf_projection[0].weight.grad.abs().sum(), 0)
                self.assertGreater(self.film.rir_films[0].mha.in_proj_weight.grad.abs().sum(), 0)
            optimizer.step()
        self.film.eval()
        first = self.film(self.speech, torch.tensor([9, 9]), {"rir_ctf": self.ctf})[0]
        second = self.film(self.speech, torch.tensor([9, 9]), {"rir_ctf": self.ctf * 0})[0]
        self.assertGreater(sum((a-b).abs().sum() for a,b in zip(first,second)), 0)

    def test_unequal_lengths_ignore_padding_with_active_film(self):
        self.film.eval()
        with torch.no_grad():
            self.film.rir_films[0].beta_proj.weight.normal_(std=0.01)
        lengths = torch.tensor([5, 9])
        joint = self.film(self.speech, lengths, {"rir_ctf": self.ctf})[0]
        for i, length in enumerate(lengths.tolist()):
            single = self.film(self.speech[i:i+1,:length], torch.tensor([length]),
                               {"rir_ctf": self.ctf[i:i+1]})[0]
            for batch, one in zip(joint, single):
                torch.testing.assert_close(batch[i,:length], one[0])
                self.assertEqual(batch[i,length:].abs().sum(), 0)

    def test_full_config_only_changes_separator(self):
        folder = ROOT / "egs2/whamr/enh9_condition/conf/tuning"
        original = yaml.safe_load((folder / "train_enh_tflocoformer_s_nf8k_oracle_ctf_concat.yaml").read_text())
        film = yaml.safe_load((folder / "train_enh_tflocoformer_s_nf8k_oracle_ctf_film.yaml").read_text())
        expected = copy.deepcopy(original)
        expected["separator"] = "tflocoformer_nocashe_ctf_film_all"
        self.assertEqual(film, expected)
        baseline = Baseline(129, **{k:v for k,v in film["separator_conf"].items()
                                  if k not in ("ctf_taps", "random_ctf_swap")})
        conditioned = FiLM(129, **film["separator_conf"])
        self.assertEqual(sum(p.numel() for p in conditioned.parameters())
                         - sum(p.numel() for p in baseline.parameters()), 227232)

    def test_native_checkpoint_inference_normalizes_wav(self):
        from espnet2.tasks.enh_rir import EnhancementRIRCTFTask
        from espnet2.bin.enh_rir_ctf_inference import SeparateSpeech
        folder = ROOT / "egs2/whamr/enh9_condition/conf/tuning"
        cfg = yaml.safe_load((folder / "train_enh_tflocoformer_s_nf8k_oracle_ctf_film.yaml").read_text())
        cfg["separator_conf"] = dict(SMALL, ctf_taps=3, random_ctf_swap=False)
        cfg["encoder_conf"] = cfg["decoder_conf"] = dict(n_fft=32, hop_length=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            args = EnhancementRIRCTFTask.get_parser().parse_args(["--config", str(config), "--output_dir", str(path)])
            model = EnhancementRIRCTFTask.build_model(args)
            torch.save(model.state_dict(), path / "model.pth")
            inference = SeparateSpeech(train_config=str(config), model_file=str(path / "model.pth"),
                                       device="cpu", normalize_output_wav=True)
            outputs = inference(np.random.randn(1,64).astype("float32"),
                                self.ctf[:1].numpy())
            self.assertEqual(len(outputs), 2)
            for wave in outputs:
                self.assertEqual(wave.shape, (1,64))
                self.assertTrue(np.isfinite(wave).all())
                self.assertAlmostEqual(float(abs(wave).max()), 0.9, places=5)

    def test_automatic_jobs_keep_film_variant(self):
        with patch.object(pipeline, "VARIANT", "film"), \
                patch.object(pipeline.subprocess, "check_output", return_value="123\n") as call:
            pipeline.submit("train", dependency="99")
            args = call.call_args.args[0]
            self.assertEqual(args[args.index("--variant")+1], "film")
            self.assertEqual(args[args.index("-hold_jid")+1], "99")

    def test_readiness_requires_a_separate_film_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "conf/tuning/train_enh_tflocoformer_s_nf8k_oracle_ctf_film.yaml"
            manifest = root / "dump_oracle_ctf_nf8k/preparation.json"
            for path in (config, manifest):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("film")
            report = dict(status="passed", device="cuda", counts=dict(tr=20000, cv=5000, tt=3000),
                          commit="probe", config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                          manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            (root / "exp").mkdir()
            (root / "exp/oracle_ctf_cuda_check.json").write_text(json.dumps(report))
            with self.assertRaises(FileNotFoundError):
                verify_readiness(root, variant="film")
            (root / "exp/oracle_ctf_film_cuda_check.json").write_text(json.dumps(report))
            script = "unchanged model inference"
            (root / "run_oracle_ctf_concat.sh").write_text(script)
            with patch("check_oracle_ctf_readiness.subprocess.run"), \
                    patch("check_oracle_ctf_readiness.subprocess.check_output", side_effect=["",script]):
                self.assertEqual(verify_readiness(root, variant="film"), report)


if __name__ == "__main__":
    unittest.main()
