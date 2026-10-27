"""Export and scoring tests, including genuine tiny-Mamba checkpoint inference."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from espnet2.bin.ctf_dereverb_inference import (
    CLEAN_BRANCHES,
    _read_mono,
    get_parser,
    inference,
    read_wav_scp,
    score_branches,
)
from espnet2.tasks.enh import EnhancementTask


class TestCTFScoring(unittest.TestCase):
    def test_final_assignment_is_shared_and_scale_is_preserved(self):
        generator = np.random.default_rng(35)
        references = generator.normal(size=(2, 2048))
        mixture = references.sum(0)
        branches = dict(
            clean_deconv=references,
            clean_final=2 * references[::-1],
        )
        assignment, metrics = score_branches(branches, references, mixture)
        self.assertEqual(assignment, (1, 0))
        self.assertGreater(metrics["clean_final"]["si_sdr_db"], 100)
        # Independent PIT for the prior would incorrectly make this large too.
        self.assertLess(metrics["clean_deconv"]["si_sdr_db"], -10)
        self.assertAlmostEqual(metrics["clean_final"]["snr_db"], 0.0, places=10)
        self.assertIsNone(metrics["clean_final"]["stoi"])
        self.assertGreater(metrics["clean_final"]["si_sdri_db"], 100)

    def test_silence_metrics_are_finite(self):
        silent = np.zeros((2, 128))
        _, metrics = score_branches(
            {name: silent for name in CLEAN_BRANCHES}, silent, silent[0]
        )
        for branch in metrics.values():
            for key in ("si_sdr_db", "si_sdri_db", "snr_db"):
                self.assertTrue(np.isfinite(branch[key]))

    def test_cli_explicit_cg_override(self):
        args = get_parser().parse_args([
            "--train_config", "config.yaml", "--model_file", "model.pth",
            "--wav_scp", "wav.scp", "--output_dir", "output",
            "--enable_deconvolution", "true", "--gamma", "0.01",
            "--iterations", "16",
        ])
        self.assertTrue(args.enable_deconvolution)
        self.assertEqual(args.gamma, .01)
        self.assertEqual(args.iterations, 16)

    def test_scp_resolution_and_unsafe_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listing = root / "lists"
            listing.mkdir()
            scp = listing / "wav.scp"
            scp.write_text("utterance ../audio/example.wav\n")
            self.assertEqual(read_wav_scp(scp)["utterance"], root / "audio/example.wav")
            scp.write_text("utterance audio/example.wav\n")
            self.assertEqual(
                read_wav_scp(scp, root)["utterance"], root / "audio/example.wav"
            )
            for text in (
                "../escape input.wav\n", "a/b input.wav\n",
                "duplicate a.wav\nduplicate b.wav\n", "pipe cat input.wav |\n",
            ):
                scp.write_text(text)
                with self.assertRaises(ValueError):
                    read_wav_scp(scp)

    def test_actual_sample_rate_and_left_microphone_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            sf.write(path, np.zeros(160), 16000)
            with self.assertRaisesRegex(ValueError, "8000"):
                _read_mono(path)
            left = np.linspace(-2, 2, 160, dtype=np.float32)
            sf.write(path, np.stack([left, -left], axis=-1), 8000, subtype="FLOAT")
            np.testing.assert_array_equal(_read_mono(path), left)
            sf.write(path, np.zeros((160, 3)), 8000)
            with self.assertRaisesRegex(ValueError, "mono or stereo"):
                _read_mono(path)


@unittest.skipUnless(importlib.util.find_spec("mamba_ssm"), "mamba_ssm unavailable")
class TestCTFInferenceCheckpoint(unittest.TestCase):
    def test_joint_checkpoint_exports_and_explicit_cg_override(self):
        torch.manual_seed(43)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = EnhancementTask.get_parser().parse_args([
                "--output_dir", str(root / "training"),
            ])
            options.encoder = options.decoder = "stft"
            options.encoder_conf = dict(n_fft=64, hop_length=16)
            options.decoder_conf = dict(n_fft=64, hop_length=16)
            options.separator = "tflocoformer_ctf_dereverb"
            options.separator_conf = dict(
                n_layers=1, emb_dim=8, attention_dim=8, n_heads=2,
                num_groups=2, ffn_hidden_dim=[8, 8], conv1d_kernel=4,
                ctf_taps=4, ctf_hidden_dim=8, mamba_d_state=2,
                mamba_d_conv=2, mamba_backend="torch", use_deconvolution=True,
            )
            options.model_conf = dict(normalize_variance=True)
            options.criterions = [dict(
                name="si_snr", conf=dict(eps=1e-7), wrapper="pit",
                wrapper_conf=dict(weight=1.0, independent_perm=True),
            )]
            network = EnhancementTask.build_model(options).eval()
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(vars(options)))
            checkpoint = root / "checkpoint.pth"
            torch.save(network.state_dict(), checkpoint)
            audio = root / "audio"
            audio.mkdir()
            listing = root / "lists"
            listing.mkdir()
            scp = listing / "wav.scp"
            generator = np.random.default_rng(47)
            sources = {
                "first": generator.normal(size=256).astype(np.float32) * 2,
                "second": generator.normal(size=192).astype(np.float32),
            }
            lines = []
            reference_lines = [[], []]
            for uid, samples in sources.items():
                # Actual WHAMR indexes can point to stereo files. The right
                # microphone must never alter left-microphone predictions.
                stored = np.stack([samples, -samples], axis=-1)
                sf.write(audio / f"{uid}.wav", stored, 8000, subtype="FLOAT")
                lines.append(f"{uid} ../audio/{uid}.wav\n")
                for speaker, weight in enumerate((.3, .7)):
                    path = audio / f"{uid}.ref{speaker + 1}.wav"
                    sf.write(path, stored * weight, 8000, subtype="FLOAT")
                    reference_lines[speaker].append(f"{uid} {path}\n")
            scp.write_text("".join(lines))
            refs = [listing / "ref1.scp", listing / "ref2.scp"]
            for path, entries in zip(refs, reference_lines):
                path.write_text("".join(entries))
            destination = root / "joint"
            summary = inference(
                config, checkpoint, scp, destination,
                ref1_scp=refs[0], ref2_scp=refs[1], write_reverb=True,
            )
            self.assertEqual(summary["num_utterances"], 2)
            self.assertEqual(summary["branches"], list(CLEAN_BRANCHES) + ["reverb"])
            self.assertTrue(summary["deconvolution"]["enabled"])
            self.assertTrue(summary["ctf_available"])
            self.assertTrue(summary["scored"])
            self.assertIn("available", summary["stoi"])
            self.assertEqual(len(list(destination.rglob("*.wav"))), 12)
            for uid, samples in sources.items():
                tensor = torch.from_numpy(samples).unsqueeze(0)
                with torch.no_grad():
                    expected = network.predict_all(tensor)
                for branch in summary["branches"]:
                    for speaker in range(2):
                        path = destination / branch / f"spk{speaker + 1}" / f"{uid}.wav"
                        saved, rate = sf.read(path, dtype="float32")
                        self.assertEqual(rate, 8000)
                        self.assertEqual(sf.info(path).subtype, "FLOAT")
                        self.assertEqual(saved.shape, samples.shape)
                        np.testing.assert_array_equal(
                            saved, expected["waveforms"][branch][0, speaker].numpy()
                        )
                    entries = (
                        destination / branch / "spk1.scp"
                    ).read_text().splitlines()
                    self.assertEqual(len(entries), 2)
                saved_ctf = np.load(
                    destination / "ctf" / f"{uid}.npy", allow_pickle=False
                )
                self.assertEqual(saved_ctf.shape, (2, 33, 4))
                self.assertTrue(np.iscomplexobj(saved_ctf))
                np.testing.assert_array_equal(saved_ctf, expected["ctf"][0].numpy())
                self.assertEqual(summary["utterances"][uid]["samples"], len(samples))
                self.assertEqual(set(summary["utterances"][uid]["metrics"]),
                                 set(summary["branches"]))
            self.assertEqual(
                json.loads((destination / "summary.json").read_text()), summary
            )
            with self.assertRaises(FileExistsError):
                inference(config, checkpoint, scp, destination)
            # Override numerical CG settings on the same jointly trained architecture.
            inverted = root / "inverted"
            summary = inference(
                config, checkpoint, scp, inverted,
                enable_deconvolution=True, gamma=.23, iterations=3,
            )
            self.assertEqual(summary["deconvolution"],
                             dict(enabled=True, gamma=.23, iterations=3))
            self.assertFalse(summary["scored"])
            self.assertNotIn("reverb", summary["branches"])
            network.separator.use_deconvolution = True
            network.separator.cg_gamma = .23
            network.separator.cg_iterations = 3
            tensor = torch.from_numpy(sources["first"]).unsqueeze(0)
            with torch.no_grad():
                expected = network.predict_all(tensor)
            for branch in CLEAN_BRANCHES:
                path = inverted / branch / "spk1" / "first.wav"
                saved, _ = sf.read(path, dtype="float32")
                np.testing.assert_array_equal(
                    saved, expected["waveforms"][branch][0, 0].numpy()
                )
            self.assertGreater(
                (expected["waveforms"]["reverb"]
                 - expected["waveforms"]["clean_deconv"]).abs().max(), 1e-5,
            )


if __name__ == "__main__":
    unittest.main()
