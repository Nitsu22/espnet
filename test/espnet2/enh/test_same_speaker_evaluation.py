"""CPU-only contract tests; all temporary files remain within this repository."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "egs2/whamr/enh_rir/local/check_same_speaker_evaluation.py"
MODULE = importlib.util.spec_from_file_location("same_speaker_check", SCRIPT)
check = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(check)


class TestSameSpeakerEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".same_speaker_eval_test_", dir=REPO)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.exps = {}
        for model in check.MODELS:
            exp = self.root / model
            exp.mkdir()
            (exp / "config.yaml").write_text("num_spk: 2\n")
            (exp / "1epoch.pth").write_bytes(b"fixed checkpoint")
            (exp / "valid.loss.best.pth").symlink_to("1epoch.pth")
            self.exps[model] = exp
        self.audio = self.data / "speech.wav"
        sf.write(self.audio, np.sin(np.arange(256) * 0.1) * 0.1, 8000, subtype="FLOAT")
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            (self.data / name).write_text(f"utt1 {self.audio}\nutt2 {self.audio}\n")
        self.args = SimpleNamespace(output_dir=self.root / "evaluation", test_dir=self.data,
                                    baseline_exp=self.exps["baseline"],
                                    conditioned_exp=self.exps["conditioned"],
                                    checkpoint="valid.loss.best.pth", resume=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_checkpoint_resolved_and_resume_keeps_pinned_epoch(self):
        check.prepare(self.args)
        manifest = check.verify(self.args.output_dir)
        self.assertEqual(manifest["inputs"]["count"], 2)
        self.assertTrue(manifest["models"]["baseline"]["checkpoint"].endswith("1epoch.pth"))
        with self.assertRaises(ValueError):
            check.prepare(self.args)
        alias = self.exps["baseline"] / "valid.loss.best.pth"
        alias.unlink()
        (alias.parent / "2epoch.pth").write_bytes(b"different checkpoint")
        alias.symlink_to("2epoch.pth")
        self.args.resume = True
        check.prepare(self.args)
        self.assertEqual(check.verify(self.args.output_dir), manifest)

    def test_modified_checkpoint_rejected(self):
        check.prepare(self.args)
        (self.exps["conditioned"] / "1epoch.pth").write_bytes(b"modified")
        with self.assertRaises(ValueError):
            check.verify(self.args.output_dir)

    def test_keys_duplicates_missing_paths_and_audio_lengths_rejected(self):
        index = self.data / "spk2.scp"
        for content in (f"utt1 {self.audio}\n", f"utt1 {self.audio}\nutt1 {self.audio}\n",
                        "utt1 /nonexistent.wav\nutt2 /nonexistent.wav\n"):
            index.write_text(content)
            with self.assertRaises(ValueError):
                check.input_spec(self.data)
        short = self.data / "short.wav"
        sf.write(short, np.zeros(200), 8000, subtype="FLOAT")
        index.write_text(f"utt1 {short}\nutt2 {short}\n")
        with self.assertRaises(ValueError):
            check.input_spec(self.data)

    def test_training_or_data_output_rejected(self):
        for output in (self.data / "output", self.exps["baseline"] / "output"):
            self.args.output_dir = output
            with self.assertRaises(ValueError):
                check.prepare(self.args)

    def test_generated_transfer_checksums(self):
        generated = self.root / "generated"
        test = generated / "raw" / "tt"
        test.mkdir(parents=True)
        payload = generated / "audio.wav"
        payload.write_bytes(b"original wave bytes")
        (generated / "preparation.json").write_text(json.dumps({
            "output_hashes": {"audio.wav": check.digest(payload)}}))
        check.verify_generation_transfer(test)
        payload.write_bytes(b"corrupted wave bytes")
        with self.assertRaisesRegex(ValueError, "Transferred generation file changed"):
            check.verify_generation_transfer(test)

    def test_complete_output_and_summary(self):
        check.prepare(self.args)
        for model, number in (("baseline", 10), ("conditioned", 11)):
            out = self.args.output_dir / model
            (out / "score").mkdir(parents=True)
            for speaker in (1, 2):
                (out / f"spk{speaker}.scp").write_text(f"utt1 {self.audio}\nutt2 {self.audio}\n")
                for metric in check.METRICS:
                    (out / "score" / f"{metric}_spk{speaker}").write_text(
                        f"utt1 {number}\nutt2 {number}\n")
            check.check_inference(self.args.output_dir, model)
            check.scores(self.args.output_dir, model)
        check.summary(self.args.output_dir)
        self.assertIn("SI_SNR,10.0,11.0,1.0", (self.args.output_dir / "summary.csv").read_text())
        result = subprocess.run([
            "bash", str(SCRIPT.with_name("evaluate_same_speaker_test.sh")),
            "--test_dir", str(self.data), "--baseline_exp", str(self.exps["baseline"]),
            "--conditioned_exp", str(self.exps["conditioned"]),
            "--output_dir", str(self.args.output_dir), "--resume", "true",
            "--stage", "8", "--stop_stage", "8", "--python", sys.executable,
        ], capture_output=True, text=True, check=True)
        self.assertIn("Verified existing complete conditioned scores", result.stdout)
        self.assertFalse((self.args.output_dir / ".tmp").exists())
        score = self.args.output_dir / "conditioned/score/SI_SNR_spk1"
        score.write_text("utt1 nan\nutt2 10\n")
        with self.assertRaises(ValueError):
            check.scores(self.args.output_dir, "conditioned")
        score.write_text("utt1 10\n")
        with self.assertRaises(ValueError):
            check.scores(self.args.output_dir, "conditioned")


if __name__ == "__main__":
    unittest.main()
