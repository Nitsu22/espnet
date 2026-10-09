"""Absolute-path indexes must preserve the exact existing Baseline WAVs."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


LOCAL = Path(__file__).resolve().parents[3] / "egs2/whamr/enh_rir/local"
sys.path.insert(0, str(LOCAL))
spec = importlib.util.spec_from_file_location("prepare_baseline8k", LOCAL / "prepare_bimamba_baseline8k_inputs.py")
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


class InputPreparationTests(unittest.TestCase):
    def test_exact_paths_shapes_and_repeatability(self):
        with tempfile.TemporaryDirectory(prefix="bimamba_baseline8k_") as temporary:
            root = Path(temporary)
            baseline = root / "enh7_baseline"
            baseline.mkdir()
            source = baseline / "dump_clean"
            expected = (("tr", 2), ("cv", 2), ("tt", 1))
            for split, count in expected:
                folder = source / "raw" / f"{split}_mix_clean_reverb_min_8k"
                folder.mkdir(parents=True)
                entries = []
                for i in range(count):
                    uid = f"{split}_speaker_utterance_{i}_reverb"
                    path = folder / f"{uid}.wav"
                    sf.write(path, np.random.default_rng(i).normal(0, .1, (512+i, 2)), 8000)
                    entries.append(f"{uid} {path.relative_to(baseline)}\n")
                for name in ("wav.scp", "spk1.scp", "spk2.scp"):
                    (folder / name).write_text("".join(entries))
            link = root / "dump_nf_baseline_8k"
            link.symlink_to(source, target_is_directory=True)
            output = root / "indexes"
            report = prepare_module.prepare(link, baseline, output, expected_counts=expected)
            before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            self.assertEqual(report["counts"], dict(expected))
            for split, _ in expected:
                relative = Path("raw") / f"{split}_mix_clean_reverb_min_8k"
                maps = prepare_module.read_scp(output / relative / "wav.scp")
                lengths = prepare_module.read_scp(output / relative / "utt2num_samples")
                for uid, value in maps.items():
                    self.assertEqual(Path(value), (source / relative / f"{uid}.wav").resolve())
                    self.assertEqual(sf.info(value).frames, int(lengths[uid]))
            prepare_module.prepare(link, baseline, output, expected_counts=expected)
            after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            # A modified prepared index must fail instead of silently being reused.
            first = output / "raw/tr_mix_clean_reverb_min_8k/wav.scp"
            first.write_text(first.read_text() + "other /missing.wav\n")
            with self.assertRaisesRegex(ValueError, "index changed"):
                prepare_module.prepare(link, baseline, output, expected_counts=expected)


if __name__ == "__main__":
    unittest.main()
