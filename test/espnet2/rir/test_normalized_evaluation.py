"""Keep corrected evaluation separate from original waves and cached scores."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "egs2/whamr/enh9_condition/local"))
import oracle_ctf_pipeline as pipeline
from prepare_normalized_evaluation import prepare


class TestNormalizedEvaluation(unittest.TestCase):
    def source(self, root):
        source = root / "trained"
        (source / ".pipeline").mkdir(parents=True)
        for name in ("config.yaml", "checkpoint.pth", "train.log", "120epoch.pth"):
            (source / name).write_text(name)
        (source / "valid.loss.best.pth").symlink_to("120epoch.pth")
        (source / "RESULTS.md").write_text("original results")
        pipeline.save(source / ".pipeline/inference.done.json", dict(
            model_sha256=pipeline.digest(source / "valid.loss.best.pth")))
        pipeline.save(source / ".pipeline/run.json", dict(
            config_sha256="template-hash-is-not-the-expanded-config-hash"))
        return source

    def test_reuses_exact_weights_without_old_outputs_or_score_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            output = source / "normalized"
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            record = prepare(source, output)
            self.assertTrue(record["normalize_output_wav"])
            self.assertEqual((output / "valid.loss.best.pth").resolve(), source / "120epoch.pth")
            self.assertFalse((output / "inference_cv").exists())
            self.assertFalse((output / ".pipeline/score_work").exists())
            self.assertEqual(record, prepare(source, output))
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)
            with patch.object(pipeline, "EXP", output), patch.object(pipeline.subprocess, "run") as run:
                pipeline.run_stage(7)
                args = run.call_args.args[0]
                self.assertEqual(args[args.index("--enh_exp") + 1], str(output))

    def test_changed_model_or_config_is_rejected(self):
        for changed in ("120epoch.pth", "config.yaml"):
            with tempfile.TemporaryDirectory() as directory:
                source = self.source(Path(directory))
                output = source / "normalized"
                prepare(source, output)
                (source / changed).write_text("changed")
                with self.assertRaises(RuntimeError):
                    prepare(source, output)

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            output = source / "normalized"
            output.mkdir()
            with self.assertRaises(RuntimeError):
                prepare(source, output)
            with self.assertRaises(ValueError):
                prepare(source, source)


if __name__ == "__main__":
    unittest.main()
