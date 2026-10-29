"""Reject mismatched teachers/noise and keep noisy jobs/data isolated."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import tempfile
import unittest
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh9_condition/local"
sys.path.insert(0, str(LOCAL))
import oracle_ctf_pipeline as pipeline
from oracle_condition import settings
from prepare_oracle_ctf_dump import load_wave
from prepare_whamr_oracle_dump import compare_config, prepare, validate, verify_example



def fixtures(tmp_path):
    t = np.arange(512)
    clean = .1 * np.sin(t / 13)
    noise = .03 * np.cos(t / 7)
    paths = {}
    for name, wave in (("wav.scp", clean + noise), ("spk1.scp", clean / 2),
                       ("spk2.scp", clean / 3), ("clean", clean), ("noise", noise)):
        paths[name] = tmp_path / (name + ".wav")
        sf.write(paths[name], wave, 8000, subtype="PCM_16")
    source = {name: load_wave(paths[name])[1] for name in ("wav.scp", "spk1.scp", "spk2.scp")}
    return paths, source



class TestWHAMROracle(unittest.TestCase):
    def test_noisy_condition_reaches_held_jobs_and_stages(self):
        paths = settings("whamr")
        assert "whamr8k" in str(paths["exp"])
        assert paths["suffix"] == "mix_both_reverb_min_8k"
        with self.assertRaises(ValueError):
            settings("whamr", "film")
        with patch.object(pipeline, "CONDITION", "whamr"), \
                patch.object(pipeline.subprocess, "check_output", return_value="42\n") as submit:
            pipeline.submit("train", dependency="41")
            args = submit.call_args.args[0]
            assert args[args.index("--condition") + 1] == "whamr"
            assert args[args.index("-hold_jid") + 1] == "41"
        with patch.object(pipeline, "CONDITION", "whamr"), \
                patch.object(pipeline.subprocess, "run") as run:
            pipeline.run_stage(7)
            args = run.call_args.args[0]
            assert args[args.index("--condition") + 1] == "whamr"

    def test_correspondence_rejects_wrong_speaker_or_noise(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        tmp_path = Path(scratch.name)
        paths, source = fixtures(tmp_path)
        _, report = verify_example("u", paths, paths["clean"], source, paths["noise"])
        assert report["maximum_noise_residual"] <= 3 / 32768
        changed = copy.deepcopy(source)
        changed["spk1.scp"]["sha256"] = source["spk2.scp"]["sha256"]
        with self.assertRaisesRegex(ValueError, "teacher mismatch"):
            verify_example("u", paths, paths["clean"], changed, paths["noise"])
        with self.assertRaisesRegex(ValueError, "correspondence differs"):
            verify_example("u", paths, paths["clean"], source, paths["spk1.scp"])

    def test_publish_reuses_features_and_detects_audio_or_index_changes(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        tmp_path = Path(scratch.name)
        paths, source_audio = fixtures(tmp_path)
        source = tmp_path / "source"
        nf = source / "raw/tr_mix_clean_reverb_min_8k"
        nf.mkdir(parents=True)
        features = source / "features/tr"
        features.mkdir(parents=True)
        np.save(features / "u.npy", np.zeros((60, 2, 129, 2), dtype=np.float32))
        (nf / "wav.scp").write_text(f"u {paths['clean']}\n")
        (nf / "rir_ctf.scp").write_text(f"u {features / 'u.npy'}\n")
        native = dict(counts={"tr": 1}, baseline_audio={"tr": {"u": source_audio}},
                      spec=dict(method="direct_reverb_sweep_ridge_ls", config={}, shape=[60, 2, 129, 2]))
        (source / "preparation.json").write_text(json.dumps(native))
        baseline = tmp_path / "baseline"
        folder = baseline / "dump/raw/tr_mix_both_reverb_min_8k"
        folder.mkdir(parents=True)
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            (folder / name).write_text(f"u {paths[name]}\n")
        (folder / "noise1.scp").write_text(f"u {paths['noise']}\n")
        stats = tmp_path / "stats/train"
        stats.mkdir(parents=True)
        for name in ("speech_mix", "speech_ref1", "speech_ref2"):
            (stats / (name + "_shape")).write_text("u 512,2\n")
        config = ROOT / "egs2/whamr/enh9_condition/conf/tuning/train_enh_tflocoformer_s_whamr8k_oracle_ctf_concat.yaml"
        saved = yaml.safe_load(config.read_text())
        saved.update(separator="tflocoformer_nocashe", model_conf={"normalize_variance": True},
                     ngpu=1, accum_grad=1, fold_length=[200000] * 3)
        del saved["separator_conf"]["ctf_taps"], saved["separator_conf"]["random_ctf_swap"]
        baseline_config = tmp_path / "baseline.yaml"
        baseline_config.write_text(yaml.safe_dump(saved))
        compare_config(config, baseline_config)
        saved["batch_size"] = 8
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text(yaml.safe_dump(saved))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            compare_config(config, bad_config)
        output = tmp_path / "published"
        with patch("prepare_whamr_oracle_dump.COUNTS", {"tr": 1}), \
                patch("prepare_whamr_oracle_dump.validate_prepared_dump", return_value=native):
            report = prepare(source, baseline, baseline_config, stats.parent, config, output,
                             runtime_roots=(ROOT, ROOT))
            assert report["counts"] == {"tr": 1}
            assert (output / "features").is_symlink()
            assert (output / "features/tr/u.npy").resolve() == features / "u.npy"
            assert validate(output)["status"] == "complete"
            audio = output / "raw/tr_mix_both_reverb_min_8k/wav.scp"
            original = audio.read_text()
            audio.write_text(original.replace("u ", "wrong "))
            with self.assertRaisesRegex(ValueError, "index changed"):
                validate(output)
            audio.write_text(original)
            paths["wav.scp"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "audio changed"):
                validate(output)
