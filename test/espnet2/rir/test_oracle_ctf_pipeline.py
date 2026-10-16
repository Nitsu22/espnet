"""Autonomous continuation, unchanged-runtime readiness and complete scoring."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh9_condition/local"
sys.path.insert(0, str(LOCAL))
import oracle_ctf_pipeline as pipeline
from check_oracle_ctf_readiness import verify_readiness


class TestOraclePipeline(unittest.TestCase):
    def test_submission_resources_dependencies_and_points(self):
        with patch.object(pipeline.subprocess, "check_output", return_value="123\n") as call:
            entry = pipeline.submit("train", dependency="100")
            self.assertEqual(entry["upper_points"], 3.84)
            self.assertEqual(entry["resource"], "gpu_1")
            args = call.call_args.args[0]
            self.assertEqual(args[args.index("-hold_jid") + 1], "100")
            self.assertEqual(args[args.index("h_rt=24:00:00")], "h_rt=24:00:00")
            for phase, points in (("infer", .64), ("score", .144), ("control", .002)):
                self.assertEqual(pipeline.submit(phase)["upper_points"], points)

    def test_transitions_and_fail_closed_retries(self):
        choose = pipeline.next_phase
        self.assertEqual(choose([], None, None, None, 0, False, False, False), "train")
        previous = dict(phase="train", epoch=3)
        self.assertEqual(choose([], previous, 44, 137, 4, False, False, False), "train")
        self.assertEqual(choose([], previous, 44, 137, 4, True, False, False), "infer")
        for failed, status, epoch in ((44, 137, 3), (0, 1, 4), (100, 137, 4), (0, 0, 4)):
            with self.assertRaises(RuntimeError):
                choose([], previous, failed, status, epoch, False, False, False)
        self.assertEqual(choose([], dict(phase="infer"), 0, 0, 4, True, True, False), "score")
        self.assertIsNone(choose([], dict(phase="score"), 0, 0, 4, True, True, True))
        with self.assertRaises(RuntimeError):
            choose([], dict(phase="infer"), 0, 0, 4, True, False, False)
        with self.assertRaises(RuntimeError):
            choose([dict(phase="train")] * 12, None, None, None, 4, False, False, False)

    def test_controller_dependencies_journal_and_no_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            serial = iter(("101", "102", "103", "104"))
            def submit(phase, dependency=None, previous=None):
                return dict(job=next(serial), phase=phase, dependency=dependency, previous=previous)
            with patch.object(pipeline, "STATE", state), \
                    patch.object(pipeline, "training_state", return_value=(0, False)), \
                    patch.object(pipeline, "submit", side_effect=submit), \
                    patch.dict(os.environ, {"JOB_ID": "100"}):
                pipeline.control()
                jobs = json.loads((state / "jobs.json").read_text())
                self.assertEqual(jobs[0]["dependency"], "100")
                self.assertEqual(jobs[0]["controller"]["dependency"], "101")
                with self.assertRaises(RuntimeError):
                    pipeline.control()
                with patch.object(pipeline, "accounting", return_value=(44, 137)), \
                        patch.object(pipeline, "training_state", return_value=(1, False)):
                    pipeline.control("101")
                jobs = json.loads((state / "jobs.json").read_text())
                self.assertTrue(jobs[0]["handled"])
                self.assertEqual(jobs[1]["epoch"], 1)
                pipeline.control("101")
                self.assertEqual(len(json.loads((state / "jobs.json").read_text())), 2)

    def test_native_checkpoint_completion_and_best_model(self):
        import torch
        import yaml
        from espnet2.train.reporter import Reporter
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            reporter = Reporter()
            for epoch, loss in enumerate((1.0, 2.0, 3.0, 4.0), 1):
                reporter.set_epoch(epoch)
                with reporter.observe("valid") as sub:
                    sub.register(dict(loss=loss))
                    sub.next()
            torch.save(dict(reporter=reporter.state_dict()), exp / "checkpoint.pth")
            (exp / "config.yaml").write_text(yaml.safe_dump(dict(
                max_epoch=150, patience=2, early_stopping_criterion=["valid", "loss", "min"])))
            (exp / "train.log").write_text("Normal training\n")
            (exp / "1epoch.pth").write_bytes(b"weights")
            (exp / "valid.loss.best.pth").symlink_to("1epoch.pth")
            with patch.object(pipeline, "EXP", exp):
                self.assertEqual(pipeline.training_state(), (4, True))
                (exp / "valid.loss.best.pth").unlink()
                with self.assertRaises(RuntimeError):
                    pipeline.training_state()
                (exp / "train.log").write_text("The gradients at all steps are invalid")
                with self.assertRaises(RuntimeError):
                    pipeline.training_state()

    def test_readiness_accepts_output_normalization_and_rejects_runtime_or_data_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "egs2/whamr/enh9_condition"
            config = recipe / "conf/tuning/train_enh_tflocoformer_s_nf8k_oracle_ctf_concat.yaml"
            manifest = recipe / "dump_oracle_ctf_nf8k/preparation.json"
            source = root / "espnet2/runtime.py"
            run = recipe / "run_oracle_ctf_concat.sh"
            for path in (config, manifest, source, run):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            inference = (
                '        "${python}" -m espnet2.bin.enh_rir_ctf_inference \\\n'
                '            --ngpu "${ngpu}" --fs 8000 --batch_size 1 \\\n'
                '            --model_file "${enh_exp}/${inference_model}"\n'
            )
            prefix = "training\n" + inference
            scoring = "    if (( stage <= 8 && stop_stage >= 8 )); then\n"
            run.write_text(prefix + scoring + "old scoring\n")
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "model")
            report = dict(status="passed", device="cuda", counts=dict(tr=20000, cv=5000, tt=3000),
                          commit=git("rev-parse", "HEAD"),
                          config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                          manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            marker = recipe / "exp/oracle_ctf_cuda_check.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps(report))
            automation = recipe / "local/oracle_ctf_pipeline.py"
            automation.parent.mkdir()
            automation.write_text("automation only")
            git("add", str(automation.relative_to(root)))
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "automation")
            self.assertEqual(verify_readiness(recipe), report)
            run.write_text(prefix + scoring + "new scoring\n")
            self.assertEqual(verify_readiness(recipe), report)
            normalized = prefix.replace(
                "--batch_size 1", "--batch_size 1 --normalize_output_wav true")
            run.write_text(normalized + scoring + "new scoring\n")
            self.assertEqual(verify_readiness(recipe), report)
            for changed in (
                normalized.replace("training\n", "changed training\n"),
                normalized.replace("--fs 8000", "--fs 16000"),
                normalized.replace("--batch_size 1", "--batch_size 2"),
                normalized.replace("--normalize_output_wav true", "--normalize_output_wav false"),
                normalized.replace("--normalize_output_wav true", "--normalize_output_wav true --normalize_output_wav false"),
                normalized.replace("${inference_model}", "other_model.pth"),
                prefix.replace("training\n", "training --normalize_output_wav true\n"),
            ):
                with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                    run.write_text(changed + scoring + "new scoring\n")
                    verify_readiness(recipe)
            run.write_text(normalized + scoring + "new scoring\n")
            source.write_text("changed runtime")
            with self.assertRaises(RuntimeError):
                verify_readiness(recipe)
            source.write_text("original")
            manifest.write_text("changed data")
            with self.assertRaises(RuntimeError):
                verify_readiness(recipe)

    def test_score_parallel_merge_and_reuse_complete_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                exp, state = Path("exp/model"), Path("exp/model/.pipeline")
                state.mkdir(parents=True)
                (exp / "valid.loss.best.pth").write_bytes(b"model")
                pipeline.save(state / "inference.done.json", dict(
                    model_sha256=pipeline.digest(exp / "valid.loss.best.pth")))
                for split in ("cv", "tt"):
                    folder = Path(f"dump_oracle_ctf_nf8k/raw/{split}_mix_clean_reverb_min_8k")
                    folder.mkdir(parents=True)
                    (folder / "wav.scp").write_text("".join(f"u{i} /wav{i}\n" for i in range(16)))
                calls = []
                def scoring(command, **kwargs):
                    if command[1] == "local/check_oracle_ctf_inputs.py":
                        return
                    calls.append(command)
                    destination = Path(command[command.index("--output_dir") + 1])
                    destination.mkdir(parents=True, exist_ok=True)
                    ids = Path(command[command.index("--key_file") + 1]).read_text().splitlines()
                    for metric in pipeline.METRICS:
                        for speaker in (1, 2):
                            (destination / f"{metric}_spk{speaker}").write_text(
                                "".join(f"{uid} {int(uid[1:])}\n" for uid in ids))
                    for speaker in (1, 2):
                        (destination / f"wav_spk{speaker}").write_text(
                            "".join(f"{uid} /{uid}.wav\n" for uid in ids))
                with patch.object(pipeline, "EXP", exp), patch.object(pipeline, "STATE", state), \
                        patch.object(pipeline.subprocess, "run", side_effect=scoring):
                    pipeline.score()
                    self.assertEqual(len(calls), 16)
                    result = json.loads((state / "score.done.json").read_text())
                    self.assertEqual(result["metrics"]["tt"]["SI_SNR"], 7.5)
                    self.assertEqual(len(pipeline.read_scp(exp / "inference_cv/score/SI_SNR_spk1")), 16)
                    (state / "score_work/cv/0/SI_SNR_spk1").write_text("u0 0\n")
                    pipeline.score()
                    self.assertEqual(len(calls), 17)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
