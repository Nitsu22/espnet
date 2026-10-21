"""Noisy Baseline pairing and configuration parity; no GPU or source mutation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / 'egs2/whamr/enh_rir'
sys.path.insert(0, str(RECIPE / 'local'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, RECIPE / 'local' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = RECIPE / 'conf/tuning/train_enh_tflocoformer_s_whamr8k_bimamba_ctf_concat.yaml'


class NoisyInputsTests(unittest.TestCase):
    def fixture(self, root):
        baseline = root / 'baseline'
        dump = baseline / 'dump'
        for split in ('tr', 'cv', 'tt'):
            folder = dump / 'raw' / f'{split}_mix_both_reverb_min_8k'
            folder.mkdir(parents=True)
            for field in ('wav', 'spk1', 'spk2'):
                path = folder / f'{field}.wav'
                sf.write(path, np.full((1024, 2), .02), 8000)
                (folder / f'{field}.scp').write_text(f'{split}_uid {path.relative_to(baseline)}\n')
        link = root / 'linked_dump'
        link.symlink_to(dump, target_is_directory=True)
        output = root / 'prepared'
        prepare = load('prepare_bimamba_baseline8k_inputs').prepare
        prepare(link, baseline, output, expected_counts=(('tr', 1), ('cv', 1), ('tt', 1)),
                condition='whamr')
        return baseline, link, output

    def test_noisy_exact_paths_idempotence_and_condition_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, link, output = self.fixture(root)
            before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            prepare = load('prepare_bimamba_baseline8k_inputs').prepare
            prepare(link, baseline, output, expected_counts=(('tr', 1), ('cv', 1), ('tt', 1)),
                    condition='whamr')
            self.assertEqual(before, {str(p.relative_to(output)): p.read_bytes()
                                     for p in output.rglob('*') if p.is_file()})
            prepared = (output / 'raw/tr_mix_both_reverb_min_8k/wav.scp').read_text()
            self.assertIn(str((baseline / 'dump/raw/tr_mix_both_reverb_min_8k/wav.wav').resolve()), prepared)
            self.assertEqual(json.loads((output / 'preparation.json').read_text())['spec']['condition'], 'whamr')
            with self.assertRaises(ValueError):
                prepare(link, baseline, output, condition='unknown')
            with self.assertRaises(FileNotFoundError):
                prepare(link, baseline, output)  # Never silently treat noisy as NF.

    def test_saved_baseline_audit_and_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, inputs = self.fixture(root)
            new = yaml.safe_load(CONFIG.read_text())
            checkpoint = root / 'source.pth'
            checkpoint.write_bytes(b'fixture weights')
            new['model_conf']['bimamba_checkpoint'] = str(checkpoint)
            new['model_conf']['bimamba_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(new))
            base = yaml.safe_load((ROOT / 'egs2/whamr/enh7_baseline/conf/tuning/'
                                   'train_enh_tflocoformer_s_whamr_8k.yaml').read_text())
            base.update(ngpu=1, accum_grad=1, fold_length=[200000] * 3)
            baseline = root / 'baseline.yaml'
            baseline.write_text(yaml.safe_dump(base))
            stats = root / 'stats'
            for split, phase in (('tr', 'train'), ('cv', 'valid')):
                (stats / phase).mkdir(parents=True)
                for field in ('speech_mix', 'speech_ref1', 'speech_ref2'):
                    (stats / phase / f'{field}_shape').write_text(f'{split}_uid 1024\n')
            check = load('check_whamr8k_baseline_match').check
            self.assertEqual(check(config, baseline, stats, inputs)['status'], 'passed')
            base['accum_grad'] = 2
            baseline.write_text(yaml.safe_dump(base))
            with self.assertRaisesRegex(ValueError, 'accum-grad'):
                check(config, baseline, stats, inputs)
            base['accum_grad'] = 1
            baseline.write_text(yaml.safe_dump(base))
            (stats / 'valid/speech_ref2_shape').write_text('cv_uid 1023\n')
            with self.assertRaisesRegex(ValueError, 'IDs/lengths'):
                check(config, baseline, stats, inputs)

    def test_noisy_config_only_changes_predictor_weights(self):
        new = yaml.safe_load(CONFIG.read_text())
        original = yaml.safe_load((RECIPE / 'conf/tuning/'
            'train_enh_tflocoformer_s_nf8k_bimamba_ctf_concat.yaml').read_text())
        for key in ('bimamba_checkpoint', 'bimamba_sha256'):
            original['model_conf'][key] = new['model_conf'][key]
        self.assertEqual(new, original)
        self.assertEqual(new['model_conf']['bimamba_sha256'],
                         'ef163313e47febb3cccb4366bfb1fe45b769fe0a550b08da3e053b55573c11e7')

    def test_each_shell_script_parses(self):
        paths = [RECIPE / 'run_whamr8k_bimamba_ctf_concat_s.sh',
                 RECIPE / 'run_nf16k_bimamba_ctf_film_all_s.sh',
                 RECIPE / 'qsub/prepare_whamr8k_bimamba_concat_s.sh',
                 RECIPE / 'qsub/train_whamr8k_bimamba_ctf_concat_s_nodeq.sh']
        for path in paths:
            subprocess.run(['bash', '-n', str(path)], check=True)


if __name__ == '__main__':
    unittest.main()
