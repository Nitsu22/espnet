import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[3] / "egs2/whamr/rir_2spk/local/evaluate_tfdamsep_noisy_ls8k.py"
spec = importlib.util.spec_from_file_location("noisy_ls_eval", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_id_mapping_preserves_source_gains_and_rejects_wrong_speakers():
    assert module.canonical_uid("050_051_050a050m_1.6347_051c010x_-1.6347_reverb") == "050a050m_1.6347_051c010x_-1.6347"
    for uid in ["050_052_050a050m_1.6347_051c010x_-1.6347_reverb", "bad", "050_051_050a050m_1.6347_051c010x_-1.6347"]:
        with pytest.raises(ValueError):
            module.canonical_uid(uid)


def test_paired_bootstrap_keeps_pairing_and_difference_direction():
    a = np.arange(20, dtype=float)
    actual = module.paired_bootstrap(a, a+0.25, repetitions=500)
    assert actual["difference_b_minus_a"] == 0.25
    np.testing.assert_allclose(actual["ci95"], [0.25, 0.25])
    np.testing.assert_allclose(module.paired_bootstrap(a, a, repetitions=500)["ci95"], [0, 0])
    with pytest.raises(ValueError):
        module.paired_bootstrap(a, [np.nan]*20)


def test_read_index_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "wav.scp"
    path.write_text("utt /one\nutt /two\n")
    with pytest.raises(ValueError):
        module.read_index(path)
