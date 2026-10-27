"""Check speaker assignment for CA predictions conditioned on true CD RIRs."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "oracle_cd_ca_evaluation",
    ROOT / "egs2/whamr/rir_2spk/local/evaluate_oracle_cd_ca.py",
)
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def fixture(costs):
    costs = np.asarray(costs, dtype=float)
    # Each column identifies the true CD used during reconstruction. Its CA
    # reference must retain the same identity under either PIT assignment.
    candidates = np.empty((2, 2, 3))
    for pred in range(2):
        for ref in range(2):
            candidates[pred, ref] = [pred, ref, costs[pred, ref]]
    references = np.array([[0, 10], [1, 20]])
    calls = []

    def score_pair(pred, ref, *settings):
        assert int(pred[1]) == int(ref[0]), "CD and CA reference columns differ"
        assert settings == (8000, 16000, "peak", "peak", 2.5, None, False)
        calls.append((int(pred[0]), int(ref[0])))
        return {
            "rmse": float(pred[2]),
            # Favor the opposite permutation for this diagnostic. It must not
            # affect the assignment chosen with full-RIR RMSE.
            "corr": float(pred[0] != ref[0]),
            "rt60_err": float(10 * pred[0] + ref[0]),
        }

    return candidates, references, score_pair, calls


def selected_rows(result):
    _, permutation, matrix = result
    return [matrix[pred][ref] for pred, ref in enumerate(permutation)]


def test_reference_conditioning_scores_all_four_pairs_without_mutation():
    candidates, references, scorer, calls = fixture([[0.9, 0.1], [1.2, 0.2]])
    before_candidates, before_references = candidates.copy(), references.copy()
    EVALUATE.select_pairs(candidates, references, scorer)
    assert sorted(calls) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    np.testing.assert_array_equal(candidates, before_candidates)
    np.testing.assert_array_equal(references, before_references)


def test_one_global_permutation_is_selected_by_rmse_for_all_metrics():
    # Both independent row minima select reference 1. A valid assignment must
    # instead include each reference once; identity has the lower total RMSE.
    candidates, references, scorer, _ = fixture([[0.9, 0.1], [1.2, 0.2]])
    result = EVALUATE.select_pairs(candidates, references, scorer)
    assert result[:2] == (0, (0, 1))
    rows = selected_rows(result)
    np.testing.assert_allclose([row["rmse"] for row in rows], [0.9, 0.2])
    assert [row["corr"] for row in rows] == [0, 0]
    assert [row["rt60_err"] for row in rows] == [0, 11]
    assert result[2][0][1]["corr"] == result[2][1][0]["corr"] == 1


def test_swapping_predicted_slots_preserves_reference_ordered_pairs():
    candidates, references, scorer, _ = fixture([[0.1, 2.0], [3.0, 0.2]])
    original = EVALUATE.select_pairs(candidates, references, scorer)
    swapped = EVALUATE.select_pairs(candidates[::-1], references, scorer)
    assert original[:2] == (0, (0, 1))
    assert swapped[:2] == (1, (1, 0))
    assert selected_rows(swapped)[::-1] == selected_rows(original)


def test_swapping_true_cd_and_ca_together_preserves_physical_pairs():
    candidates, references, scorer, _ = fixture([[0.1, 2.0], [3.0, 0.2]])
    original = EVALUATE.select_pairs(candidates, references, scorer)
    # Reorder the reconstructed CD columns together with the raw CA references.
    swapped = EVALUATE.select_pairs(candidates[:, ::-1], references[::-1], scorer)
    assert original[:2] == (0, (0, 1))
    assert swapped[:2] == (1, (1, 0))
    assert selected_rows(swapped) == selected_rows(original)


def test_equal_global_costs_choose_identity_deterministically():
    candidates, references, scorer, _ = fixture([[0.5, 0.5], [0.5, 0.5]])
    assert EVALUATE.select_pairs(candidates, references, scorer)[:2] == (0, (0, 1))


def test_nonfinite_costs_are_rejected_even_in_the_other_assignment():
    for invalid in (float("nan"), float("inf"), -float("inf")):
        candidates, references, scorer, _ = fixture([[0.1, invalid], [3.0, 0.2]])
        with unittest.TestCase().assertRaisesRegex(ValueError, "Nonfinite PIT costs"):
            EVALUATE.select_pairs(candidates, references, scorer)


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items())
        if name.startswith("test_")
    )
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
