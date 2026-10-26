"""Numerical checks for finite-length CTF operators and unrolled CG."""

import unittest

import torch

from espnet2.enh.layers.ctf_deconvolution import (
    ctf_adjoint,
    ctf_forward,
    regularized_deconvolution,
)


def random_complex(shape, dtype=torch.complex128):
    return torch.randn(shape, dtype=dtype)


def causal_matrix(filter_, frames):
    zero = filter_.new_zeros(())
    return torch.stack(
        [
            torch.stack(
                [
                    filter_[row - col] if 0 <= row - col < len(filter_) else zero
                    for col in range(frames)
                ]
            )
            for row in range(frames)
        ]
    )


class TestCTFDeconvolution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(732)

    def test_identity_and_analytic_regularized_solution(self):
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                observation = random_complex((2, 2, 3, 9), dtype)
                prior = random_complex(observation.shape, dtype)
                ctf = torch.zeros((2, 2, 3, 5), dtype=dtype)
                ctf[..., 0] = 1
                lengths = torch.tensor([9, 5])
                valid = (
                    torch.arange(9)[None, None, None, :]
                    < lengths[:, None, None, None]
                )
                tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
                torch.testing.assert_close(
                    ctf_forward(observation, ctf, lengths),
                    observation.masked_fill(~valid, 0),
                    atol=tolerance,
                    rtol=tolerance,
                )
                # Identity is not generally y when lambda > 0 and s0 != y.
                gamma, eps = 0.37, 0.013
                expected = (
                    observation + (gamma + eps) * prior
                ) / (1 + gamma + eps)
                expected = expected.masked_fill(~valid, 0)
                result = regularized_deconvolution(
                    observation, ctf, prior, lengths, gamma=gamma, eps=eps
                )
                torch.testing.assert_close(
                    result, expected, atol=tolerance, rtol=tolerance
                )

    def test_forward_matches_explicit_causal_convolution(self):
        signal = random_complex((2, 2, 2, 5))
        ctf = random_complex((2, 2, 2, 11))
        result = ctf_forward(signal, ctf)
        expected = torch.empty_like(result)
        for batch in range(2):
            for speaker in range(2):
                for frequency in range(2):
                    index = (batch, speaker, frequency)
                    expected[index] = causal_matrix(ctf[index], 5) @ signal[index]
        torch.testing.assert_close(result, expected, atol=2e-14, rtol=2e-14)

    def test_complex_adjoint_with_lengths_and_weights(self):
        signal = random_complex((2, 2, 3, 7))
        observation = random_complex(signal.shape)
        ctf = random_complex((2, 2, 3, 12))
        lengths = torch.tensor([[7, 4], [0, 6]])
        weights = torch.rand((2, 2, 1, 7), dtype=torch.float64)
        weights[..., :2] = 0
        left = (
            ctf_forward(signal, ctf, lengths, weights).conj() * observation
        ).sum()
        right = (
            signal.conj() * ctf_adjoint(observation, ctf, lengths, weights)
        ).sum()
        torch.testing.assert_close(left, right, atol=3e-13, rtol=3e-13)

    def test_cg_matches_explicit_weighted_hermitian_solve(self):
        frames = 6
        observation = random_complex((2, 2, 2, frames))
        prior = random_complex(observation.shape)
        ctf = random_complex((2, 2, 2, 9))
        lengths = torch.tensor([[6, 4], [1, 0]])
        weights = torch.rand((2, 2, 2, frames), dtype=torch.float64)
        weights[..., 0] = 0
        gamma, eps = 0.23, 0.002
        result = regularized_deconvolution(
            observation, ctf, prior, lengths, weights,
            gamma=gamma, eps=eps, iterations=20,
        )
        expected = torch.zeros_like(result)
        for batch in range(2):
            for speaker in range(2):
                length = int(lengths[batch, speaker])
                if length == 0:
                    continue
                for frequency in range(2):
                    index = (batch, speaker, frequency)
                    a = causal_matrix(ctf[index], length)
                    w = torch.diag(weights[index][:length]).to(a.dtype)
                    lam = gamma * ctf[index].abs().square().sum() + eps
                    matrix = a.mH @ w @ a + lam * torch.eye(length, dtype=a.dtype)
                    torch.testing.assert_close(
                        matrix, matrix.mH, atol=2e-14, rtol=2e-14
                    )
                    rhs = (
                        a.mH @ w @ observation[index][:length]
                        + lam * prior[index][:length]
                    )
                    expected[index][:length] = torch.linalg.solve(matrix, rhs)
        torch.testing.assert_close(result, expected, atol=2e-11, rtol=2e-11)

    def test_observation_mask_keeps_input_context(self):
        signal = torch.zeros((1, 1, 1, 5), dtype=torch.complex128)
        signal[..., 0] = 2 + 3j
        ctf = torch.ones((1, 1, 1, 5), dtype=torch.complex128)
        mask = torch.tensor([[False, False, True, True, True]])
        expected = signal.new_tensor(
            [0, 0, 2 + 3j, 2 + 3j, 2 + 3j]
        ).reshape_as(signal)
        torch.testing.assert_close(
            ctf_forward(signal, ctf, observation_mask=mask), expected
        )

    def test_padding_cannot_contaminate_valid_frames_or_gradients(self):
        lengths = torch.tensor([5, 2])
        shape = (2, 2, 3, 5)
        valid = torch.arange(5)[None, None, None, :] < lengths[:, None, None, None]
        signal = random_complex(shape).requires_grad_()
        ctf = random_complex((2, 2, 3, 8)).requires_grad_()
        contaminated = torch.where(valid, signal, 1000 + 2000j)
        result = ctf_forward(contaminated, ctf, lengths)
        expected = ctf_forward(signal.masked_fill(~valid, 0), ctf, lengths)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        self.assertTrue((result.masked_select(~valid.expand(shape)) == 0).all())
        result.real.square().sum().backward()
        self.assertTrue((signal.grad.masked_select(~valid.expand(shape)) == 0).all())
        self.assertTrue(torch.isfinite(ctf.grad).all())
        observation = random_complex(shape).requires_grad_()
        prior = random_complex(shape).requires_grad_()
        solution = regularized_deconvolution(observation, ctf, prior, lengths)
        self.assertTrue((solution.masked_select(~valid.expand(shape)) == 0).all())
        solution.abs().square().sum().backward()
        self.assertTrue(
            (observation.grad.masked_select(~valid.expand(shape)) == 0).all()
        )
        self.assertTrue((prior.grad.masked_select(~valid.expand(shape)) == 0).all())

    def test_complex_gradcheck_for_y_h_and_prior(self):
        observation = random_complex((1, 1, 1, 3)).requires_grad_()
        ctf = random_complex((1, 1, 1, 2)).requires_grad_()
        prior = random_complex(observation.shape).requires_grad_()
        weights = torch.tensor([0.5, 1, 0.8], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda y, h, s0: regularized_deconvolution(
                y, h, s0, observation_mask=weights,
                gamma=0.2, eps=1e-3, iterations=6,
            ),
            (observation, ctf, prior), eps=1e-6, atol=2e-5, rtol=2e-4,
        ))

    def test_silence_and_zero_residual_have_finite_gradients(self):
        observation = torch.zeros(
            (2, 2, 2, 7), dtype=torch.complex64, requires_grad=True
        )
        ctf = random_complex((2, 2, 2, 11), torch.complex64).requires_grad_()
        prior = torch.zeros_like(observation, requires_grad=True)
        result = regularized_deconvolution(observation, ctf, prior, iterations=16)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
        result.real.sum().backward()
        for value in (observation, ctf, prior):
            self.assertTrue(torch.isfinite(value.grad).all())
        prior = random_complex((1, 1, 1, 5), torch.complex64)
        identity = prior.new_ones((1, 1, 1, 1))
        result = regularized_deconvolution(prior, identity, prior)
        torch.testing.assert_close(result, prior, atol=1e-6, rtol=1e-6)

    def test_zero_filter_tiny_regularization_and_no_observations(self):
        observation = random_complex((1, 2, 2, 6), torch.complex64)
        prior = random_complex(observation.shape, torch.complex64)
        ctf = torch.zeros((1, 2, 2, 10), dtype=torch.complex64)
        result = regularized_deconvolution(
            observation, ctf, prior, gamma=0, eps=1e-20,
        )
        torch.testing.assert_close(result, prior, atol=0, rtol=0)
        ctf = random_complex(ctf.shape, torch.complex64)
        result = regularized_deconvolution(
            observation, ctf, prior, observation_mask=torch.zeros((1, 6)),
            gamma=0, eps=1e-12,
        )
        torch.testing.assert_close(result, prior, atol=0, rtol=0)

    def test_small_filter_and_regularizer_analytic_solution(self):
        observation = random_complex((1, 1, 2, 5), torch.complex64) * 1e-6
        prior = random_complex(observation.shape, torch.complex64)
        ctf = torch.full((1, 1, 2, 1), 1e-6 + 2e-6j, dtype=torch.complex64)
        gamma, eps = 0.1, 1e-20
        lam = gamma * ctf.abs().square() + eps
        expected = (ctf.conj() * observation + lam * prior) / (
            ctf.abs().square() + lam
        )
        result = regularized_deconvolution(
            observation, ctf, prior, gamma=gamma, eps=eps
        )
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)

    def test_nearly_silent_residual_does_not_underflow(self):
        observation = random_complex((1, 1, 2, 5), torch.complex64) * 1e-30
        prior = torch.zeros_like(observation)
        ctf = torch.ones((1, 1, 2, 1), dtype=torch.complex64)
        result = regularized_deconvolution(observation, ctf, prior)
        expected = observation / (1 + 0.1 + 1e-6)
        # Compare in units of the tiny signal; ordinary absolute tolerances
        # would incorrectly let a completely zero reconstruction pass.
        torch.testing.assert_close(
            result / 1e-30, expected / 1e-30, atol=2e-6, rtol=2e-6
        )

    def test_single_frame_and_per_speaker_lengths(self):
        observation = random_complex((2, 2, 2, 1))
        prior = random_complex(observation.shape)
        ctf = random_complex((2, 2, 2, 1))
        lengths = torch.tensor([[1, 0], [0, 1]])
        result = regularized_deconvolution(observation, ctf, prior, lengths)
        lam = 0.1 * ctf.abs().square() + 1e-6
        expected = (ctf.conj() * observation + lam * prior) / (
            ctf.abs().square() + lam
        )
        expected = expected * lengths[:, :, None, None]
        torch.testing.assert_close(result, expected, atol=1e-13, rtol=1e-13)

    def test_invalid_inputs_fail_explicitly(self):
        signal = random_complex((1, 1, 1, 4))
        ctf = random_complex((1, 1, 1, 2))
        for bad in (signal.real, signal[..., :0], signal.squeeze(0)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    ctf_forward(bad, ctf)
        for value in (float('nan'), float('inf')):
            bad_signal = signal.clone()
            bad_signal[..., -1] = value
            with self.assertRaises(ValueError):
                ctf_forward(bad_signal, ctf, torch.tensor([2]))
            bad_ctf = ctf.clone()
            bad_ctf[..., 0] = value
            with self.assertRaises(ValueError):
                ctf_adjoint(signal, bad_ctf)
            with self.assertRaises(ValueError):
                regularized_deconvolution(signal, ctf, bad_signal)
        for lengths in (torch.tensor([5]), torch.tensor([-1]), torch.tensor([1.5])):
            with self.assertRaises((TypeError, ValueError)):
                ctf_forward(signal, ctf, lengths)
        for mask in (
            torch.tensor([-1., 0., 1., 1.]),
            torch.tensor([float('nan'), 0., 1., 1.]),
            torch.ones(3, 4),
            torch.ones(4, dtype=torch.int64),
        ):
            with self.assertRaises((TypeError, ValueError)):
                ctf_forward(signal, ctf, observation_mask=mask)
        for kwargs in (
            {'gamma': -1}, {'gamma': float('inf')}, {'eps': 0},
            {'eps': float('nan')}, {'iterations': 0}, {'iterations': 1.5},
        ):
            with self.assertRaises((TypeError, ValueError)):
                regularized_deconvolution(signal, ctf, signal, **kwargs)


if __name__ == '__main__':
    unittest.main()
