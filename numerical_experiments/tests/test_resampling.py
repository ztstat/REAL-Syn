from __future__ import annotations
import unittest
import numpy as np
from experiments.weighting_resampling import methods
from experiments.weighting_resampling import core


def weighted_ols(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(x.shape[0]), x])
    root = np.sqrt(weights)[:, None]
    return np.linalg.lstsq(root * design, root[:, 0] * y, rcond=None)[0]


class ResidualResamplingTests(unittest.TestCase):
    def test_integer_nonnegative_exact_budget_and_reproducible(self) -> None:
        probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
        first = methods.residual_resample(
            probabilities, 17, np.random.RandomState(123)
        )
        second = methods.residual_resample(
            probabilities, 17, np.random.RandomState(123)
        )
        self.assertTrue(
            np.issubdtype(first["multiplicities"].dtype, np.integer)
        )
        self.assertTrue(np.all(first["multiplicities"] >= 0))
        self.assertEqual(int(np.sum(first["multiplicities"])), 17)
        np.testing.assert_array_equal(
            first["multiplicities"], second["multiplicities"]
        )

    def test_r_zero_uniform_and_zero_residual_masses(self) -> None:
        result = methods.residual_resample(
            np.ones(5), 20, np.random.RandomState(2)
        )
        np.testing.assert_array_equal(
            result["multiplicities"], np.full(5, 4)
        )
        self.assertEqual(result["residual_draw_count"], 0)

    def test_one_dominant_weight(self) -> None:
        result = methods.residual_resample(
            np.asarray([0.999, 0.0005, 0.0005]),
            100,
            np.random.RandomState(4),
        )
        self.assertGreaterEqual(result["multiplicities"][0], 99)
        self.assertEqual(int(np.sum(result["multiplicities"])), 100)

    def test_empirical_expected_counts(self) -> None:
        probabilities = np.asarray([0.05, 0.15, 0.3, 0.5])
        total = np.zeros(4)
        trials = 3000
        for seed in range(trials):
            total += methods.residual_resample(
                probabilities, 11, np.random.RandomState(seed)
            )["multiplicities"]
        np.testing.assert_allclose(
            total / trials, 11 * probabilities, atol=0.06, rtol=0.0
        )


class StratifiedResamplingTests(unittest.TestCase):
    def test_exact_draws_valid_indices_and_reproducibility(self) -> None:
        probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
        first = methods.stratified_resample(
            probabilities, 23, np.random.RandomState(19)
        )
        second = methods.stratified_resample(
            probabilities, 23, np.random.RandomState(19)
        )
        self.assertEqual(first["draws"].size, 23)
        self.assertTrue(np.all(first["draws"] >= 0))
        self.assertTrue(np.all(first["draws"] < probabilities.size))
        np.testing.assert_array_equal(first["draws"], second["draws"])

    def test_uniform_and_one_dominant_weight(self) -> None:
        uniform = methods.stratified_resample(
            np.ones(5), 20, np.random.RandomState(2)
        )
        np.testing.assert_array_equal(
            uniform["multiplicities"], np.full(5, 4)
        )
        dominant = methods.stratified_resample(
            np.asarray([1e-12, 1e-12, 1.0]),
            30,
            np.random.RandomState(3),
        )
        self.assertEqual(dominant["multiplicities"][2], 30)

    def test_final_cdf_and_empirical_expected_counts(self) -> None:
        probabilities = np.asarray([0.05, 0.15, 0.3, 0.5])
        total = np.zeros(4)
        trials = 3000
        for seed in range(trials):
            result = methods.stratified_resample(
                probabilities, 11, np.random.RandomState(seed)
            )
            total += result["multiplicities"]
            self.assertLess(int(np.max(result["draws"])), 4)
        np.testing.assert_allclose(
            total / trials, 11 * probabilities, atol=0.04, rtol=0.0
        )


class MultiplicityFittingTests(unittest.TestCase):
    def test_unique_weighted_fit_matches_duplicated_fit(self) -> None:
        x = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
        y = np.asarray([-1.2, 0.1, 1.1, 2.4])
        counts = np.asarray([1, 3, 2, 4])
        indices = np.flatnonzero(counts)
        weighted = weighted_ols(
            x[indices], y[indices], counts[indices].astype(float)
        )
        draws = np.repeat(indices, counts[indices])
        duplicated = weighted_ols(
            x[draws], y[draws], np.ones(draws.size)
        )
        np.testing.assert_allclose(weighted, duplicated, atol=1e-12, rtol=0.0)

    def test_sir_covariance_compression_equals_explicit_duplication(self) -> None:
        psi = np.asarray(
            [
                [1.0, -2.0, 0.5],
                [-0.25, 3.0, 1.5],
                [2.0, 0.75, -1.0],
            ]
        )
        multiplicities = np.asarray([1, 3, 2], dtype=np.int64)
        compressed = core.sir_draw_level_vw_from_psi(
            psi, multiplicities, budget=6
        )
        explicit = np.repeat(psi, multiplicities, axis=0)
        explicit_vw = explicit.T @ explicit / 6.0
        np.testing.assert_allclose(
            compressed, explicit_vw, atol=1e-12, rtol=0.0
        )

        # Squaring multiplicities is not the draw-level second moment.
        squared_weighted = multiplicities[:, None] * psi
        squared_vw = squared_weighted.T @ squared_weighted / float(psi.shape[0])
        self.assertGreater(float(np.max(np.abs(squared_vw - explicit_vw))), 1e-3)


if __name__ == "__main__":
    unittest.main()
