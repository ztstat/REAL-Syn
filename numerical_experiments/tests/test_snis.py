from __future__ import annotations
import unittest
import numpy as np
from experiments.weighting_resampling import methods


class SubsampledSNISTests(unittest.TestCase):
    def test_uniform_subset_exact_distinct_and_reproducible(self) -> None:
        first = methods.uniform_subset(100, 20, np.random.RandomState(5))
        second = methods.uniform_subset(100, 20, np.random.RandomState(5))
        self.assertEqual(first.size, 20)
        self.assertEqual(np.unique(first).size, 20)
        np.testing.assert_array_equal(first, second)

    def test_uniform_without_replacement_inclusion_frequency(self) -> None:
        counts = np.zeros(10)
        trials = 3000
        for seed in range(trials):
            counts[
                methods.uniform_subset(10, 3, np.random.RandomState(seed))
            ] += 1
        np.testing.assert_allclose(
            counts / trials, np.full(10, 0.3), atol=0.025, rtol=0.0
        )

    def test_weights_sum_ess_uniform_and_unequal(self) -> None:
        ratios = np.geomspace(1e-5, 1e5, 100)
        subset = np.arange(20)
        result = methods.subsampled_snis(ratios, subset, 20)
        self.assertAlmostEqual(float(np.sum(result["weights"])), 20.0, 12)
        self.assertTrue(np.all(np.isfinite(result["weights"])))
        self.assertTrue(np.all(result["weights"] > 0.0))
        self.assertAlmostEqual(
            result["final_weight_ess"], result["raw_ess"], 11
        )
        uniform = methods.subsampled_snis(
            np.ones(100), subset, 20
        )
        np.testing.assert_allclose(uniform["weights"], np.ones(20))
        self.assertAlmostEqual(uniform["raw_ess"], 20.0, 12)


class ClippedSNISTests(unittest.TestCase):
    def test_shared_subset_threshold_bounds_and_activation(self) -> None:
        ratios = np.ones(100)
        ratios[19] = 1e6
        subset = np.arange(20)
        raw = methods.subsampled_snis(ratios, subset, 20)
        clipped = methods.clipped_snis(ratios, subset, 20)
        np.testing.assert_array_equal(raw["indices"], clipped["indices"])
        expected_threshold = float(
            np.mean(ratios[subset]) * np.sqrt(20)
        )
        self.assertAlmostEqual(
            clipped["clipping_threshold"], expected_threshold, 12
        )
        self.assertTrue(
            np.all(clipped["clipped_ratios"] <= expected_threshold)
        )
        self.assertEqual(
            clipped["clipped_count"],
            int(np.count_nonzero(ratios[subset] > expected_threshold)),
        )
        self.assertGreaterEqual(
            clipped["clipped_ess"] + 1e-12, clipped["raw_ess"]
        )
        self.assertAlmostEqual(float(np.sum(clipped["weights"])), 20.0, 12)

    def test_no_clipping_matches_subsampled_and_reproducible(self) -> None:
        ratios = np.ones(100)
        subset = methods.uniform_subset(100, 20, np.random.RandomState(8))
        raw = methods.subsampled_snis(ratios, subset, 20)
        first = methods.clipped_snis(ratios, subset, 20)
        second = methods.clipped_snis(ratios, subset, 20)
        self.assertEqual(first["clipped_count"], 0)
        np.testing.assert_array_equal(first["weights"], raw["weights"])
        np.testing.assert_array_equal(first["weights"], second["weights"])


if __name__ == "__main__":
    unittest.main()
