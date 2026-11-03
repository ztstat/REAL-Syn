from __future__ import annotations
import json
import unittest
from pathlib import Path
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "experiments" / "logistic_regression"

from experiments.logistic_regression import metrics as logistic_metrics
from experiments.logistic_regression import parameters as logistic_parameters
from experiments.logistic_regression.model import (
    add_intercept,
    compute_score_psi,
    fit_weighted_logistic,
    logistic_objective_gradient_hessian,
)
from experiments.logistic_regression.pipeline import compute_calibrated_pi, sample_with_pi


def load_config(name: str) -> dict:
    with (PROJECT / "configs" / name).open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


class DistributionTests(unittest.TestCase):
    def test_target_posterior_matches_logistic_formula(self) -> None:
        rng = np.random.RandomState(42)
        x = rng.normal(size=(500, 5))
        for pi_p in [0.2, 0.5]:
            factorized = logistic_parameters.target_posterior_from_factorization(
                x, pi_p
            )
            logistic = logistic_parameters.target_posterior_from_logistic(x, pi_p)
            np.testing.assert_allclose(factorized, logistic, rtol=1e-13, atol=1e-13)

    def test_density_ratio_identity_on_proposal_samples(self) -> None:
        config = load_config("imbalanced.json")
        rng = np.random.RandomState(123)
        x, y = logistic_parameters.sample_joint(
            300000, config["pi_q"], config["proposal"], rng
        )
        clipped, raw = logistic_parameters.oracle_log_density_ratio(x, y, config)
        ratio = np.exp(clipped)
        self.assertLess(abs(float(np.mean(ratio)) - 1.0), 0.02)
        self.assertTrue(np.allclose(clipped, np.clip(raw, -20.0, 20.0)))


class LogisticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("balanced.json")

    def test_weighted_gradient_matches_finite_difference(self) -> None:
        rng = np.random.RandomState(7)
        x = rng.normal(size=(17, 5))
        x_bar = add_intercept(x)
        y = rng.binomial(1, 0.4, size=17)
        weights = 0.2 + rng.random_sample(17)
        theta = rng.normal(scale=0.2, size=6)
        objective, gradient, _ = logistic_objective_gradient_hessian(
            theta, x_bar, y, weights, 1e-6
        )
        self.assertTrue(np.isfinite(objective))
        epsilon = 1e-6
        numerical = np.empty_like(theta)
        for index in range(theta.size):
            displacement = np.zeros_like(theta)
            displacement[index] = epsilon
            plus = logistic_objective_gradient_hessian(
                theta + displacement, x_bar, y, weights, 1e-6
            )[0]
            minus = logistic_objective_gradient_hessian(
                theta - displacement, x_bar, y, weights, 1e-6
            )[0]
            numerical[index] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)

    def test_solver_reports_clear_status_for_separable_and_nonseparable(self) -> None:
        separable_x = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        separable_y = np.asarray([0, 0, 0, 1, 1, 1])
        nonseparable_x = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
        nonseparable_y = np.asarray([0, 1, 0, 1, 0, 1])
        for x, y in [
            (separable_x, separable_y),
            (nonseparable_x, nonseparable_y),
        ]:
            result = fit_weighted_logistic(x, y, None, self.config["solver"])
            self.assertIn(
                result["status"],
                {
                    "converged",
                    "maximum_iterations_reached",
                    "line_search_failed",
                    "hessian_solve_failed",
                    "nonfinite_objective_or_gradient",
                },
            )
            self.assertIsInstance(result["converged"], bool)
            self.assertTrue(np.all(np.isfinite(result["theta"])))
            self.assertTrue(np.isfinite(result["gradient_norm"]))

    def test_score_dimension(self) -> None:
        rng = np.random.RandomState(12)
        x = rng.normal(size=(30, 5))
        y = rng.binomial(1, 0.5, size=30)
        model = fit_weighted_logistic(x, y, None, self.config["solver"])
        psi = compute_score_psi(x, y, model)
        self.assertEqual(psi.shape, (30, 6))


class AcceptanceAndMetricTests(unittest.TestCase):
    def test_pi_calibration_weight_identity_and_ess(self) -> None:
        rng = np.random.RandomState(99)
        score = np.exp(rng.normal(size=5000))
        pi, _ = compute_calibrated_pi(score, 0.1)
        self.assertTrue(np.all((0.0 <= pi) & (pi <= 1.0)))
        self.assertAlmostEqual(float(np.mean(pi)), 0.1, places=12)
        x = rng.normal(size=(5000, 5))
        y = rng.binomial(1, 0.5, size=5000)
        ratio = np.exp(rng.normal(scale=0.2, size=5000))
        sample = sample_with_pi(x, y, pi, ratio, 0.1, rng)
        np.testing.assert_allclose(
            sample["weight"],
            0.1 * sample["accepted_ratio"] / sample["accepted_pi"],
            rtol=0.0,
            atol=0.0,
        )
        ess, relative = logistic_metrics.effective_sample_size(sample["weight"])
        self.assertGreaterEqual(ess, 1.0 - 1e-12)
        self.assertLessEqual(ess, sample["weight"].size + 1e-10)
        self.assertGreater(relative, 0.0)
        self.assertLessEqual(relative, 1.0 + 1e-12)

    def test_auroc_and_auprc_manual_example(self) -> None:
        y = np.asarray([1, 0, 1, 0])
        probability = np.asarray([0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(logistic_metrics.binary_auroc(y, probability), 0.75)
        self.assertAlmostEqual(
            logistic_metrics.binary_auprc(y, probability),
            5.0 / 6.0,
        )


if __name__ == "__main__":
    unittest.main()
