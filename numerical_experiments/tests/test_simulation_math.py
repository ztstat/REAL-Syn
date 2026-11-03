"""Configuration contracts and mathematical identities for the simulations."""
import json
from pathlib import Path
import unittest
import numpy as np
from common import evaluation, ols_model, ols_pipeline, pipeline_math, importance
from experiments.acceptance_budget.sweep import get_child_n_candidates
from experiments.highdim_ols.parameters import make_dense_beta, make_sparse_beta
from experiments.full_pool_importance_weighting import core as full_pool
from experiments.density_ratio_sensitivity import core as ratio_core
from experiments.proposal_mismatch import grid_core, stress_core

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    registry = json.loads((ROOT / "experiments.json").read_text())
    return json.loads((ROOT / registry[name]["config"]).read_text())


class ConfigurationTests(unittest.TestCase):
    def test_low_dimensions_and_seeds(self):
        for d in (1, 2, 5):
            c = config(f"ols-d{d}")
            self.assertEqual(c["data"]["x_dim"], d)
            self.assertEqual(c["data"]["n_train"], 40)
            self.assertEqual(c["data"]["n_test"], 8000)
            self.assertEqual(c["pipeline"]["n_candidates"], 6000)
            self.assertEqual(c["pipeline"]["reps"], 100)
            self.assertEqual(c["pipeline"]["alpha"], 0.1)
            self.assertEqual(c["seed"], 0)
            self.assertEqual(c["pipeline"]["rep_seed_stride"], 1337)
            self.assertEqual(c["ratio"]["clip_log_ratio"], 20)
            self.assertEqual(c["model"]["ridge"], 1e-10)

    def test_high_dimensions_and_coefficients(self):
        for arm in ("sparse", "dense"):
            for d in (10, 20):
                c = config(f"highdim-{arm}-d{d}")
                beta = make_sparse_beta(d) if arm == "sparse" else make_dense_beta(d)
                np.testing.assert_array_equal(c["data"]["beta"], beta)
                np.testing.assert_allclose(c["synthetic_candidate"]["beta"], 0.8 * beta, atol=1e-15)
                self.assertEqual(c["data"]["n_train"], 200)
                self.assertEqual(c["pipeline"]["n_candidates"], 30000)
                self.assertEqual(c["pipeline"]["reps"], 100)
                self.assertAlmostEqual(float(beta @ beta), 2.2)

    def test_fixed_pool_is_not_fixed_expected_count(self):
        self.assertEqual(config("full-pool-alpha")["alpha_grid"], list(full_pool.ALPHAS))
        self.assertEqual(config("full-pool-alpha")["n_candidates_fixed"], 6000)
        for name in ("figure4", "figure4-low-alpha"):
            c = config(name)
            self.assertEqual(c["pipeline"]["reps"], 300)
            self.assertEqual(c["alpha_sweep"]["target_expected_accept_count"], 500)
            for alpha in c["alpha_sweep"]["alphas"]:
                self.assertEqual(get_child_n_candidates(c, alpha), round(500 / alpha))
                self.assertLessEqual(abs(alpha * get_child_n_candidates(c, alpha) - 500), alpha / 2 + 1e-12)

    def test_visualization_configuration(self):
        c = config("figure1")
        self.assertEqual(c["pipeline"]["reps"], 50)
        self.assertEqual(c["pipeline"]["n_candidates"], 30000)
        self.assertEqual(c["pipeline"]["alpha"], 0.3)
        self.assertEqual(c["data"]["n_train"], 200)

    def test_grids_and_lambda(self):
        for name, core in (("proposal-grid", grid_core), ("proposal-stress", stress_core)):
            c = config(name)
            self.assertEqual(tuple((x["grid_cell"], x["mu_x"], x["sigma_x"]) for x in c["grid"]), core.GRID)
            self.assertEqual(c["repetitions"], 100)
        self.assertEqual(tuple(x["lambda"] for x in config("ratio-misspecification")["ratio_regimes"]),
                         tuple(v for _, v in ratio_core.RATIO_REGIMES))

    def test_logistic_seeds_and_repetitions(self):
        for name, pi_p, base, test in (
            ("logistic-imbalanced", 0.2, 2734101, 8734101),
            ("logistic-balanced", 0.5, 2734102, 8734102),
        ):
            c = config(name)
            self.assertEqual((c["pi_p"], c["pi_q"]), (pi_p, 0.5))
            self.assertEqual(c["repetitions"], 100)
            self.assertEqual(c["seeds"]["base_seed"], base)
            self.assertEqual(c["seeds"]["fixed_test_seed"], test)
            self.assertFalse(c["solver"]["penalize_intercept"])


class OLSIdentityTests(unittest.TestCase):
    def test_calibration_and_correction(self):
        rng = np.random.RandomState(25)
        score = np.exp(rng.normal(size=2000))
        ratio = np.exp(rng.normal(size=2000))
        x, y = rng.normal(size=(2000, 2)), rng.normal(size=2000)
        pi, _ = ols_pipeline.compute_calibrated_pi(score, 0.1)
        self.assertAlmostEqual(float(np.mean(pi)), 0.1, places=12)
        sample = pipeline_math.sample_with_pi(x, y, pi, ratio, 0.1, rng)
        mask = sample["accepted_mask"]
        np.testing.assert_array_equal(sample["weight"], 0.1 * ratio[mask] / np.clip(pi[mask], 1e-12, 1))
        endpoint, _ = ols_pipeline.compute_calibrated_pi(score, 1.0)
        np.testing.assert_array_equal(endpoint, np.ones(2000))

    def test_joint_ratio_matches_joint_gaussian_density(self):
        from scipy.stats import multivariate_normal
        c = config("ols-d5")
        rng = np.random.RandomState(9)
        x, y = rng.normal(size=(20, 5)), rng.normal(size=20)
        p_mean, p_cov = grid_core.linear_gaussian_joint_parameters(c["data"])
        q_mean, q_cov = grid_core.linear_gaussian_joint_parameters(c["synthetic_candidate"])
        expected = multivariate_normal.logpdf(np.column_stack((x, y)), p_mean, p_cov) - multivariate_normal.logpdf(np.column_stack((x, y)), q_mean, q_cov)
        observed = ols_pipeline.linear_gaussian_joint_logpdf(x, y, c["data"], 5) - ols_pipeline.linear_gaussian_joint_logpdf(x, y, c["synthetic_candidate"], 5)
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_ridge_includes_intercept_in_ols(self):
        x = np.arange(8.0)[:, None]
        y = 2.0 + 0.8 * x[:, 0]
        w = np.arange(1.0, 9.0)
        design = np.column_stack((np.ones(8), x))
        expected = np.linalg.solve(design.T @ (w[:, None] * design) + 1e-10 * np.eye(2), design.T @ (w * y))
        observed = ols_model._weighted_linear_regression_fit(x, y, w, 1e-10)
        np.testing.assert_array_equal(observed["beta_full"], expected)

    def test_score_and_weighted_second_moment(self):
        x = np.array([[1., 2.], [-1., 3.], [2., -2.]])
        y = np.array([1., 2., -1.])
        psi = np.column_stack((y, x))
        weights = np.array([0.5, 2., 3.])
        inv = np.diag([1., 2., 0.5])
        psi_fn = lambda *args: psi
        stats = pipeline_math.compute_theory_statistics(
            x, y, None, None, inv, {},
            lambda *args: np.log(np.array([0.5, 2., 1.])), psi_fn)
        np.testing.assert_allclose(stats["score"],
            np.array([0.5, 2., 1.]) * np.sqrt(np.einsum("ij,jk,ik->i", psi, inv, psi)))
        vw = pipeline_math.estimate_vw_from_accepted(x, y, weights, None, {}, 3, psi_fn)
        expected = (weights[:, None] * psi).T @ (weights[:, None] * psi) / 3
        np.testing.assert_array_equal(vw, expected)

    def test_mean_se_and_complete_covariance(self):
        x = np.array([1., 2., 4., 8.])
        mean, se = evaluation.mean_se(x)
        self.assertEqual(mean, np.mean(x))
        self.assertEqual(se, np.std(x, ddof=1) / np.sqrt(4))
        theta = np.column_stack((x, -x, x / 2))
        self.assertEqual(evaluation.trace_covariance_of_vectors(theta),
                         np.trace(np.cov(theta, rowvar=False, ddof=1)))

    def test_all_iw_weights_and_ess_scale_invariance(self):
        source = config("ols-d5")
        setting = {"distribution": "gaussian_linear", "alpha": 0.1,
                   "synthetic_budget": 600, "seeds": {"sir_resampling_offset": 8000051}}
        test_x, test_y = importance.fixed_test_data(setting, source)
        prepared = importance.prepare_repetition(setting, source, 0, test_x, test_y)
        matched = importance.build_method_sample("All-IW-matched", prepared, setting)
        full = importance.build_method_sample("All-IW-full", prepared, setting)
        np.testing.assert_array_equal(matched["weights"], 0.1 * prepared["ratio"])
        np.testing.assert_array_equal(full["weights"], prepared["ratio"])
        self.assertEqual(len(full["weights"]), 6000)
        self.assertAlmostEqual(importance.effective_sample_size(full["weights"]),
                               importance.effective_sample_size(matched["weights"]), places=10)


if __name__ == "__main__":
    unittest.main()
