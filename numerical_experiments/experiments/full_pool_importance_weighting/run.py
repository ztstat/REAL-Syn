"""Run the fixed-candidate-pool alpha grid."""
from common.io import write_json
from . import core, reporting


def run(config, source, output, repetitions):
    if tuple(config["alpha_grid"]) != core.ALPHAS:
        raise ValueError("Alpha grid does not match the experiment.")
    if tuple(config["methods"]) != core.METHODS:
        raise ValueError("Method configuration does not match the experiment.")
    test_x, test_y = core.fixed_test_data(source)
    metrics, parameters, diagnostics, ratios, checks = [], [], [], [], []
    for rep in range(repetitions):
        result = core.run_repetition(source, rep, test_x, test_y)
        metrics.extend(result["metric_rows"])
        parameters.extend(result["parameter_rows"])
        diagnostics.extend(result["diagnostic_rows"])
        ratios.append(result["ratio_row"])
        checks.append(result["sanity"])
        print(f"Completed repetition {rep + 1}/{repetitions}", flush=True)
    summary, covariance, matrices = reporting.build_summaries(
        metrics, parameters, diagnostics, source
    )
    for filename, rows in {
        "per_rep_metrics.csv": metrics, "parameter_vectors.csv": parameters,
        "per_rep_method_diagnostics.csv": diagnostics,
        "ratio_diagnostics.csv": ratios, "sanity_checks.csv": checks,
        "method_summary.csv": summary, "covariance_summary.csv": covariance,
        "coefficient_covariance_matrices.csv": matrices,
        "paired_contrasts.csv": reporting.build_paired_contrasts(metrics),
    }.items():
        reporting.write_csv(output / filename, rows)
    reporting.generate_figures(summary, output)
    write_json(output / "seeds.json", core.seed_manifest(source, repetitions))
