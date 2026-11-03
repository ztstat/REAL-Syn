"""Run equal-budget weighting and resampling comparisons."""
from common.io import write_json, write_csv
from . import core, reporting


def run(config, source, output, repetitions, quick):
    config = dict(config, run_label="quick" if quick else "full")
    if tuple(config["methods"]) != core.METHODS:
        raise ValueError("Method configuration does not match the experiment.")
    test_x, test_y = core.fixed_test_data(config, source)
    metrics, parameters, ratios, checks = [], [], [], []
    for rep in range(repetitions):
        result = core.run_repetition(config, source, rep, test_x, test_y)
        metrics.extend(result["per_rep_rows"])
        parameters.extend(result["parameter_rows"])
        ratios.append(result["ratio_row"])
        checks.append(result["sanity_row"])
        print(f"Completed repetition {rep + 1}/{repetitions}", flush=True)
    summary, covariance, contrasts = reporting.aggregate(
        metrics, source, config["run_label"]
    )
    for filename, rows in {
        "per_rep_metrics.csv": metrics, "parameter_vectors.csv": parameters,
        "ratio_diagnostics.csv": ratios, "sanity_checks.csv": checks,
        "method_summary.csv": summary,
        "coefficient_covariance_matrices.csv": covariance,
        "paired_contrasts.csv": contrasts,
    }.items():
        write_csv(output / filename, rows)
    write_json(output / "seeds.json",
               core.seed_manifest(config, source, repetitions))
