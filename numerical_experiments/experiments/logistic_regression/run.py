"""Run either class-prior configuration."""
import numpy as np
from common.io import write_json
from . import parameters, pipeline, reporting


def run(config, output, repetitions):
    parameters.validate_config(config)
    test_x, test_y = parameters.sample_joint(
        int(config["n_test"]), float(config["pi_p"]), config["target"],
        np.random.RandomState(int(config["seeds"]["fixed_test_seed"])),
    )
    probability = parameters.target_posterior_from_logistic(
        test_x, float(config["pi_p"]), float(config["a"])
    )
    metrics, coefficients, diagnostics = [], [], []
    for rep in range(repetitions):
        result = pipeline.run_single_repetition(
            config, rep, test_x, test_y, probability
        )
        metrics.extend(result["metric_rows"])
        coefficients.extend(result["coefficient_rows"])
        diagnostics.append(result["diagnostic_row"])
        print(f"Completed repetition {rep + 1}/{repetitions}", flush=True)
    pipeline._write_csv(output / "per_rep_metrics.csv", metrics)
    pipeline._write_csv(output / "per_rep_diagnostics.csv", diagnostics)
    pipeline._write_coefficients(
        output / "coefficient_estimates.csv", coefficients, int(config["d"])
    )
    contrasts = pipeline._paired_rows(metrics)
    pipeline._write_csv(output / "paired_contrasts.csv", contrasts)
    pipeline._write_csv(output / "paired_summary.csv", reporting.build_paired([
        {"config": config, "contrasts": contrasts}
    ]))
    write_json(output / "summary_metrics.json",
               pipeline._summary(metrics, coefficients, config))
