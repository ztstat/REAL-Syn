"""Run the joint-ratio shrinkage grid."""
from common.io import collect, write_json
from . import core, reporting


def run(config, source, output, repetitions):
    if tuple(config["methods"]) != core.METHODS:
        raise ValueError("Method configuration does not match the experiment.")
    regimes = tuple((item["name"], item["lambda"]) for item in config["ratio_regimes"])
    if regimes != core.RATIO_REGIMES:
        raise ValueError("Ratio grid does not match the experiment.")
    result = collect(config, source, repetitions, core)
    rows = result["rows"]
    summary, covariance = reporting.method_summary_rows_full(rows)
    tolerance = float(config["degradation_denominator_tolerance"])
    outputs = {
        "per_rep_metrics.csv": rows,
        "parameter_vectors.csv": result["parameter_rows"],
        "ratio_diagnostics.csv": result["ratio_rows"],
        "acceptance_diagnostics.csv": result["acceptance_rows"],
        "sanity_checks.csv": result["sanity_row"],
        "method_summary.csv": summary,
        "coefficient_covariance_matrices.csv": covariance,
        "ratio_diagnostics_summary.csv": reporting.ratio_summary_rows(result["ratio_rows"]),
        "acceptance_diagnostics_summary.csv": reporting.acceptance_summary_rows(result["acceptance_rows"]),
        "paired_contrasts.csv": reporting.paired_contrast_rows(rows),
        "covariance_bootstrap.csv": reporting.covariance_bootstrap_rows(
            rows, seed=int(config["seeds"]["analysis_seed"]),
            resamples=int(config["bootstrap_resamples"])),
        "degradation_summary.csv": reporting.degradation_rows(rows, tolerance),
        "gain_retained.csv": reporting.gain_retained_rows(rows, tolerance),
        "diagnostics_summary.csv": reporting.diagnostics_summary_rows(rows),
    }
    for filename, values in outputs.items():
        reporting.write_csv(output / filename, values)
    write_json(output / "seeds.json", core.seed_manifest(config, source, repetitions))
