"""Run the local or stress proposal grid with paired candidate draws."""
from common.io import collect, write_json
from . import grid_core, grid_reporting, stress_core, stress_reporting


def run(config, source, output, repetitions, stress=False):
    core = stress_core if stress else grid_core
    reporting = stress_reporting if stress else grid_reporting
    grid = tuple((item["grid_cell"], item["mu_x"], item["sigma_x"]) for item in config["grid"])
    if grid != core.GRID or tuple(config["methods"]) != core.METHODS:
        raise ValueError("Grid or method configuration does not match the experiment.")
    analytic = core.analytic_grid(config, source)
    result = collect(config, source, repetitions, core)
    rows, ratios = result["rows"], result["ratio_rows"]
    outputs = {
        "per_rep_metrics.csv": rows,
        "parameter_vectors.csv": result["parameter_rows"],
        "ratio_diagnostics.csv": ratios,
        "sanity_checks.csv": result["sanity_row"],
    }
    if stress:
        summary, covariance = reporting.aggregate_methods(rows)
        severity, detail = reporting.severity_outputs(analytic, ratios, rows)
        outputs.update({
            "paired_contrasts.csv": reporting.build_pairwise_contrasts(rows),
            "ratio_severity_summary.csv": severity,
            "covariance_bootstrap.csv": reporting.build_covariance_bootstrap(
                rows, seed=int(config["seeds"]["analysis_seed"]),
                resamples=int(config["bootstrap_resamples"])),
        })
        write_json(output / "analytic_ratio_severity_matrices.json", {"cells": detail})
    else:
        summary, covariance = reporting.method_summary_rows_full(rows)
        outputs.update({
            "paired_contrasts.csv": reporting.paired_contrast_rows(rows),
            "ratio_severity_summary.csv": reporting.ratio_summary_rows(analytic, ratios),
            "diagnostics_summary.csv": reporting.diagnostics_summary_rows(rows),
        })
    outputs["method_summary.csv"] = summary
    outputs["coefficient_covariance_matrices.csv"] = covariance
    for filename, values in outputs.items():
        reporting.write_csv(output / filename, values)
    write_json(output / "analytic_ratio_severity.json", {"cells": analytic})
    write_json(output / "seeds.json", core.seed_manifest(config, source, repetitions))
