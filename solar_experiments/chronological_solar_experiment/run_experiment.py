"""Run the bundled strict chronological last-frame solar experiment locally."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent
TRAIN_FRACTIONS = [index / 100 for index in range(1, 11)]
CANDIDATE_MULTIPLIERS = [0.01, 0.10, 0.20, 0.50, 1.0, 2.0]


def float_list(value):
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc
    if not values or any(not math.isfinite(number) for number in values):
        raise argparse.ArgumentTypeError("Expected finite numbers.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Duplicate grid values are not allowed.")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-seed", type=int, default=397)
    parser.add_argument(
        "--train-fractions", type=float_list, default=None,
        help="Fractions of the later 20%%, default 0.01,...,0.10.",
    )
    parser.add_argument(
        "--candidate-multipliers", type=float_list, default=None,
        help="Candidate counts relative to training size; default .01,.1,.2,.5,1,2.",
    )
    parser.add_argument("--outer-reps", type=int, default=None)
    parser.add_argument("--inner-reps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=BUNDLE_ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=BUNDLE_ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=BUNDLE_ROOT / "outputs")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate metadata and print the run plan without fitting models.",
    )
    parser.add_argument(
        "--verify-data", action="store_true",
        help="Check the downloaded data's SHA-256 hashes before planning or running.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip per-repetition diagnostic figures; keep all numeric results.",
    )
    parser.add_argument(
        "--no-candidate-csv", action="store_true",
        help="Skip individual candidate decisions and aggregate acceptance matrices.",
    )
    return parser


def validate_args(args):
    if args.global_seed < 0:
        raise ValueError("--global-seed must be nonnegative.")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie strictly between 0 and 1.")
    for name in ("threads", "outer_reps", "inner_reps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.train_fractions is not None and any(
        not 0 < value < 1 for value in args.train_fractions
    ):
        raise ValueError("--train-fractions must lie strictly between 0 and 1.")
    if args.candidate_multipliers is not None and any(
        value <= 0 for value in args.candidate_multipliers
    ):
        raise ValueError("--candidate-multipliers must be positive.")
    for name in ("data_dir", "cache_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def configure_environment(threads):
    # Set the backend before importing the bundled evaluation helpers.
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "temporal-experiment-mpl")
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def verify_data(data_dir):
    from experiment_core import artifacts

    sources = resolve_data_sources(data_dir)
    artifacts.load_training_only_manifest(sources, verify_csv_hashes=True)
    print("Verified the five-file epoch-75 data bundle.", flush=True)


def resolve_data_sources(data_dir, cache_dir=None):
    from experiment_core import artifacts

    return artifacts.resolve_split_artifacts(
        split_label="chronological_split",
        training_metadata_path=data_dir / "training_metadata.json",
        real_pool_csv_path=data_dir / "test_chronological.csv",
        synthetic_samples_path=data_dir / "synthetic_samples_chronological_split.csv",
        cache_root=cache_dir,
    )


def build_plan(args):
    from experiment_core import artifacts, temporal_artifact_data as data

    sources = resolve_data_sources(args.data_dir, args.cache_dir)
    sources["feature_view"] = "last_frame"
    summary = data.load_temporal_source_summary(sources)
    multipliers = args.candidate_multipliers or list(CANDIDATE_MULTIPLIERS)
    outer_reps = args.outer_reps or 30
    inner_reps = args.inner_reps or 5
    planned_splits = []
    fractions = args.train_fractions or TRAIN_FRACTIONS
    fractions_and_counts = [
        (fraction, data.resolve_chronological_protocol_counts(
            summary["n_real"], downstream_train_fraction=fraction,
        ))
        for fraction in fractions
    ]
    protocol = data.STRICT_CHRONOLOGICAL_DRE_DOWNSTREAM_PROTOCOL
    for index, (fraction, counts) in enumerate(fractions_and_counts):
        actual_counts = data.resolve_chronological_protocol_counts(
            summary["n_real"], downstream_train_fraction=1.0 - (1.0 - fraction),
        )
        planned_splits.append({
            "split_index": index,
            "split_name": f"split_{index:02d}_train_n{counts['n_train']}_test_n{counts['n_test']}",
            "train_fraction": fraction,
            "test_fraction": 1.0 - fraction,
            **counts,
            "actual_n_train": actual_counts["n_train"],
            "actual_n_test": actual_counts["n_test"],
            "x_dim": summary["x_dim"],
        })
    return {
        "execution_mode": "local_sequential",
        "split_label": "chronological_split",
        "strict_chronological": True,
        "feature_view": sources["feature_view"],
        "temporal_split_protocol": protocol,
        "global_seed": args.global_seed,
        "alpha": args.alpha,
        "outer_reps": outer_reps,
        "inner_reps": inner_reps,
        "repetitions_per_cell": outer_reps * inner_reps,
        "n_cells": len(planned_splits) * len(multipliers),
        "n_repetitions": len(planned_splits) * len(multipliers) * outer_reps * inner_reps,
        "threads": args.threads,
        "train_fractions": [entry["train_fraction"] for entry in planned_splits],
        "candidate_train_multipliers": multipliers,
        "data_artifacts": artifacts.jsonable_artifact_paths(sources),
        "source_summary": summary,
        "split_plan": planned_splits,
        "save_histograms": not args.no_plots,
        "save_candidate_csv": not args.no_candidate_csv,
    }


def build_config(plan, split, multiplier, cache_path):
    from experiment_core import run_pipeline

    config = run_pipeline.build_full_config()
    config["seed"] = plan["global_seed"]
    config["pipeline"].update({
        "reps": plan["outer_reps"],
        "inner_reps": plan["inner_reps"],
        "alpha": plan["alpha"],
        "n_candidates": max(1, int(round(split["n_train"] * multiplier))),
        "save_histograms": plan["save_histograms"],
        "save_candidate_csv": plan["save_candidate_csv"],
        "compute_analytic_covariance": True,
    })
    config["ratio"]["lgbm_params"]["n_jobs"] = plan["threads"]
    config["artifacts"].update({
        "split_label": "chronological_split",
        "feature_view": plan["feature_view"],
        "temporal_cache_manifest_path": str(cache_path),
        "temporal_split_protocol": plan["temporal_split_protocol"],
        "pipeline_test_fraction": split["test_fraction"],
        "dre_real_fraction": 0.8,
        "m_class_tail_flux_threshold_raw": 1e-5,
    })
    return config


def save_json(path, payload):
    from experiment_core import evaluation

    evaluation.save_json(str(path), payload)


def run_cell(plan, split, multiplier, cell_index, cache_path):
    from experiment_core import run_pipeline as pipeline, user_module_temporal as model

    label = format(multiplier, ".12g").replace(".", "p")
    run_dir = Path(split["split_dir"]) / "runs" / f"cand_{label}x_train"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = build_config(plan, split, multiplier, cache_path)
    config["output_base_dir"] = str(run_dir.parent)
    config["experiment_name"] = run_dir.name
    rep_plan = pipeline.build_rep_plan(
        plan["global_seed"], plan["outer_reps"], plan["inner_reps"],
    )
    summary_config = pipeline.copy_config_with_run_seeds(
        config, split_seed=rep_plan[0]["split_seed"], dre_seed=rep_plan[0]["dre_seed"],
    )
    try:
        actual_data_summary = pipeline.summarize_data_split(pipeline.load_summary_data(summary_config))
    finally:
        model.clear_runtime_caches()
    for key in ("n_train", "n_test", "n_ratio_train", "x_dim"):
        expected = split[f"actual_{key}"] if key in ("n_train", "n_test") else split[key]
        if actual_data_summary[key] != expected:
            raise ValueError(f"Planned and loaded {key} differ.")
    # Preserve the source grid's reporting and candidate-budget reference sizes.
    data_summary = {
        key: split[key] for key in ("n_train", "n_test", "n_ratio_train", "x_dim", "n_real_pool", "n_downstream_pool")
    }
    save_json(run_dir / "run_manifest.json", {
        "execution_mode": "local_sequential",
        "active_model_module": pipeline.ACTIVE_MODEL_MODULE,
        "config": config,
        "expected_reps": len(rep_plan),
        "rep_plan": rep_plan,
        "data_summary": data_summary,
        "actual_data_summary": actual_data_summary,
    })
    payload_paths = []
    acceptance_paths = []
    for entry in rep_plan:
        start = time.monotonic()
        print(
            f"Cell {cell_index}/{plan['n_cells']} | train={split['actual_n_train']} | "
            f"candidates={config['pipeline']['n_candidates']} | "
            f"repetition {entry['rep'] + 1}/{len(rep_plan)}", flush=True,
        )
        rep_dir = run_dir / "reps" / f"rep_{entry['rep']:03d}"
        rep_dir.mkdir(parents=True)
        try:
            output = pipeline.run_single_rep(config=config, **entry)
            pipeline.save_local_rep_outputs(
                str(rep_dir), output["plot_bundle"], entry["rep"], config,
            )
            payload_path = rep_dir / "rep_metrics.json"
            pipeline.save_rep_payload(str(payload_path), output)
            payload_paths.append(payload_path)
            acceptance_paths.append(str(rep_dir / pipeline.get_candidate_acceptance_filename(entry["rep"])))
            del output
        finally:
            model.clear_runtime_caches()
        print(f"  completed in {time.monotonic() - start:.1f}s", flush=True)

    payloads = [pipeline.load_rep_payload(str(path)) for path in payload_paths]
    aggregate = pipeline.build_aggregate_state(payloads)
    pipeline.write_aggregate_outputs(
        str(run_dir), pipeline.get_figures_dir(str(run_dir)), config, data_summary, aggregate,
    )
    if config["pipeline"]["save_candidate_csv"]:
        for method in ("uniform", "theory", "dre_only", "influence_only"):
            pipeline.save_acceptance_matrix_from_paths(
                str(run_dir / f"{method}_acceptance_by_rep.csv"), acceptance_paths,
                payloads[0]["q_prop_pool_size"], acceptance_key=f"{method}_accepted",
            )
        shutil.copy2(
            run_dir / "reps/rep_000/candidate_diagnostics_rep0.csv",
            run_dir / "candidate_diagnostics_rep0.csv",
        )
    return {
        "split_index": split["split_index"],
        "train_fraction": split["train_fraction"],
        "test_fraction": split["test_fraction"],
        "candidate_train_multiplier": multiplier,
        "target_n_candidates": config["pipeline"]["n_candidates"],
        "run_dir": str(run_dir),
    }


def run_experiment(args, plan):
    # Fail before reading the large CSVs if the classifier cannot be imported.
    import lightgbm  # noqa: F401
    from threadpoolctl import threadpool_limits
    from experiment_core import analyze_grid, temporal_artifact_data as data

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir / f"strict_{plan['feature_view']}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    plan["experiment_dir"] = str(output)
    for split in plan["split_plan"]:
        split["split_dir"] = str(output / "splits" / split["split_name"])
    save_json(output / "experiment_manifest.json", plan)
    print(f"Output: {output}", flush=True)
    print("Preparing normalized data cache (reused on later runs)...", flush=True)
    cache_path = data.prepare_temporal_cache(plan["data_artifacts"])
    plan["temporal_cache_manifest_path"] = str(cache_path)
    save_json(output / "experiment_manifest.json", plan)
    print(f"Cache ready: {cache_path}", flush=True)
    cell_index = 0
    with threadpool_limits(limits=plan["threads"]):
        for split in plan["split_plan"]:
            split_dir = Path(split["split_dir"])
            split_dir.mkdir(parents=True)
            children = {"child_runs": []}
            save_json(split_dir / "split_children.json", children)
            for multiplier in plan["candidate_train_multipliers"]:
                cell_index += 1
                child = run_cell(plan, split, multiplier, cell_index, cache_path)
                children["child_runs"].append(child)
                save_json(split_dir / "split_children.json", children)
            save_json(split_dir / "split_complete.json", children)
    result = analyze_grid.analyze_grid_experiment(output)
    save_json(output / "analysis_complete.json", result)
    print(f"Completed {plan['n_cells']} cells. Results: {output / 'analysis'}", flush=True)
    return output


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    configure_environment(args.threads)
    if args.verify_data:
        verify_data(args.data_dir)
    plan = build_plan(args)
    if args.dry_run:
        from experiment_core.portable_metadata import portable_values

        print(json.dumps(portable_values(plan, BUNDLE_ROOT), indent=2))
        return
    print(
        f"Strict chronological {plan['feature_view']}: {plan['n_cells']} cells, "
        f"{plan['repetitions_per_cell']} repetitions/cell "
        f"({plan['n_repetitions']} total), {plan['threads']} CPU threads.", flush=True,
    )
    run_experiment(args, plan)


if __name__ == "__main__":
    main()
