"""Command-line entry point for REAL-Syn numerical experiments."""
from __future__ import annotations
import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from common.io import write_json

ROOT = Path(__file__).resolve().parent


def load_registry():
    return json.loads((ROOT / "experiments.json").read_text(encoding="utf-8"))


def run_experiment(name, quick=False, output=None):
    entry = load_registry()[name]
    config_path = ROOT / entry["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(output).resolve() if output else ROOT / "results" / f"{name}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    planned = int(config.get("repetitions", config.get("pipeline", {}).get("reps", 100)))
    repetitions = (3 if name == "table2" else 2) if quick else planned
    write_json(output / "run_info.json", {
        "experiment": name, "run_kind": "quick" if quick else "full",
        "planned_repetitions": planned, "executed_repetitions_per_setting": repetitions,
        "paper_location": entry["paper"],
    })
    if quick:
        print("QUICK CHECK: reduced repetitions; not paper estimates.", flush=True)
    kind = entry["kind"]
    if kind in ("ols", "sweep", "highdim"):
        config["output_base_dir"] = str(output)
        if kind == "ols":
            config["pipeline"]["reps"] = repetitions
        write_json(output / "resolved_config.json", config)
        if kind == "ols":
            from common.ols_pipeline import run
            actual = run(str(output / "resolved_config.json"), str(output))
        elif kind == "highdim":
            from experiments.highdim_ols.pipeline import run
            actual = run(str(output / "resolved_config.json"), str(output),
                         quick_repetitions=2 if quick else None)
        else:
            from experiments.acceptance_budget.sweep import run_sweep
            actual = run_sweep(str(output / "resolved_config.json"),
                               output_dir=str(output), quick=quick)
        output = Path(actual)
    else:
        write_json(output / "resolved_config.json", config)
        if kind == "logistic":
            from experiments.logistic_regression.run import run
            run(config, output, repetitions)
        else:
            source = json.loads((ROOT / config["source_config"]).read_text(encoding="utf-8"))
            write_json(output / "resolved_source_config.json", source)
            modules = {
                "equal_budget": "weighting_resampling",
                "full_pool": "full_pool_importance_weighting",
                "ratio": "density_ratio_sensitivity",
                "grid": "proposal_mismatch", "stress": "proposal_mismatch",
            }
            runner = importlib.import_module(f"experiments.{modules[kind]}.run")
            kwargs = {"quick": quick} if kind == "equal_budget" else {}
            if kind in ("grid", "stress"):
                kwargs["stress"] = kind == "stress"
            runner.run(config, source, output, repetitions, **kwargs)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)
    return output


def main():
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=registry)
    parser.add_argument("--list", action="store_true", help="List paper-to-experiment mappings.")
    parser.add_argument("--quick", action="store_true",
                        help="Use two repetitions (three for Table 2); never use these as paper estimates.")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused.")
    args = parser.parse_args()
    if args.list:
        for name, entry in registry.items():
            print(f"{name:24} {entry['paper']}")
        return
    if args.experiment is None:
        parser.error("Select an experiment, or use --list.")
    run_experiment(args.experiment, args.quick, args.output)


if __name__ == "__main__":
    main()
