# Standalone chronological temporal solar experiments

This source package contains the Python implementation and instructions for the chronological solar regression experiment. The large data assets are distributed separately. After downloading and extracting the five input files into `data/`, the populated working directory contains the same 27,373 real observations and 100,000 epoch-75 synthetic samples as the Great Lakes batch experiment, with their metadata. The entry script runs every cell and repetition sequentially, then writes combined results. It does not submit jobs or depend on either original repository.

## Run

Before running, download and extract the five epoch-75 input files directly into this folder's `data/` directory, preserving their filenames. They are the same inputs used for the corresponding Great Lakes batch experiment. A source-only ZIP is not a populated working directory; see [data/README.md](data/README.md) for the required files.

Use Python 3.12 and the dependencies in `requirements.txt`. In an existing environment with those dependencies, the full experiment is one command from this folder:

```bash
python run_experiment.py
```

For a fresh Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py
```

All default paths are relative to the script, which can also be invoked from another working directory. Python packages must be installed once; running the experiment does not download data or retrain the generator.

The default full endpoint grid has **60 cells and 9,000 repetitions**. Progress is printed before and after every repetition.

## Visualize saved results

After the experiment finishes, generate the figures and tables with:

```bash
python visualize.py
```

This selects the latest completed run under `outputs/` and writes `visualizations/all/` and `visualizations/tails/` inside that run. It reads saved aggregate results only. The full-test and M1.0+ heatmaps, best-method maps, MSE mean ± sample-SD tables, and covariance tables use `analysis/grid_summary.csv`. The script also supports smaller custom grids.

To select a particular run or a downloaded Great Lakes analysis folder:

```bash
python visualize.py --analysis-dir outputs/strict_last_frame_<timestamp>/analysis
```

All visualization inputs come from the analysis folder. PNG and PDF are both saved by default. Optional `--font-style stix` or `--font-style computer-modern` selects embedded publication fonts without requiring LaTeX. `--formats png`, `--dpi`, and `--output-dir` customize exports. Heatmaps retain the original method colors and guarded MSE/covariance selection rule. Visualization creates no additional JSON metadata.

## Inspect the run plan

After placing the input files in `data/`, inspect the full plan without fitting models, or check the downloaded data against their recorded hashes. The plan reads input metadata to determine the split sizes; `--dry-run` is not a data-free command. Use `python run_experiment.py --help` to inspect CLI options before downloading data.

```bash
python run_experiment.py --dry-run
python run_experiment.py --verify-data --dry-run
```

## Experiment settings

The experiment uses only the 20 SHARP features at the final frame of each temporal history. Features use signed `log1p`, and endpoint flux uses `log10`; both are standardized with the chronological generator's saved training statistics.

The real pool contains 27,373 histories from 2022–2024. Rows are sorted by their history-window endpoint times; original row index breaks timestamp ties.

1. The earliest 80% (21,898 observations) are used for density-ratio estimation.
2. The remaining 20% (5,475 observations) form the downstream regression pool.
3. The earliest 1%, 2%, ..., 10% of that downstream pool train the regression; all subsequent observations are tested. The real partitions are fixed across repetitions, and training prefixes are nested across fractions.

The candidate-budget multipliers are `0.01, 0.10, 0.20, 0.50, 1.00, 2.00` times the real regression training count. The acceptance target is `alpha=0.05`, the seed is 397, and each cell has 30 outer groups × 5 inner repetitions. Seeds vary DRE partitions, proposal draws, and acceptance decisions. These are not repeated real-data splits. Candidate budgets refer to proposed samples; accepted counts are random.

The five methods are real-only, uniform acceptance, DRE-only acceptance, influence-only acceptance, and theory acceptance (REAL-Syn). The bundle retains the LightGBM source classifier with Platt calibration, its 500-observation minimum leaf size, the regression/influence/weighting calculations, and overall and M-class-tail MSE. The tail threshold is raw flux `>= 1e-5 W/m^2`; MSE is measured in standardized `log10` flux units.

For a smaller custom grid:

```bash
python run_experiment.py \
  --train-fractions 0.05,0.10 \
  --candidate-multipliers 0.5,1,2 \
  --outer-reps 2 --inner-reps 3 --threads 4
```

Cells and repetitions run serially; the four-thread default applies only to LightGBM and numerical libraries. It matches the batch configuration's CPU allocation and describes a runtime setting, not a machine specification or record of generator hardware. Use `--no-plots` to skip diagnostic figures or `--no-candidate-csv` to omit individual candidate decisions and acceptance matrices. Numeric repetition and aggregate results are still saved. `--data-dir`, `--cache-dir`, and `--output-dir` accept alternative locations.

## Data and outputs

After downloading and extracting the data assets, `data/` contains approximately **2.72 GB**: all 27,373 real histories, all 100,000 synthetic histories, training metadata, the sample summary, and the artifact manifest. This size describes the populated directory, not the source ZIP. [data/README.md](data/README.md) lists the files and provenance. Both CSVs preserve the original histories; the runner reads the final-frame predictors and endpoint flux. Keep these files directly in `data/` rather than linking to another checkout. No generator checkpoint or PyTorch installation is needed for downstream analysis.

The first run streams the selected features into a normalized NumPy cache under `cache/`. Later runs reuse it when the input paths, files, and feature view are unchanged. Copying the bundle to another location rebuilds the cache from its local CSVs. Per-repetition arrays and fitted DRE models are released between repetitions to keep a long local run from accumulating them.

Each invocation creates a new directory under `outputs/`:

```text
outputs/strict_last_frame_<timestamp>/
  experiment_manifest.json
  splits/<split>/
    split_children.json
    split_complete.json
    runs/<candidate-budget>/
      run_manifest.json
      results_details.csv
      summary_metrics.json
      results_summary.txt
      *_acceptance_by_rep.csv
      reps/rep_000/
        rep_metrics.json
        candidate_acceptance_rep0.csv
        figures/
  analysis/
    grid_summary.csv
    grid_summary.json
    grid_test_mse_pivot.csv
    grid_m_class_tail_mse_pivot.csv
    grid_empirical_trace_cov_pivot.csv
  analysis_complete.json
  visualizations/       # created separately by visualize.py
    all/
    tails/
```

Each repetition's JSON retains the full coefficient vectors for every method to calculate empirical covariance across repetitions. Coefficient covariance is conditional on the fixed real partition and summarizes DRE/synthetic randomness. Completed repetitions and cells are saved as they finish; an interrupted invocation remains a partial run. Rerunning the script starts a new output directory. Saved JSON uses paths relative to the containing JSON file and omits machine information; the input manifest uses filenames only. Analysis resolves those relative references, so completed results can be moved with the folder and analyzed again.

The standalone copy preserves the original repository's behavior, including its split rounding. At 6% and 10%, the loader reconstructs the training fraction as `1 - test_fraction`, selecting 329 and 547 rows while the grid plan specifies 328 and 548. Candidate budgets and the standard summary fields retain those original planned sizes. Additional `actual_n_train`/`actual_n_test` fields in the grid manifest and `actual_data_summary` in each run manifest expose the loaded sizes. No rounding correction is applied. Seed derivation and scientific callbacks are retained.

The generator fits normalization and model parameters on 2010–2021, trains for a fixed 75 epochs, and saves only the final checkpoint. It does not evaluate the 2022–2024 downstream pool or use it to choose an epoch. The boundary-matched synthetic flux proposal is fitted only on the 74,890 generator-training rows: the 99.5th-percentile threshold is `1.375404324335246e-05`, the upper flux limit is `8.097962127067e-04`, and the Pareto density exponent is `2.3749013017254264`. This bundle uses the same new samples, normalization, split rules, scientific methods, seeds, and repetition settings as the current Great Lakes run.

Use native Python execution; Docker is not part of this bundle. Large CSVs, caches, and results are excluded from Git. Distribute the source ZIP with the separate input-data download, or explicitly include all five inputs when distributing a populated working directory.
