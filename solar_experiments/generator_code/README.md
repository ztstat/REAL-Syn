# Chronological temporal SHARPs generator

Train the 60-frame conditional OT-CFM generator exclusively on 2010–2021 observations for 75 epochs, save only the final epoch-75 checkpoint, and fit the synthetic flux proposal from the same generator training rows. The chronological solar-experiment observations are never evaluated by this workflow.

## Quick start

Use Python 3.12 with the pinned dependencies in `requirements.txt`, in an existing environment or a local virtual environment. Do not use Docker for this bundle.

The source ZIP supplies code and instructions; the large input CSV is distributed separately. Download `SHARPs_flux_temporal_u12m_w60_s10_wide.csv` into this folder's `data/` directory. This is the only external data file needed for a fresh generator setup. Training creates the chronological split CSVs, condition reference, and final checkpoint locally. The following commands train the model and then explicitly sample its newly saved epoch-75 checkpoint. From the directory containing `generator_code/`:

```bash
python generator_code/train_generator.py
python generator_code/generate_synthetic_samples.py \
  --checkpoint generator_code/artifacts/chronological_training_only_seed3097/model_epoch_075.pt
```

Omitting `--checkpoint` selects the same `artifacts/chronological_training_only_seed3097/model_epoch_075.pt` relative to this package; it must already exist from training. A source-only ZIP does not supply a pretrained checkpoint.

The default trainer creates any missing `data/train_chronological.csv` and `data/test_chronological.csv` using the fixed endpoint-year cutoff: through 2021 for training, 2022 onward for testing. It also fits the condition reference from training rows, writes `condition_reference_summary.json` in the run's artifact directory, and embeds that reference in the epoch-75 checkpoint. Sampling uses the embedded reference automatically. Existing split files are never overwritten. If both exist, the full CSV is not opened. A prepared training CSV outside the default data path can still be supplied directly with `--data-csv`.

From this folder, run:

```bash
python train_generator.py
python generate_synthetic_samples.py \
  --checkpoint artifacts/chronological_training_only_seed3097/model_epoch_075.pt
```

Training defaults to seed 3097, 75 epochs, CPU, and batch size 64. Generation defaults to 100,000 histories and uses CUDA, then MPS, then CPU when `--device auto` is selected. Both commands accept an explicit device. These are configurable runtime defaults, not statements about the hardware used for a previously reported run. Supply the full CSV at its `data/` location for automatic splitting; `--data-csv` specifies prepared training input and rejects later-year rows before loading their feature/flux values. See [data/README.md](data/README.md).

Outputs use the new `artifacts/chronological_training_only_seed3097/` directory. Existing artifacts are protected. A custom training destination can be used explicitly when sampling:

```bash
python train_generator.py --output-dir artifacts/my_run
python generate_synthetic_samples.py \
  --checkpoint artifacts/my_run/model_epoch_075.pt
```

The sampler defaults to the final epoch-75 checkpoint named above; no epoch argument is needed. It fails if that model is absent. Keep the downloaded full temporal CSV in `data/` as source data; chronological training uses the generated `train_chronological.csv`.

## Default training recipe

| Setting | Value |
| --- | --- |
| Modeled state | 60 frames × 20 features = 1,200 coordinates |
| Condition | Training-standardized `log10(endpoint_flux)` |
| Feature transform | Signed `log1p`, then training-only standardization pooled over frames per feature |
| Training input | Training-only CSV, fixed endpoint-year cutoff 2021 |
| Training observations | 2010–2021: 74,890 rows |
| Seed / default device / loader workers | 3097 / CPU / 0 |
| Model | Residual MLP, width 512, 6 blocks; 4,504,624 parameters |
| Time / condition embedding widths | 64 / 128 |
| Epochs / batch size | 75 / 64 |
| Optimizer | AdamW; learning rate 0.001, weight decay 0.0001 |
| Gradient clipping | Norm cap 5.0 |
| Objective | Exact minibatch OT assignment and unweighted velocity MSE |
| Path / time margin | Linear, zero path noise; `eps_t=0.001` |
| Tail weighting / tail batching | Disabled |


Input must contain all 60 frames of 20 SHARPs variables, `HARPNUM`, `history_window_end`, `frame_059__flux`, and the matching endpoint `flux`. Preparing missing splits reads the source's endpoint-date column and copies unchanged raw rows; it does not evaluate test feature or flux values. Subsequent training opens only the training CSV. All usable training rows are used; there is no second random or year-fraction split. Normalization and the flux reference use exactly those usable training rows.

The trainer does not construct a validation loader, compute holdout losses, or select a best epoch. The prepared `test_chronological.csv` is reserved for the solar experiment. Epochs are counted from one; each epoch is one pass through the training observations. By default, it writes only `model_epoch_075.pt` after completing all 75 epochs. `--epochs` changes the training length for custom runs; only the final epoch is saved, and sampling from a custom endpoint requires its explicit `--checkpoint` path. The checkpoint contains its epoch, architecture, schema, training normalization, training-only flux reference, and weights.

`training_metadata.json` records the training filename, fixed cutoff, usable row counts, model/schema/normalization, training settings, seed, final checkpoint epoch, and flux reference. Loss history is stored once, in `history.csv` as `epoch,train_mse`. `condition_reference_summary.json` contains the proposal parameters, fit counts, and training-only provenance. Saved JSON and checkpoint references use filenames only. Machine details, runtime/library versions, timestamps, raw command arguments, and duplicate diagnostic summaries are omitted. Metadata and history are updated after every completed epoch; model weights are written only when the final epoch completes.

The default seed is 3097 and epoch 75 is the fixed training endpoint.

## Training-only flux proposal

The default proposal is uniform in raw endpoint flux below the training 99.5th percentile, with an upper-truncated Pareto tail above it. The ordinary Pareto shape estimate is `n_tail / sum(log(flux / threshold))`; generation subsequently truncates the fitted law at the training maximum. The mixture weight matches the weighted bulk/tail densities at the threshold.

The reference fitted from all 74,890 generator training rows (2010–2021) has the following values; training regenerates it from the downloaded data:

| Quantity | Training-only value |
| --- | ---: |
| Minimum raw flux | `9.999999717180683e-10` |
| 99.5% threshold | `1.375404324335246e-05` |
| Maximum raw flux | `8.097962127067e-04` |
| Tail observations | 375 |
| Pareto shape | `1.3749013017254264` |
| Density exponent | `2.3749013017254264` |
| Boundary-matched tail probability | `0.4201880258755406` |

At 100,000 samples this gives 57,981 uniform-bulk conditions and 42,019 tail conditions. Each training run fits and embeds its own reference, including the threshold, minimum, maximum, and shape. Sampling uses that embedded reference by default. Explicit reference JSONs must declare training-only provenance; an explicit `--condition-reference-csv` is checked against the checkpoint's year cutoff before its flux values are loaded.

Raw sampled flux is transformed to `log10(flux)` and standardized with the checkpoint's training statistics. Generation uses NumPy flux seed `seed + 24680`, CPU Gaussian noise seed `seed + 1000`, batch size 64, and 64-step RK4.

Fixed-flux generation is also available:

```bash
python generate_synthetic_samples.py \
  --condition-mode explicit_log10 --log10-flux -5.0 --samples-per-condition 100
```

Samples are written separately for each run and epoch:

```text
samples/<run-name>/epoch_075/chronological_split/
  boundary_matched_q995_n100000_seed3097/
    synthetic_samples_chronological_split.csv
    sample_summary_chronological_split.json
```

Each row contains the endpoint condition and 1,200 frame-major SHARPs coordinates. Per-frame flux trajectories are not generated. The compact sampling summary records output/checkpoint filenames, checkpoint hash/epoch, training-only reference, condition normalization, seed, batch size, RK4 steps, and mixture settings/counts. Condition normalization and the saved standardized condition column are retained for the downstream solar loader. Imported reference JSON is reduced to the same essential fields before export. Identical floating-point results are not guaranteed across hardware or library versions.
