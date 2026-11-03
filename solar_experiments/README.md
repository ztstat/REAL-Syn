# Solar experiments

Choose the workflow you need. Each command block below starts at the repository root and assumes a separate, activated **Python 3.12** environment for that component. The two workflows do not require one another's environment.

### Train and sample the generator

Place `SHARPs_flux_temporal_u12m_w60_s10_wide.csv` in [generator_code/data/](generator_code/data/README.md). The default trainer prepares the chronological splits, fits preprocessing and the flux reference on training-period rows, and saves the final epoch-75 checkpoint. Sampling uses that checkpoint. Neither the source CSV nor a pretrained checkpoint is included here.

~~~sh
cd solar_experiments/generator_code
python -m pip install -r requirements.txt
python train_generator.py --help
python generate_synthetic_samples.py --help
~~~

To train and then sample, in that same directory:

~~~sh
python train_generator.py
python generate_synthetic_samples.py --checkpoint artifacts/chronological_training_only_seed3097/model_epoch_075.pt
~~~

See the [generator guide](generator_code/README.md) for device options, the training recipe, and the output schema.

### Run chronological regression on prepared inputs

Place the **five files** listed in [chronological_solar_experiment/data/README.md](chronological_solar_experiment/data/README.md) directly in that data directory: the real and synthetic CSVs, training metadata, sample summary, and artifact manifest. Use the supplied manifest with its corresponding files; the downstream code is not a manifest generator.

~~~sh
cd solar_experiments/chronological_solar_experiment
python -m pip install -r requirements.txt
python run_experiment.py --help
python visualize.py --help
~~~

Once the data are in place, inspect the plan or verify the input bundle without fitting models:

~~~sh
python run_experiment.py --dry-run
python run_experiment.py --verify-data --dry-run
~~~

The dry run is **not** a data-free demonstration. To execute the full experiment, use `python run_experiment.py`; its default grid has 60 cells and 150 repetitions per cell. Run `python visualize.py` only after aggregate results exist. The [downstream guide](chronological_solar_experiment/README.md) documents the chronological partitions, planned versus actual split counts, and output tables.

[Figshare data share](https://figshare.com/s/c18aa163f9cf4f4deacd) · [Repository overview](../README.md)
