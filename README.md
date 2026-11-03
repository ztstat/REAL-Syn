# REAL-Syn

**Risk-guided Estimation-aware Acceptance for Learning with Synthetic Data (REAL-Syn)**  

Official implementation of REAL-Syn, accepted at NeurIPS 2026.

## Repository structure

| Directory | Contents |
|---|---|
| [numerical_experiments/](numerical_experiments/README.md) | OLS and logistic simulations, configurations, figures, and mathematical tests |
| [solar_experiments/generator_code/](solar_experiments/generator_code/README.md) | Training and sampling a conditional flow-matching generator for temporal SHARPs histories |
| [solar_experiments/chronological_solar_experiment/](solar_experiments/chronological_solar_experiment/README.md) | Chronological solar regression, density-ratio estimation, and visualization of saved results |

## Numerical experiments

These CPU simulations generate their own data. They cover low- and high-dimensional OLS, logistic regression, importance weighting and SIR/SNIS comparisons, full-pool All-IW, density-ratio sensitivity, proposal mismatch, and acceptance-budget experiments. 

In a dedicated Python 3.11+ environment, starting at the repository root:

~~~sh
cd numerical_experiments
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --list
python run.py ols-d5 --quick
~~~

The last command exercises the entry point with reduced repetitions. Omit `--quick` to use the paper configuration; the two-repetition check does not produce paper estimates.

## Solar experiments and data

The [Solar guide](solar_experiments/README.md) describes two workflows with separate Python 3.12 dependency files. The generator fits its normalization and flux proposal on training-period observations and uses the final training epoch for sampling. The downstream experiment fits and evaluates regression methods on chronological partitions using a prepared synthetic pool. **Running the downstream comparison does not require training the generator or installing PyTorch.**

The Solar data are distributed separately through the [Figshare data share](https://figshare.com/s/c18aa163f9cf4f4deacd). Data files, trained weights, and experiment outputs are not included in this repository. See the [generator input list](solar_experiments/generator_code/data/README.md) and [downstream input list](solar_experiments/chronological_solar_experiment/data/README.md) before running either workflow.

## License

The code in this repository is released under the [MIT License](LICENSE).
