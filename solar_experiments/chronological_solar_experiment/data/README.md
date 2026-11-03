# Chronological temporal data assets

Download the data files from Figshare:

https://figshare.com/s/c18aa163f9cf4f4deacd

The data assets are distributed separately from the source ZIP. After downloading and extracting the data assets, this directory must contain the five files listed below, with filenames unchanged. They are the same inputs used for the corresponding Great Lakes batch experiment. Downloading the source alone leaves only this README in `data/`; both CSVs and all three JSON files must be supplied.

| File | Purpose |
| --- | --- |
| `test_chronological.csv` | 27,373 real 60-frame histories with endpoint years 2022–2024, reserved for the downstream solar experiment. |
| `synthetic_samples_chronological_split.csv` | 100,000 histories generated from the new final epoch-75 checkpoint. |
| `training_metadata.json` | Training-only provenance, full feature schema, and training normalization. |
| `sample_summary_chronological_split.json` | Checkpoint hash and epoch, sampling settings, normalization, and the training-only flux proposal. |
| `artifact_manifest.json` | Downstream row count and the four input files' names, byte sizes, and SHA-256 hashes. |

The generator trains only on 2010–2021 and does not evaluate these later real observations. Its flux proposal is also fitted only on generator-training rows. The raw source CSV and trained checkpoint are unnecessary for downstream execution. Once downloaded, the five input files occupy approximately 2.72 GB. Their JSON metadata uses filenames rather than absolute paths and omits machine information.

Run `python run_experiment.py` from the parent directory. The runner reads these local files and creates its own normalized cache. No sibling checkout or scheduler is needed.
