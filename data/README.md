# Dataset Placement

Raw datasets are not committed to the repository. Download them manually and
extract logs into these project folders:

| Dataset | Target folder | Source |
|---|---|---|
| LO2 | `data/raw/lo2/` | https://doi.org/10.5281/zenodo.14265858 |
| Loghub-2.0 | `data/raw/loghub2/` | https://zenodo.org/records/8275861 |
| RCAEval | `data/raw/rcaeval/` | https://github.com/phamquiluan/RCAEval |

After extracting a dataset, run `scripts/01_prepare_data.py` or use
`notebooks/04_final_results.ipynb`. The adapter layer lives in
`src/data/adapters.py`; if a dataset has labels/session ids in a custom file,
adapt that file rather than putting parsing logic into notebooks.
