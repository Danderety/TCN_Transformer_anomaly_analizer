from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_STAGES = {
    'prepare_data': 'scripts/01_prepare_data.py',
    'train_tcn_ae': 'scripts/02_train_baseline_tcn_ae.py',
    'train_tcn_transformer_ae': 'scripts/03_train_tcn_transformer_ae.py',
    'train_tcn_transformer_ensemble': 'scripts/08_train_tcn_transformer_ensemble.py',
    'evaluate_detection': 'scripts/04_evaluate_detection.py',
    'evaluate_localization': 'scripts/05_evaluate_localization.py',
    'evaluate_adaptive_threshold': 'scripts/06_evaluate_adaptive_threshold.py',
    'generate_report_assets': 'scripts/07_generate_report_assets.py',
}

DATASET_SOURCES = {
    'lo2': {
        'target_dir': 'data/raw/lo2',
        'url': 'https://doi.org/10.5281/zenodo.14265858',
        'note': 'LO2 / Light-OAuth2 microservice API anomaly dataset. Put extracted log files under data/raw/lo2/.',
    },
    'loghub2': {
        'target_dir': 'data/raw/loghub2',
        'url': 'https://zenodo.org/records/8275861',
        'note': 'Loghub-2.0 collection. Put extracted log files under data/raw/loghub2/.',
    },
    'rcaeval': {
        'target_dir': 'data/raw/rcaeval',
        'url': 'https://github.com/phamquiluan/RCAEval',
        'note': 'RCAEval benchmark repository/data. Put extracted logs under data/raw/rcaeval/.',
    },
}


def _load_script(stage):
    if stage not in SCRIPT_STAGES:
        raise ValueError(f'Unknown notebook workflow stage: {stage}')
    script_path = PROJECT_ROOT / SCRIPT_STAGES[stage]
    spec = spec_from_file_location(f'notebook_stage_{stage}', script_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_stage(stage, config_path):
    module = _load_script(stage)
    return module.main(str(config_path))


def run_full_pipeline(config_path, include_prepare=True):
    stages = [
        'prepare_data',
        'train_tcn_ae',
        'train_tcn_transformer_ae',
        'train_tcn_transformer_ensemble',
        'evaluate_detection',
        'evaluate_localization',
        'evaluate_adaptive_threshold',
        'generate_report_assets',
    ]
    if not include_prepare:
        stages = stages[1:]
    for stage in stages:
        print(f'\n=== {stage} ===')
        run_stage(stage, config_path)


def load_metric_tables(output_dir='outputs'):
    output_dir = Path(output_dir)
    paths = {
        'detection': output_dir / 'metrics' / 'detection_metrics.csv',
        'localization': output_dir / 'metrics' / 'localization_metrics.csv',
        'adaptive_threshold': output_dir / 'metrics' / 'adaptive_threshold_metrics.csv',
        'final_comparison': output_dir / 'metrics' / 'final_comparison.csv',
    }
    return {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}


def dataset_links_table():
    return pd.DataFrame.from_dict(DATASET_SOURCES, orient='index').reset_index(names='dataset')
