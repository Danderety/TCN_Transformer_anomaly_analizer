from importlib.util import module_from_spec, spec_from_file_location
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import pandas as pd

from src.utils.config import load_config, save_config


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
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if (name.startswith('src.') and name != __name__) or name == 'train_common':
            sys.modules.pop(name, None)
    script_path = PROJECT_ROOT / SCRIPT_STAGES[stage]
    spec = spec_from_file_location(f'notebook_stage_{stage}', script_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_stage(stage, config_path):
    module = _load_script(stage)
    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        return module.main(str(config_path))
    finally:
        os.chdir(old_cwd)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _abs_project_path(path):
    path = Path(path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def make_notebook_config(config_path, dataset_name=None, output_dir=None):
    config_path = Path(config_path)
    config = load_config(config_path)
    dataset_name = dataset_name or config['data']['dataset_name']
    for key in ['raw_dir', 'processed_dir', 'splits_dir']:
        if key in config.get('data', {}):
            config['data'][key] = _abs_project_path(config['data'][key])
    if output_dir is not None:
        config.setdefault('project', {})['output_dir'] = _abs_project_path(output_dir)
    elif config.get('project', {}).get('output_dir'):
        config['project']['output_dir'] = _abs_project_path(config['project']['output_dir'])
    config.setdefault('notebook', {})
    config['notebook'].setdefault('compact_training_log', True)
    config['notebook'].setdefault('compact_training_log_lines', 12)
    tmp_dir = PROJECT_ROOT / 'outputs' / '_notebook_configs'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f'{dataset_name}_notebook_config.yaml'
    save_config(config, tmp_path)
    return tmp_path


def make_full_data_config(config_path, dataset_name=None, output_dir=None):
    config_path = make_notebook_config(config_path, dataset_name=dataset_name, output_dir=output_dir)
    config = load_config(config_path)
    for key in ['max_runs_per_scenario', 'max_lines_per_file', 'max_raw_rows', 'max_files']:
        config.get('data', {}).pop(key, None)
    dataset_name = dataset_name or config['data']['dataset_name']
    tmp_dir = PROJECT_ROOT / 'outputs' / '_notebook_configs'
    tmp_path = tmp_dir / f'{dataset_name}_full_notebook_config.yaml'
    save_config(config, tmp_path)
    return tmp_path


def make_memory_saver_config(
    config_path,
    dataset_name=None,
    output_dir=None,
    batch_size=16,
    max_localization_samples=512,
    epochs=100,
    patience=8,
):
    config_path = make_notebook_config(config_path, dataset_name=dataset_name, output_dir=output_dir)
    config = load_config(config_path)
    dataset_name = dataset_name or config['data']['dataset_name']
    config.setdefault('training', {})['batch_size'] = int(batch_size)
    config['training']['epochs'] = int(epochs)
    config['training']['patience'] = int(patience)
    config['training'].setdefault('loss_min_delta', 0.0)
    config['training'].setdefault('threshold_patience', 9)
    config['training'].setdefault('threshold_min_delta', 0.000001)
    config['training'].setdefault('threshold_min_relative_delta', 0.01)
    config['training'].setdefault('threshold_save_loss_tolerance', 0.05)
    config['training']['num_workers'] = 0
    config.setdefault('threshold', {})
    config['threshold'].setdefault('selection_policy', 'recall_fpr_tradeoff')
    config['threshold'].setdefault('target_recall', 0.99)
    config['threshold'].setdefault('max_validation_fpr', 0.25)
    config['threshold'].setdefault('fbeta_beta', 3.0)
    config['threshold'].setdefault('enforce_validation_recall', False)
    config.setdefault('evaluation', {})['max_localization_samples'] = int(max_localization_samples)
    config['evaluation'].setdefault('use_synthetic_validation_when_no_anomalies', True)
    config.setdefault('xai', {})['save_attention_matrices'] = 0
    config.setdefault('notebook', {})['memory_saver'] = True
    tmp_dir = PROJECT_ROOT / 'outputs' / '_notebook_configs'
    tmp_path = tmp_dir / f'{dataset_name}_memory_saver_notebook_config.yaml'
    save_config(config, tmp_path)
    return tmp_path


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


def load_training_histories(output_dir='outputs'):
    metrics_dir = Path(output_dir) / 'metrics'
    return {
        path.stem: pd.read_csv(path)
        for path in sorted(metrics_dir.glob('*_training_history.csv'))
    } if metrics_dir.exists() else {}


def load_metric_tables_many(output_dirs):
    rows = {}
    for dataset_name, output_dir in output_dirs.items():
        for table_name, df in load_metric_tables(output_dir).items():
            enriched = df.copy()
            enriched.insert(0, 'dataset', dataset_name)
            rows.setdefault(table_name, []).append(enriched)
    return {
        table_name: pd.concat(parts, ignore_index=True)
        for table_name, parts in rows.items()
        if parts
    }


def load_training_histories_many(output_dirs):
    histories = {}
    for dataset_name, output_dir in output_dirs.items():
        for name, df in load_training_histories(output_dir).items():
            histories[f'{dataset_name}/{name}'] = df
    return histories


def split_label_summary(dataset_states):
    rows = []
    for dataset_name, state in dataset_states.items():
        config_path = state.get('used_config_path') or state.get('notebook_config_path')
        if not config_path or not Path(config_path).exists():
            rows.append({'dataset': dataset_name, 'split': 'missing_config', 'normal': 0, 'anomaly': 0, 'total': 0})
            continue
        config = load_config(config_path)
        splits_dir = Path(config['data']['splits_dir'])
        if not splits_dir.is_absolute():
            splits_dir = PROJECT_ROOT / splits_dir
        for split in ['train', 'val', 'val_synthetic', 'test', 'test_synthetic']:
            path = splits_dir / f'{split}.json'
            if not path.exists():
                rows.append({'dataset': dataset_name, 'split': split, 'normal': 0, 'anomaly': 0, 'total': 0, 'status': 'missing'})
                continue
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            labels = [int(x) for x in payload.get('labels', [])]
            normal = sum(x == 0 for x in labels)
            anomaly = sum(x == 1 for x in labels)
            rows.append({
                'dataset': dataset_name,
                'split': split,
                'normal': normal,
                'anomaly': anomaly,
                'total': len(labels) if labels else len(payload.get('sequences', [])),
                'status': 'ok',
            })
    return pd.DataFrame(rows)


def real_vs_synthetic_label_summary(dataset_states):
    split_df = split_label_summary(dataset_states)
    if split_df.empty:
        return split_df
    rows = []
    for dataset_name, group in split_df.groupby('dataset', sort=False):
        real = group[group['split'].isin(['train', 'val', 'test'])]
        synthetic = group[group['split'].isin(['val_synthetic', 'test_synthetic'])]
        rows.append({
            'dataset': dataset_name,
            'source': 'real',
            'normal': int(real['normal'].sum()),
            'anomaly': int(real['anomaly'].sum()),
            'total': int(real['total'].sum()),
        })
        rows.append({
            'dataset': dataset_name,
            'source': 'synthetic',
            'normal': int(synthetic['normal'].sum()),
            'anomaly': int(synthetic['anomaly'].sum()),
            'total': int(synthetic['total'].sum()),
        })
        rows.append({
            'dataset': dataset_name,
            'source': 'real+synthetic',
            'normal': int(group['normal'].sum()),
            'anomaly': int(group['anomaly'].sum()),
            'total': int(group['total'].sum()),
        })
    rows.append({
        'dataset': 'ALL',
        'source': 'real',
        'normal': int(split_df[split_df['split'].isin(['train', 'val', 'test'])]['normal'].sum()),
        'anomaly': int(split_df[split_df['split'].isin(['train', 'val', 'test'])]['anomaly'].sum()),
        'total': int(split_df[split_df['split'].isin(['train', 'val', 'test'])]['total'].sum()),
    })
    rows.append({
        'dataset': 'ALL',
        'source': 'synthetic',
        'normal': int(split_df[split_df['split'].isin(['val_synthetic', 'test_synthetic'])]['normal'].sum()),
        'anomaly': int(split_df[split_df['split'].isin(['val_synthetic', 'test_synthetic'])]['anomaly'].sum()),
        'total': int(split_df[split_df['split'].isin(['val_synthetic', 'test_synthetic'])]['total'].sum()),
    })
    rows.append({
        'dataset': 'ALL',
        'source': 'real+synthetic',
        'normal': int(split_df['normal'].sum()),
        'anomaly': int(split_df['anomaly'].sum()),
        'total': int(split_df['total'].sum()),
    })
    return pd.DataFrame(rows)


def dataset_links_table():
    return pd.DataFrame.from_dict(DATASET_SOURCES, orient='index').reset_index(names='dataset')
