from importlib.util import module_from_spec, spec_from_file_location
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
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
        'target_dir': 'data/raw/lo2 там 2 папки logs и metrics',
        'url': 'https://doi.org/10.5281/zenodo.14265858',
        'note': 'LO2 / Light-OAuth2 microservice API anomaly dataset. Put extracted log files under data/raw/lo2/.',
    },
    'loghub2': {
        'target_dir': 'data/raw/loghub2 без папки тупо данные',
        'url': 'https://zenodo.org/records/8275861',
        'note': 'Loghub-2.0 collection. Put extracted log files under data/raw/loghub2/.',
    },
    'rcaeval': {
        'target_dir': 'data/raw/rcaeval папка сюда RE1-OB тупо копировать',
        'url': 'https://zenodo.org/records/14590730',
        'note': 'RCAEval benchmark repository/data. Put extracted logs under data/raw/rcaeval/.',
    },
}

LOGHUB_LABEL_CANDIDATES = [
    PROJECT_ROOT / 'data' / 'raw' / 'loghub2' / 'anomaly_label.csv',
    PROJECT_ROOT / 'data' / 'raw' / 'loghub2' / 'preprocessed' / 'anomaly_label.csv',
    PROJECT_ROOT.parent.parent / 'ae_vs_tcn' / 'data' / 'HDFS_v1' / 'preprocessed' / 'anomaly_label.csv',
    Path.home() / 'Documents' / 'ae_vs_tcn' / 'data' / 'HDFS_v1' / 'preprocessed' / 'anomaly_label.csv',
]


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


def cuda_status():
    try:
        import torch
        info = {
            'torch': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_version': torch.version.cuda,
            'device_count': int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
        return info
    except Exception as exc:
        return {'cuda_available': False, 'error': f'{type(exc).__name__}: {exc}'}


def ensure_loghub_gold_labels(raw_dir=None, source_path=None):
    """Ensure Loghub/HDFS has anomaly_label.csv for BlockId-level gold labels."""
    raw_dir = Path(raw_dir or PROJECT_ROOT / 'data' / 'raw' / 'loghub2')
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    target = raw_dir / 'anomaly_label.csv'
    if target.exists() and target.stat().st_size > 0:
        return target

    candidates = []
    if source_path is not None:
        candidates.append(Path(source_path))
    candidates.extend(LOGHUB_LABEL_CANDIDATES)
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            return target

    searched = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(
        'Loghub/HDFS gold labels are required for supervised Loghub training. '
        f'Put anomaly_label.csv into {raw_dir} or one of:\n{searched}'
    )


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
    num_workers=0,
    max_localization_samples=512,
    epochs=100,
    patience=8,
):
    config_path = make_notebook_config(config_path, dataset_name=dataset_name, output_dir=output_dir)
    config = load_config(config_path)
    dataset_name = dataset_name or config['data']['dataset_name']
    data_dataset_name = str(config.get('data', {}).get('dataset_name', dataset_name)).lower()
    config.setdefault('training', {})['batch_size'] = int(batch_size)
    config['training']['epochs'] = int(epochs)
    config['training']['patience'] = int(patience)
    config['training'].setdefault('loss_min_delta', 0.0)
    config['training'].setdefault('threshold_patience', 9)
    config['training'].setdefault('threshold_min_delta', 0.000001)
    config['training'].setdefault('threshold_min_relative_delta', 0.01)
    config['training'].setdefault('threshold_save_loss_tolerance', 0.05)
    config['training']['num_workers'] = int(num_workers)
    config.setdefault('data', {})['window_stride'] = int(config['data'].get('window_stride', 64))
    config['data']['no_split'] = True
    config.setdefault('threshold', {})
    config['threshold'].setdefault('selection_policy', 'recall_fpr_tradeoff')
    if data_dataset_name in {'loghub2', 'rcaeval'}:
        config['threshold']['selection_policy'] = 'raw'
        config['threshold']['validation_safety_check'] = False
    config['threshold'].setdefault('target_recall', 0.99)
    config['threshold'].setdefault('max_validation_fpr', 0.25)
    config['threshold'].setdefault('fbeta_beta', 3.0)
    config['threshold'].setdefault('enforce_validation_recall', False)
    config['threshold']['adaptive_window_size'] = config['threshold'].get('adaptive_window_size', 'auto')
    config['threshold'].setdefault('adaptive_window_fraction', 0.10)
    config['threshold'].setdefault('adaptive_window_min', 30)
    config['threshold'].setdefault('adaptive_window_max', 200)
    config['threshold'].setdefault('adaptive_method', 'robust_mad')
    config['threshold'].setdefault('adaptive_min_scale', 1e-6)
    config['threshold'].setdefault('adaptive_warmup_trim_percentile', 95)
    config['threshold'].setdefault('adaptive_calibrate_k', data_dataset_name != 'loghub2')
    config['threshold'].setdefault('adaptive_k_grid', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    if data_dataset_name == 'loghub2':
        config['threshold']['adaptive_k'] = 4.0
        config['threshold']['adaptive_calibrate_k'] = False
    else:
        config['threshold']['adaptive_k'] = float(config['threshold'].get('adaptive_k', 2.0))
    config['threshold']['adaptive_min_window'] = int(config['threshold'].get('adaptive_min_window', 20))
    config.setdefault('multi_scale', {})
    config['multi_scale'].setdefault('enabled', True)
    config['multi_scale'].setdefault('lengths', [64, 128, 256])
    config['multi_scale'].setdefault('aggregation', 'max')
    config['multi_scale'].setdefault('normalize', 'robust_z_train')
    config.setdefault('evaluation', {})['max_localization_samples'] = int(max_localization_samples)
    config['evaluation'].setdefault('use_synthetic_validation_when_no_anomalies', True)
    config['evaluation']['full_real_evaluation'] = True
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


def config_audit_table(dataset_states):
    rows = []
    for dataset_name, state in dataset_states.items():
        config_path = state.get('used_config_path') if Path(state.get('used_config_path', '')).exists() else state.get('notebook_config_path')
        if not config_path or not Path(config_path).exists():
            rows.append({'dataset': dataset_name, 'status': 'missing_config'})
            continue
        config = load_config(config_path)
        data_dataset_name = str(config.get('data', {}).get('dataset_name', dataset_name)).lower()
        raw_dir = Path(config['data']['raw_dir'])
        if not raw_dir.is_absolute():
            raw_dir = PROJECT_ROOT / raw_dir
        rows.append({
            'dataset': dataset_name,
            'status': 'ok',
            'config_path': str(config_path),
            'raw_dir': str(raw_dir),
            'splits_dir': config['data'].get('splits_dir'),
            'output_dir': config.get('project', {}).get('output_dir'),
            'vocab_size': config.get('model', {}).get('vocab_size'),
            'd_model': config.get('model', {}).get('d_model'),
            'tcn_layers': config.get('model', {}).get('tcn_layers'),
            'transformer_layers': config.get('model', {}).get('transformer_layers'),
            'nhead': config.get('model', {}).get('nhead'),
            'dim_feedforward': config.get('model', {}).get('dim_feedforward'),
            'dropout': config.get('model', {}).get('dropout'),
            'max_seq_len': config.get('data', {}).get('max_seq_len'),
            'window_stride': config.get('data', {}).get('window_stride'),
            'batch_size': config.get('training', {}).get('batch_size'),
            'epochs': config.get('training', {}).get('epochs'),
            'learning_rate': config.get('training', {}).get('learning_rate'),
            'weight_decay': config.get('training', {}).get('weight_decay'),
            'patience': config.get('training', {}).get('patience'),
            'device': config.get('training', {}).get('device'),
            'num_workers': config.get('training', {}).get('num_workers'),
            'threshold_policy': config.get('threshold', {}).get('selection_policy'),
            'validation_safety_check': config.get('threshold', {}).get('validation_safety_check'),
            'adaptive_method': config.get('threshold', {}).get('adaptive_method'),
            'adaptive_k': config.get('threshold', {}).get('adaptive_k'),
            'ensemble_enabled': config.get('ensemble', {}).get('enabled'),
            'ensemble_seeds': config.get('ensemble', {}).get('member_seeds'),
            'evaluation_models': config.get('evaluation', {}).get('models'),
            'loghub_gold_labels': str(raw_dir / 'anomaly_label.csv') if data_dataset_name == 'loghub2' and (raw_dir / 'anomaly_label.csv').exists() else '',
        })
    return pd.DataFrame(rows)


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
        for split in ['train', 'val', 'val_synthetic', 'test', 'test_synthetic', 'full_real']:
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
                'source': 'synthetic' if split in {'val_synthetic', 'test_synthetic'} else ('real_full' if split == 'full_real' else 'real'),
                'split': split,
                'normal': normal,
                'anomaly': anomaly,
                'total': len(labels) if labels else len(payload.get('sequences', [])),
                'status': 'ok',
            })
    df = pd.DataFrame(rows)
    if not df.empty and 'total' in df:
        denominator = df['total'].replace(0, float('nan'))
        df['normal_pct'] = (df['normal'] / denominator * 100).round(2)
        df['anomaly_pct'] = (df['anomaly'] / denominator * 100).round(2)
    return df


def preparation_level_summary(dataset_states):
    rows = []
    for dataset_name, state in dataset_states.items():
        config_path = state.get('used_config_path') or state.get('notebook_config_path')
        if not config_path or not Path(config_path).exists():
            rows.append({'dataset': dataset_name, 'level': 'missing_config', 'normal': 0, 'anomaly': 0, 'total': 0})
            continue
        config = load_config(config_path)
        processed_dir = Path(config['data']['processed_dir'])
        if not processed_dir.is_absolute():
            processed_dir = PROJECT_ROOT / processed_dir
        summary_path = processed_dir / 'prepare_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            for item in payload.get('levels', []):
                rows.append({
                    'dataset': dataset_name,
                    'level': item.get('level'),
                    'normal': item.get('normal'),
                    'anomaly': item.get('anomaly'),
                    'total': item.get('total'),
                    'description': item.get('description', ''),
                    'status': 'ok',
                })
            continue
        parsed_path = processed_dir / 'parsed_logs.csv'
        if parsed_path.exists():
            parsed = pd.read_csv(parsed_path, usecols=['session_id', 'label'])
            rows.append({
                'dataset': dataset_name,
                'level': 'raw_rows',
                'normal': int((parsed['label'] == 0).sum()),
                'anomaly': int((parsed['label'] == 1).sum()),
                'total': int(len(parsed)),
                'description': 'original log/event rows before session aggregation',
                'status': 'fallback_from_parsed_logs',
            })
            rows.append({
                'dataset': dataset_name,
                'level': 'raw_sessions',
                'normal': None,
                'anomaly': None,
                'total': int(parsed['session_id'].nunique()),
                'description': 'unique session_id values in parsed_logs.csv',
                'status': 'fallback_from_parsed_logs',
            })
        else:
            rows.append({
                'dataset': dataset_name,
                'level': 'missing_prepare_summary',
                'normal': 0,
                'anomaly': 0,
                'total': 0,
                'description': f'Run prepare_data to create {summary_path}',
                'status': 'missing',
            })
    return pd.DataFrame(rows)


def detailed_label_summary(dataset_states):
    split_df = split_label_summary(dataset_states)
    if split_df.empty:
        return split_df
    rows = []
    for dataset_name, group in split_df.groupby('dataset', sort=False):
        ok = group[group.get('status', 'ok').eq('ok')] if 'status' in group else group
        rows.extend(ok.to_dict('records'))
        for source, source_group in ok.groupby('source', sort=False):
            rows.append({
                'dataset': dataset_name,
                'source': source,
                'split': f'{source}_total',
                'normal': int(source_group['normal'].sum()),
                'anomaly': int(source_group['anomaly'].sum()),
                'total': int(source_group['total'].sum()),
                'status': 'total',
            })
        rows.append({
            'dataset': dataset_name,
            'source': 'real+synthetic',
            'split': 'dataset_total',
            'normal': int(ok['normal'].sum()),
            'anomaly': int(ok['anomaly'].sum()),
            'total': int(ok['total'].sum()),
            'status': 'total',
        })
    all_ok = split_df[split_df.get('status', 'ok').eq('ok')] if 'status' in split_df else split_df
    rows.append({
        'dataset': 'ALL',
        'source': 'real+synthetic',
        'split': 'all_total',
        'normal': int(all_ok['normal'].sum()),
        'anomaly': int(all_ok['anomaly'].sum()),
        'total': int(all_ok['total'].sum()),
        'status': 'total',
    })
    df = pd.DataFrame(rows)
    denominator = df['total'].replace(0, float('nan'))
    df['normal_pct'] = (df['normal'] / denominator * 100).round(2)
    df['anomaly_pct'] = (df['anomaly'] / denominator * 100).round(2)
    return df[['dataset', 'source', 'split', 'normal', 'anomaly', 'total', 'normal_pct', 'anomaly_pct', 'status']]


def expected_detection_counts(dataset_states):
    split_df = split_label_summary(dataset_states)
    if split_df.empty:
        return split_df
    rows = []
    for dataset_name, group in split_df.groupby('dataset', sort=False):
        ok = group[group.get('status', 'ok').eq('ok') if 'status' in group else group.index == group.index]

        def counts(split):
            part = ok[ok['split'] == split]
            if part.empty:
                return 0, 0, 0
            row = part.iloc[0]
            return int(row['normal']), int(row['anomaly']), int(row['total'])

        train_n, train_a, train_t = counts('train')
        val_n, val_a, val_t = counts('val')
        test_n, test_a, test_t = counts('test')
        vsy_n, vsy_a, vsy_t = counts('val_synthetic')
        sy_n, sy_a, sy_t = counts('test_synthetic')
        full_n, full_a, full_t = counts('full_real')
        real_total = full_t if full_t else train_t + val_t + test_t
        real_detection_n = full_n if full_t else val_n + test_n
        real_detection_a = full_a if full_t else val_a + test_a
        rows.append({
            'dataset': dataset_name,
            'train_normal_for_ae': train_n,
            'train_anomaly_excluded': train_a,
            'val_should_detect_anomaly': val_a,
            'val_normal': val_n,
            'test_should_detect_anomaly': test_a,
            'test_normal': test_n,
            'real_should_detect_anomaly_val_test': real_detection_a,
            'real_normal_val_test': real_detection_n,
            'synthetic_should_detect_anomaly': vsy_a + sy_a,
            'synthetic_normal': vsy_n + sy_n,
            'real_total': real_total,
        })
    return pd.DataFrame(rows)


def real_vs_synthetic_label_summary(dataset_states):
    split_df = split_label_summary(dataset_states)
    if split_df.empty:
        return split_df
    rows = []
    for dataset_name, group in split_df.groupby('dataset', sort=False):
        full_real = group[(group['split'] == 'full_real') & (group.get('status', 'ok').eq('ok') if 'status' in group else True)]
        real = full_real if not full_real.empty else group[group['split'].isin(['train', 'val', 'test'])]
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
            'normal': int(real['normal'].sum() + synthetic['normal'].sum()),
            'anomaly': int(real['anomaly'].sum() + synthetic['anomaly'].sum()),
            'total': int(real['total'].sum() + synthetic['total'].sum()),
        })
    full_real_all = split_df[(split_df['split'] == 'full_real') & (split_df.get('status', 'ok').eq('ok') if 'status' in split_df else True)]
    real_all = full_real_all if not full_real_all.empty else split_df[split_df['split'].isin(['train', 'val', 'test'])]
    synthetic_all = split_df[split_df['split'].isin(['val_synthetic', 'test_synthetic'])]
    rows.append({
        'dataset': 'ALL',
        'source': 'real',
        'normal': int(real_all['normal'].sum()),
        'anomaly': int(real_all['anomaly'].sum()),
        'total': int(real_all['total'].sum()),
    })
    rows.append({
        'dataset': 'ALL',
        'source': 'synthetic',
        'normal': int(synthetic_all['normal'].sum()),
        'anomaly': int(synthetic_all['anomaly'].sum()),
        'total': int(synthetic_all['total'].sum()),
    })
    rows.append({
        'dataset': 'ALL',
        'source': 'real+synthetic',
        'normal': int(real_all['normal'].sum() + synthetic_all['normal'].sum()),
        'anomaly': int(real_all['anomaly'].sum() + synthetic_all['anomaly'].sum()),
        'total': int(real_all['total'].sum() + synthetic_all['total'].sum()),
    })
    return pd.DataFrame(rows)


def dataset_links_table():
    return pd.DataFrame.from_dict(DATASET_SOURCES, orient='index').reset_index(names='dataset')
