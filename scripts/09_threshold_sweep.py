import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LogSequenceDataset
from src.evaluation.detection_metrics import compute_detection_metrics
from src.evaluation.multiscale import score_payload_multiscale
from src.evaluation.scoring import score_dataset
from src.evaluation.thresholds import apply_threshold
from src.models.factory import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json


def loader(payload, config):
    ds = LogSequenceDataset(
        payload['sequences'],
        payload.get('labels'),
        max_len=config['data']['max_seq_len'],
        pad_id=config['data']['pad_id'],
        session_ids=payload.get('session_ids'),
    )
    return DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
    )


def score_payload(model, payload, train_payload, config, device):
    if config.get('multi_scale', {}).get('enabled', False):
        return score_payload_multiscale(model, payload, train_payload, config, device)
    return score_dataset(model, loader(payload, config), device, config['data']['pad_id'], include_details=False)


def score_model(config, model_name, device):
    output_dir = Path(config.get('project', {}).get('output_dir', 'outputs'))
    checkpoint = output_dir / 'models' / f'{model_name}_best.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing checkpoint for {model_name}: {checkpoint}')
    train = load_json(f"{config['data']['splits_dir']}/train.json")
    test = load_json(f"{config['data']['splits_dir']}/test.json")
    model = build_model(model_name, config).to(device)
    model, _ = load_checkpoint(model, checkpoint, device)
    try:
        train_r = score_payload(model, train, train, config, device)
        test_r = score_payload(model, test, train, config, device)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.asarray(train_r['scores'], dtype=float), np.asarray(test_r['scores'], dtype=float), test_r['labels']


def sweep(config, model_name, percentiles, device):
    train_scores, test_scores, test_labels = score_model(config, model_name, device)
    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(train_scores[np.isfinite(train_scores)], float(percentile)))
        pred = apply_threshold(test_scores, threshold)
        metrics = compute_detection_metrics(test_labels, test_scores, pred)
        metrics.update({
            'dataset': config['data']['dataset_name'],
            'model': model_name,
            'threshold_type': 'train_quantile_sweep',
            'percentile': float(percentile),
            'threshold': threshold,
            'train_score_min': float(np.nanmin(train_scores)),
            'train_score_median': float(np.nanmedian(train_scores)),
            'train_score_max': float(np.nanmax(train_scores)),
            'test_score_min': float(np.nanmin(test_scores)),
            'test_score_median': float(np.nanmedian(test_scores)),
            'test_score_max': float(np.nanmax(test_scores)),
        })
        rows.append(metrics)
    return rows


def main(config_path, models, percentiles):
    config = load_config(config_path)
    output_dir = Path(config.get('project', {}).get('output_dir', 'outputs'))
    ensure_dir(output_dir / 'metrics')
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    rows = []
    for model_name in models:
        print(f'Sweeping {config["data"]["dataset_name"]}/{model_name} on {device} ...', flush=True)
        rows.extend(sweep(config, model_name, percentiles, device))
    df = pd.DataFrame(rows)
    path = output_dir / 'metrics' / 'threshold_sweep_metrics.csv'
    df.to_csv(path, index=False)
    print(df[['dataset', 'model', 'percentile', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp', 'false_positive_rate']].to_string(index=False))
    print(f'Saved: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--models', default='tcn_ae')
    parser.add_argument('--percentiles', default='99,98,97,95,90,85,80,75')
    args = parser.parse_args()
    main(
        args.config,
        [x.strip() for x in args.models.split(',') if x.strip()],
        [float(x.strip()) for x in args.percentiles.split(',') if x.strip()],
    )
