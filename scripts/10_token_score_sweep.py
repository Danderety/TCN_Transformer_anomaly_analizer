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
        localization_masks=payload.get('localization_masks'),
    )
    return DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers', 0))


def aggregate(token_errors, mode):
    values = np.asarray(token_errors, dtype=float)
    if values.ndim != 2:
        raise ValueError('token_errors must be a 2D array')
    mask = values > 0
    clean = np.where(mask, values, np.nan)
    mode = str(mode).lower()
    if mode == 'mean':
        return np.nanmean(clean, axis=1)
    if mode == 'max':
        return np.nanmax(clean, axis=1)
    if mode.startswith('top') and mode.endswith('_mean'):
        k = int(mode[3:-5])
        filled = np.where(np.isfinite(clean), clean, -np.inf)
        top = np.sort(filled, axis=1)[:, -k:]
        top = np.where(np.isfinite(top), top, np.nan)
        return np.nanmean(top, axis=1)
    raise ValueError(f'Unknown aggregation mode: {mode}')


def score_token_errors(config, model_name, device):
    output_dir = Path(config.get('project', {}).get('output_dir', 'outputs'))
    checkpoint = output_dir / 'models' / f'{model_name}_best.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = build_model(model_name, config).to(device)
    model, _ = load_checkpoint(model, checkpoint, device)
    train = load_json(f"{config['data']['splits_dir']}/train.json")
    test = load_json(f"{config['data']['splits_dir']}/test.json")
    try:
        train_r = score_dataset(model, loader(train, config), device, config['data']['pad_id'], include_details=True)
        test_r = score_dataset(model, loader(test, config), device, config['data']['pad_id'], include_details=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return train_r, test_r


def main(config_path, model_name, modes, percentiles):
    config = load_config(config_path)
    output_dir = Path(config.get('project', {}).get('output_dir', 'outputs'))
    ensure_dir(output_dir / 'metrics')
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Scoring token errors for {config["data"]["dataset_name"]}/{model_name} on {device} ...', flush=True)
    train_r, test_r = score_token_errors(config, model_name, device)
    rows = []
    for mode in modes:
        train_scores = aggregate(train_r['token_errors'], mode)
        test_scores = aggregate(test_r['token_errors'], mode)
        finite_train = train_scores[np.isfinite(train_scores)]
        for percentile in percentiles:
            threshold = float(np.percentile(finite_train, float(percentile)))
            pred = apply_threshold(test_scores, threshold)
            metrics = compute_detection_metrics(test_r['labels'], test_scores, pred)
            metrics.update({
                'dataset': config['data']['dataset_name'],
                'model': model_name,
                'score_aggregation': mode,
                'percentile': float(percentile),
                'threshold': threshold,
            })
            rows.append(metrics)
    df = pd.DataFrame(rows)
    path = output_dir / 'metrics' / 'token_score_sweep_metrics.csv'
    df.to_csv(path, index=False)
    view = df.sort_values(['f1', 'precision', 'recall'], ascending=False).head(20)
    print(view[['dataset', 'model', 'score_aggregation', 'percentile', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp', 'false_positive_rate']].to_string(index=False))
    print(f'Saved: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', default='tcn_ae')
    parser.add_argument('--modes', default='mean,max,top3_mean,top5_mean,top10_mean')
    parser.add_argument('--percentiles', default='99,98,97,95,90,85,80,75')
    args = parser.parse_args()
    main(
        args.config,
        args.model,
        [x.strip() for x in args.modes.split(',') if x.strip()],
        [float(x.strip()) for x in args.percentiles.split(',') if x.strip()],
    )
