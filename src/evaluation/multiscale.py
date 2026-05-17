import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import LogSequenceDataset
from src.evaluation.ensemble import robust_score_normalize
from src.evaluation.scoring import score_dataset


def multi_scale_config(config):
    cfg = config.get('multi_scale', {})
    return {
        'enabled': bool(cfg.get('enabled', False)),
        'lengths': [int(x) for x in cfg.get('lengths', [config['data']['max_seq_len']])],
        'aggregation': str(cfg.get('aggregation', 'max')).lower(),
        'normalize': str(cfg.get('normalize', 'robust_z_train')).lower(),
    }


def make_loader(payload, config, max_len):
    ds = LogSequenceDataset(
        payload['sequences'],
        payload.get('labels'),
        max_len=max_len,
        pad_id=config['data']['pad_id'],
        session_ids=payload.get('session_ids'),
    )
    return DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
    )


def score_payload(model, payload, config, device, max_len=None):
    max_len = int(max_len or config['data']['max_seq_len'])
    return score_dataset(
        model,
        make_loader(payload, config, max_len),
        device,
        config['data']['pad_id'],
        include_details=False,
    )


def score_payload_multiscale(model, payload, train_payload, config, device):
    cfg = multi_scale_config(config)
    if not cfg['enabled']:
        return score_payload(model, payload, config, device)

    scaled_scores = []
    scale_stats = []
    raw_scores = {}
    labels = None
    session_ids = None
    for length in cfg['lengths']:
        train_r = score_payload(model, train_payload, config, device, max_len=length)
        data_r = score_payload(model, payload, config, device, max_len=length)
        labels = data_r.get('labels', labels)
        session_ids = data_r.get('session_ids', session_ids)
        raw_scores[str(length)] = data_r['scores']
        if cfg['normalize'] == 'robust_z_train':
            values, stats = robust_score_normalize(data_r['scores'], train_r['scores'])
        elif cfg['normalize'] == 'none':
            values = np.asarray(data_r['scores'], dtype=np.float32)
            stats = {'median': 0.0, 'scale': 1.0, 'note': 'none'}
        else:
            raise ValueError(f"Unknown multi_scale.normalize: {cfg['normalize']}")
        stats['length'] = length
        scaled_scores.append(values)
        scale_stats.append(stats)

    matrix = np.vstack(scaled_scores)
    if cfg['aggregation'] == 'max':
        final_scores = matrix.max(axis=0)
    elif cfg['aggregation'] == 'mean':
        final_scores = matrix.mean(axis=0)
    else:
        raise ValueError(f"Unknown multi_scale.aggregation: {cfg['aggregation']}")
    return {
        'scores': final_scores.astype(np.float32).tolist(),
        'labels': labels or payload.get('labels', []),
        'session_ids': session_ids or payload.get('session_ids', []),
        'multi_scale_raw_scores': raw_scores,
        'multi_scale_norm_stats': scale_stats,
        'multi_scale_aggregation': cfg['aggregation'],
        'multi_scale_normalize': cfg['normalize'],
    }
