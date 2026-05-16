import numpy as np


def robust_score_normalize(scores, train_scores):
    scores = np.asarray(scores, dtype=np.float32)
    train_scores = np.asarray(train_scores, dtype=np.float32)
    train_scores = train_scores[np.isfinite(train_scores)]
    if len(train_scores) == 0:
        return scores, {'median': 0.0, 'scale': 1.0, 'note': 'empty_train_scores'}
    median = float(np.median(train_scores))
    q75, q25 = np.percentile(train_scores, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr if iqr > 1e-12 else float(np.std(train_scores) + 1e-6)
    return ((scores - median) / max(scale, 1e-6)).astype(np.float32), {
        'median': median,
        'scale': float(scale),
        'note': 'robust_z_train',
    }


def combine_ensemble_scores(member_scores, member_train_scores, norm='robust_z_train'):
    if len(member_scores) == 0:
        raise ValueError('member_scores is empty')
    normalized = []
    norm_stats = []
    for scores, train_scores in zip(member_scores, member_train_scores):
        if norm == 'robust_z_train':
            values, stats = robust_score_normalize(scores, train_scores)
        elif norm == 'none':
            values = np.asarray(scores, dtype=np.float32)
            stats = {'median': 0.0, 'scale': 1.0, 'note': 'none'}
        else:
            raise ValueError(f'Unknown ensemble score norm: {norm}')
        normalized.append(values)
        norm_stats.append(stats)
    return np.mean(np.vstack(normalized), axis=0).astype(np.float32), norm_stats
