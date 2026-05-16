from collections import deque
import numpy as np
from src.evaluation.detection_metrics import compute_detection_metrics

def compute_fixed_threshold(scores, percentile=95):
    if len(scores)==0: raise ValueError('empty scores')
    return float(np.percentile(scores, _percentile_value(percentile)))

def apply_threshold(scores, threshold):
    return (np.array(scores) > threshold).astype(int).tolist()

def _percentile_value(value):
    value = float(value)
    return value * 100.0 if 0.0 < value <= 1.0 else value

def _finite_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)]

def next_below(x):
    return float(np.nextafter(float(x), -np.inf))

def train_quantile_candidate(train_scores, percentile=95):
    train_scores = _finite_scores(train_scores)
    if len(train_scores) == 0:
        return float('inf'), 'train_quantile_empty_scores'
    percentile = _percentile_value(percentile)
    return float(np.percentile(train_scores, percentile)), f'train_q{percentile:g}'

def spot_candidate(train_scores, init_quantile=98, risk=1e-4, fallback_percentile=95, min_excess=20):
    """SPOT/POT threshold candidate fitted on normal train scores.

    If scipy is unavailable or the tail is too small, the function falls back
    to a conservative train quantile instead of failing the whole experiment.
    """
    train_scores = _finite_scores(train_scores)
    if len(train_scores) == 0:
        return float('inf'), 'spot_empty_scores'

    fallback_percentile = max(_percentile_value(fallback_percentile), _percentile_value(init_quantile))

    def _fallback(note):
        threshold, _ = train_quantile_candidate(train_scores, min(99.9, fallback_percentile))
        return threshold, note

    base = float(np.percentile(train_scores, _percentile_value(init_quantile)))
    excess = train_scores[train_scores > base] - base
    if len(excess) < int(min_excess) or np.allclose(excess, 0):
        return _fallback('spot_fallback_quantile')
    try:
        from scipy.stats import genpareto
    except Exception:
        return _fallback('spot_no_scipy_fallback_quantile')
    try:
        shape, _loc, scale = genpareto.fit(excess, floc=0)
        n, nt = len(train_scores), len(excess)
        risk = max(float(risk), 1e-12)
        if abs(shape) < 1e-8:
            threshold = base + scale * np.log(nt / max(n * risk, 1e-12))
        else:
            threshold = base + (scale / shape) * ((nt / max(n * risk, 1e-12)) ** shape - 1.0)
        if not np.isfinite(threshold):
            raise ValueError('non-finite SPOT threshold')
        return float(threshold), 'spot_gpd'
    except Exception:
        return _fallback('spot_fit_failed_fallback')

def recall_lock_threshold(y_true, scores, fallback_percentile=95):
    """Highest deployable threshold that keeps validation false negatives at 0."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return float('inf'), 'recall_lock_empty_scores'
    finite_scores = scores.copy()
    finite_scores[~np.isfinite(finite_scores)] = np.inf
    positive_scores = finite_scores[y_true == 1]
    if len(positive_scores) == 0:
        threshold, _ = train_quantile_candidate(finite_scores, fallback_percentile)
        return threshold, 'recall_lock_no_positive_labels'
    return next_below(np.min(positive_scores)), 'recall_lock_min_anomaly'

def threshold_for_exact_recall(y_true, scores):
    threshold, note = recall_lock_threshold(y_true, scores)
    pred = apply_threshold(scores, threshold)
    metrics = compute_detection_metrics(y_true, scores, pred)
    return threshold, metrics, note

def select_validation_safe_threshold(train_scores, val_scores, val_labels, threshold_config=None, raw_method='quantile'):
    """Select raw threshold, then optionally apply validation safety-check.

    The safety-check follows the old notebook policy: final=min(raw, recall-lock).
    Lowering the threshold may increase FP, but it prevents validation FN when
    validation contains positive labels.
    """
    threshold_config = threshold_config or {}
    raw_method = str(raw_method or threshold_config.get('method', 'quantile')).lower()
    fixed_percentile = threshold_config.get('fixed_percentile', 95)
    if raw_method == 'spot':
        raw_threshold, raw_note = spot_candidate(
            train_scores,
            init_quantile=threshold_config.get('spot_init_quantile', 98),
            risk=threshold_config.get('spot_risk', 1e-4),
            fallback_percentile=fixed_percentile,
            min_excess=threshold_config.get('spot_min_excess', 20),
        )
    else:
        raw_threshold, raw_note = train_quantile_candidate(train_scores, fixed_percentile)

    raw_pred = apply_threshold(val_scores, raw_threshold)
    raw_metrics = compute_detection_metrics(val_labels, val_scores, raw_pred)
    safe_threshold, safe_note = recall_lock_threshold(val_labels, val_scores, fixed_percentile)
    safe_pred = apply_threshold(val_scores, safe_threshold)
    safe_metrics = compute_detection_metrics(val_labels, val_scores, safe_pred)

    safety_enabled = bool(threshold_config.get('validation_safety_check', True))
    final_threshold = min(float(raw_threshold), float(safe_threshold)) if safety_enabled else float(raw_threshold)
    final_pred = apply_threshold(val_scores, final_threshold)
    final_metrics = compute_detection_metrics(val_labels, val_scores, final_pred)
    return {
        'threshold': float(final_threshold),
        'threshold_note': f'final=min(raw,{safe_note})' if safety_enabled else 'final=raw',
        'threshold_type': f'{raw_method}_validation_safe' if safety_enabled else raw_method,
        'raw_threshold': float(raw_threshold),
        'raw_note': raw_note,
        'raw_metrics': raw_metrics,
        'safe_threshold': float(safe_threshold),
        'safe_note': safe_note,
        'safe_metrics': safe_metrics,
        'metrics': final_metrics,
        'was_safety_applied': bool(safety_enabled and final_threshold < raw_threshold - 1e-12),
    }

class AdaptiveThreshold:
    def __init__(self, window_size=200, k=3.0, min_window=30, update_only_normal=True):
        self.scores=deque(maxlen=window_size); self.k=k; self.min_window=min_window; self.update_only_normal=update_only_normal
    def warmup(self, scores):
        for s in scores: self.scores.append(float(s))
    def get_threshold(self):
        if len(self.scores) < self.min_window: return None
        a=np.array(self.scores); return float(a.mean()+self.k*a.std())
    def predict_one(self, score):
        thr=self.get_threshold(); pred=0 if thr is None else int(score > thr)
        if (not self.update_only_normal) or thr is None or score <= thr: self.scores.append(float(score))
        return pred, thr
    def predict_many(self, scores):
        p,t=[],[]
        for s in scores:
            pp,tt=self.predict_one(s); p.append(pp); t.append(tt)
        return p,t
