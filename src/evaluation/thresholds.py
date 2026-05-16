from collections import deque
import numpy as np
from src.evaluation.detection_metrics import compute_detection_metrics

def compute_fixed_threshold(scores, percentile=95):
    if len(scores)==0: raise ValueError('empty scores')
    return float(np.percentile(scores, _percentile_value(percentile)))

def apply_threshold(scores, threshold):
    scores = _scores_for_prediction(scores)
    return (scores >= float(threshold)).astype(int).tolist()

def _percentile_value(value):
    value = float(value)
    return value * 100.0 if 0.0 < value <= 1.0 else value

def _finite_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)]

def _scores_for_prediction(scores):
    scores = np.asarray(scores, dtype=float)
    # Recall-first fail-closed behavior: a broken/non-finite anomaly score is
    # treated as suspicious, not as normal.
    return np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=np.inf)

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
    safe_scores = _scores_for_prediction(scores)
    positive_scores = safe_scores[y_true == 1]
    if len(positive_scores) == 0:
        threshold, _ = train_quantile_candidate(safe_scores, fallback_percentile)
        return threshold, 'recall_lock_no_positive_labels'
    return next_below(np.min(positive_scores)), 'recall_lock_min_anomaly'

def threshold_for_min_recall(y_true, scores, min_recall=1.0, fallback_percentile=95):
    y_true = np.asarray(y_true, dtype=int)
    scores = _scores_for_prediction(scores)
    positive_scores = np.sort(scores[y_true == 1])
    if len(positive_scores) == 0:
        return recall_lock_threshold(y_true, scores, fallback_percentile)
    min_recall = min(max(float(min_recall), 0.0), 1.0)
    max_fn = int(np.floor((1.0 - min_recall) * len(positive_scores)))
    max_fn = min(max(max_fn, 0), len(positive_scores) - 1)
    return next_below(positive_scores[max_fn]), f'min_recall_{min_recall:g}_fn_budget_{max_fn}'

def threshold_for_exact_recall(y_true, scores):
    threshold, note = recall_lock_threshold(y_true, scores)
    pred = apply_threshold(scores, threshold)
    metrics = compute_detection_metrics(y_true, scores, pred)
    return threshold, metrics, note

def _fbeta_score(precision, recall, beta=3.0):
    precision = float(precision or 0.0)
    recall = float(recall or 0.0)
    beta2 = float(beta) ** 2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom

def recall_fpr_tradeoff_candidate(y_true, scores, target_recall=0.99, max_fpr=0.25, beta=3.0):
    """Pick a validation operating point that protects recall without burning all normals."""
    y_true = np.asarray(y_true, dtype=int)
    scores = _scores_for_prediction(scores)
    if len(scores) == 0:
        return float('inf'), 'tradeoff_empty_scores', {}
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return float('inf'), 'tradeoff_nonfinite_scores', {}

    unique = np.unique(finite)
    candidates = [next_below(float(unique.min()))]
    candidates += [float(x) for x in unique]
    candidates.append(float(np.nextafter(float(unique.max()), np.inf)))

    rows = []
    for threshold in candidates:
        pred = apply_threshold(scores, threshold)
        metrics = compute_detection_metrics(y_true, scores, pred)
        metrics['_threshold'] = float(threshold)
        metrics['_fbeta'] = _fbeta_score(metrics['precision'], metrics['recall'], beta)
        rows.append(metrics)

    target_recall = float(target_recall)
    max_fpr = float(max_fpr)
    tiers = [
        ('recall_fpr', [m for m in rows if m['recall'] + 1e-12 >= target_recall and m['false_positive_rate'] <= max_fpr + 1e-12]),
        ('recall_only', [m for m in rows if m['recall'] + 1e-12 >= target_recall]),
        ('fpr_only', [m for m in rows if m['false_positive_rate'] <= max_fpr + 1e-12]),
        ('best_effort', rows),
    ]
    for tier_name, tier_rows in tiers:
        if tier_rows:
            best = max(
                tier_rows,
                key=lambda m: (m['_fbeta'], m['recall'], -m['false_positive_rate'], m['_threshold']),
            )
            threshold = best.pop('_threshold')
            fbeta = best.pop('_fbeta')
            best['fbeta'] = fbeta
            return float(threshold), f'tradeoff_{tier_name}_recall{target_recall:g}_maxfpr{max_fpr:g}_beta{float(beta):g}', best
    return float('inf'), 'tradeoff_no_candidates', {}

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
    target_recall = float(threshold_config.get('target_recall', 1.0))
    safe_threshold, safe_note = threshold_for_min_recall(val_labels, val_scores, target_recall, fixed_percentile)
    safe_pred = apply_threshold(val_scores, safe_threshold)
    safe_metrics = compute_detection_metrics(val_labels, val_scores, safe_pred)

    safety_enabled = bool(threshold_config.get('validation_safety_check', True))
    selection_policy = str(threshold_config.get('selection_policy', 'recall_lock')).lower()
    tradeoff_threshold = None
    tradeoff_note = None
    tradeoff_metrics = None
    if selection_policy in {'recall_fpr_tradeoff', 'balanced_recall'}:
        tradeoff_threshold, tradeoff_note, tradeoff_metrics = recall_fpr_tradeoff_candidate(
            val_labels,
            val_scores,
            target_recall=threshold_config.get('target_recall', 0.99),
            max_fpr=threshold_config.get('max_validation_fpr', 0.25),
            beta=threshold_config.get('fbeta_beta', 3.0),
        )
        final_threshold = float(tradeoff_threshold)
    else:
        final_threshold = min(float(raw_threshold), float(safe_threshold)) if safety_enabled else float(raw_threshold)
    final_pred = apply_threshold(val_scores, final_threshold)
    final_metrics = compute_detection_metrics(val_labels, val_scores, final_pred)
    if threshold_config.get('enforce_validation_recall', True) and selection_policy not in {'recall_fpr_tradeoff', 'balanced_recall'} and np.sum(np.asarray(val_labels, dtype=int) == 1) > 0:
        if final_metrics['recall'] + 1e-12 < target_recall:
            raise AssertionError(
                f"validation target recall failed: recall={final_metrics['recall']:.6f} target={target_recall:.6f}"
            )
    return {
        'threshold': float(final_threshold),
        'threshold_note': tradeoff_note or (f'final=min(raw,{safe_note})' if safety_enabled else 'final=raw'),
        'threshold_type': f'{raw_method}_{selection_policy}' if selection_policy in {'recall_fpr_tradeoff', 'balanced_recall'} else (f'{raw_method}_validation_safe' if safety_enabled else raw_method),
        'raw_threshold': float(raw_threshold),
        'raw_note': raw_note,
        'raw_metrics': raw_metrics,
        'safe_threshold': float(safe_threshold),
        'safe_note': safe_note,
        'safe_metrics': safe_metrics,
        'tradeoff_threshold': None if tradeoff_threshold is None else float(tradeoff_threshold),
        'tradeoff_note': tradeoff_note,
        'tradeoff_metrics': tradeoff_metrics,
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
