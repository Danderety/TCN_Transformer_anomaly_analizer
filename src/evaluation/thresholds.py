from collections import deque
import numpy as np

def compute_fixed_threshold(scores, percentile=95):
    if len(scores)==0: raise ValueError('empty scores')
    return float(np.percentile(scores, percentile))

def apply_threshold(scores, threshold):
    return (np.array(scores) > threshold).astype(int).tolist()

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
