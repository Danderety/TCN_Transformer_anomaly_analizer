from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _save(path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True); plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distribution(normal_scores, anomaly_scores, save_path=None):
    plt.figure(figsize=(8,5)); plt.hist(normal_scores,bins=50,alpha=0.6,label='Normal'); plt.hist(anomaly_scores,bins=50,alpha=0.6,label='Anomaly')
    plt.xlabel('Anomaly score'); plt.ylabel('Count'); plt.title('Anomaly score distribution'); plt.legend(); plt.tight_layout(); _save(save_path)

def plot_threshold_dynamics(scores, thresholds, labels=None, save_path=None):
    plt.figure(figsize=(12,5)); plt.plot(scores,label='Anomaly score'); plt.plot([np.nan if t is None else t for t in thresholds],label='Adaptive threshold')
    if labels is not None:
        idx=[i for i,y in enumerate(labels) if int(y)==1]; plt.scatter(idx,[scores[i] for i in idx],marker='x',label='True anomaly')
    plt.xlabel('Sequence index'); plt.ylabel('Score'); plt.title('Adaptive threshold dynamics'); plt.legend(); plt.tight_layout(); _save(save_path)

def plot_heatmap(event_ids, token_scores, true_mask=None, pad_id=0, title='Anomaly heatmap', save_path=None):
    e=np.array(event_ids); s=np.array(token_scores,dtype=float); valid=e!=pad_id; e=e[valid]; s=s[valid]
    if len(e)==0: return
    if true_mask is not None: true_mask=np.array(true_mask)[valid]
    plt.figure(figsize=(14,2.8)); plt.imshow(s.reshape(1,-1),aspect='auto'); plt.colorbar(label='Contribution'); plt.yticks([]); plt.xticks(range(len(e)), e, rotation=90)
    if true_mask is not None:
        for p in np.where(true_mask==1)[0]: plt.axvline(p, linestyle='--', linewidth=2)
    plt.title(title); plt.xlabel('Event position'); plt.tight_layout(); _save(save_path)

def plot_model_comparison(df, metric='f1', save_path=None):
    if df.empty or metric not in df:
        return
    labels = df.apply(lambda r: f"{r.get('model','model')}\n{r.get('threshold_type','')}", axis=1)
    plt.figure(figsize=(10,5))
    plt.bar(labels, df[metric].astype(float))
    plt.ylabel(metric)
    plt.title(f'Model comparison by {metric}')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    _save(save_path)
