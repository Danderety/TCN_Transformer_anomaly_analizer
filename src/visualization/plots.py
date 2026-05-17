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

def _robust_ylim(values, labels=None, lower_q=0.5, upper_q=99.0):
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo=float(np.percentile(v, lower_q))
    hi=float(np.percentile(v, upper_q))
    if labels is not None:
        idx=np.asarray(labels,dtype=int)==1
        all_values=np.asarray(values,dtype=float)
        anomaly_values=all_values[idx[:len(all_values)]] if len(idx) else np.array([])
        anomaly_values=anomaly_values[np.isfinite(anomaly_values)]
        if anomaly_values.size:
            hi=max(hi, float(np.percentile(anomaly_values, 90)))
    if hi <= lo:
        span=max(abs(hi), 1.0)
        return lo - 0.05*span, hi + 0.05*span
    pad=(hi-lo)*0.08
    return lo-pad, hi+pad

def plot_threshold_dynamics(scores, thresholds, labels=None, save_path=None, y_mode='robust'):
    score_values=np.asarray(scores,dtype=float)
    threshold_values=np.asarray([np.nan if t is None else t for t in thresholds],dtype=float)
    plt.figure(figsize=(12,5))
    plt.plot(score_values,label='Anomaly score',linewidth=0.9)
    plt.plot(threshold_values,label='Adaptive threshold',linewidth=1.2)
    if labels is not None:
        idx=[i for i,y in enumerate(labels) if int(y)==1]
        plt.scatter(idx,[score_values[i] for i in idx],marker='x',label='True anomaly',s=28,linewidths=1.2)
    if y_mode == 'robust':
        values=np.concatenate([score_values[np.isfinite(score_values)], threshold_values[np.isfinite(threshold_values)]])
        ylim=_robust_ylim(values, labels=None)
        if ylim:
            plt.ylim(*ylim)
    elif y_mode == 'symlog':
        finite=np.abs(np.concatenate([score_values[np.isfinite(score_values)], threshold_values[np.isfinite(threshold_values)]]))
        linthresh=float(np.percentile(finite, 50)) if finite.size else 1.0
        plt.yscale('symlog', linthresh=max(linthresh, 1e-9))
    plt.xlabel('Sequence index'); plt.ylabel('Score')
    suffix={'robust':'robust view','symlog':'symmetric log view','full':'full scale'}.get(y_mode, y_mode)
    plt.title(f'Adaptive threshold dynamics ({suffix})')
    plt.legend(); plt.tight_layout(); _save(save_path)

def plot_heatmap(event_ids, token_scores, true_mask=None, pad_id=0, title='Anomaly heatmap', save_path=None):
    e=np.array(event_ids); s=np.array(token_scores,dtype=float); valid=e!=pad_id; e=e[valid]; s=s[valid]
    if len(e)==0: return
    if true_mask is not None: true_mask=np.array(true_mask)[valid]
    plt.figure(figsize=(14,2.8)); plt.imshow(s.reshape(1,-1),aspect='auto'); plt.colorbar(label='Contribution'); plt.yticks([]); plt.xticks(range(len(e)), e, rotation=90)
    if true_mask is not None:
        for p in np.where(true_mask==1)[0]: plt.axvline(p, linestyle='--', linewidth=2)
    plt.title(title); plt.xlabel('Event position'); plt.tight_layout(); _save(save_path)

def plot_confusion_matrix_counts(y_true, y_pred, title='Confusion matrix', save_path=None):
    y=np.array(y_true,dtype=int); p=np.array(y_pred,dtype=int)
    tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
    cm=np.array([[tn,fp],[fn,tp]],dtype=int)
    plt.figure(figsize=(5.4,4.6))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xticks([0,1], ['Pred normal','Pred anomaly'])
    plt.yticks([0,1], ['True normal','True anomaly'])
    for i in range(2):
        for j in range(2):
            color='white' if cm[i,j] > cm.max()/2 else 'black'
            plt.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=13,color=color)
    plt.colorbar(label='Count')
    plt.tight_layout(); _save(save_path)

def plot_confusion_matrix_grid(items, title='Confusion matrices overview', save_path=None, max_cols=3):
    if not items:
        return
    cols=min(max_cols, len(items))
    rows=int(np.ceil(len(items)/cols))
    fig, axes=plt.subplots(rows, cols, figsize=(5.2*cols, 4.4*rows), squeeze=False)
    max_count=1
    matrices=[]
    for item in items:
        y=np.array(item['labels'],dtype=int); p=np.array(item['predictions'],dtype=int)
        tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
        fn=int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
        cm=np.array([[tn,fp],[fn,tp]],dtype=int)
        matrices.append(cm)
        max_count=max(max_count, int(cm.max()))
    for ax, item, cm in zip(axes.ravel(), items, matrices):
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=max_count)
        ax.set_title(item.get('title','Confusion matrix'), fontsize=10)
        ax.set_xticks([0,1]); ax.set_xticklabels(['Pred N','Pred A'])
        ax.set_yticks([0,1]); ax.set_yticklabels(['True N','True A'])
        for i in range(2):
            for j in range(2):
                color='white' if cm[i,j] > max_count/2 else 'black'
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=12,color=color)
        tn,fp,fn,tp=cm.ravel()
        precision=tp/max(tp+fp,1)
        recall=tp/max(tp+fn,1)
        fpr=fp/max(fp+tn,1)
        ax.set_xlabel(f'P={precision:.3f} R={recall:.3f} FPR={fpr:.3f}', fontsize=9)
    for ax in axes.ravel()[len(items):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.97])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_aggregate_token_heatmap(score_rows, mask_rows=None, title='Aggregate heatmap', save_path=None):
    scores=np.asarray(score_rows,dtype=float)
    if scores.ndim != 2 or scores.size == 0:
        return
    scores=np.nan_to_num(scores,nan=0.0,posinf=0.0,neginf=0.0)
    mean_scores=scores.mean(axis=0)
    rows=[mean_scores]
    labels=['Mean score']
    if mask_rows is not None and len(mask_rows):
        masks=np.asarray(mask_rows,dtype=float)
        if masks.ndim == 2 and masks.shape[1] == scores.shape[1]:
            rows.append(masks.mean(axis=0))
            labels.append('Anomaly mask freq')
    matrix=np.vstack(rows)
    plt.figure(figsize=(14,2.6 + 1.1*(len(rows)-1)))
    plt.imshow(matrix,aspect='auto')
    plt.colorbar(label='Value')
    plt.yticks(range(len(labels)),labels)
    step=max(1, matrix.shape[1]//32)
    plt.xticks(range(0,matrix.shape[1],step),range(0,matrix.shape[1],step),rotation=90)
    plt.xlabel('Token position')
    plt.title(title)
    plt.tight_layout(); _save(save_path)

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
