import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix

def compute_detection_metrics(y_true, scores, y_pred):
    y=np.array(y_true).astype(int); s=np.array(scores); p=np.array(y_pred).astype(int)
    metrics={'precision': float(precision_score(y,p,zero_division=0)), 'recall': float(recall_score(y,p,zero_division=0)), 'f1': float(f1_score(y,p,zero_division=0))}
    metrics['pr_auc'] = float(average_precision_score(y,s)) if len(set(y))>1 else None
    try: metrics['roc_auc'] = float(roc_auc_score(y,s))
    except ValueError: metrics['roc_auc'] = None
    tn,fp,fn,tp = confusion_matrix(y,p,labels=[0,1]).ravel()
    metrics.update({'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'false_positive_rate':float(fp/max(fp+tn,1)),'false_negative_rate':float(fn/max(fn+tp,1))})
    return metrics
