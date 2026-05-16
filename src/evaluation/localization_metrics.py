import numpy as np

def compute_topk_hit(token_scores, true_masks, k=3):
    hits=[]
    for scores,mask in zip(token_scores,true_masks):
        scores=np.array(scores); mask=np.array(mask)
        if mask.sum()==0: continue
        idx=np.argsort(scores)[-k:]; hits.append(int(mask[idx].sum()>0))
    return float(np.mean(hits)) if hits else 0.0

def compute_mean_rank(token_scores, true_masks):
    ranks=[]
    for scores,mask in zip(token_scores,true_masks):
        scores=np.array(scores); mask=np.array(mask); pos=np.where(mask==1)[0]
        if len(pos)==0: continue
        ranking=np.argsort(scores)[::-1]
        ranks.append(min(np.where(ranking==p)[0][0]+1 for p in pos))
    return float(np.mean(ranks)) if ranks else 0.0

def compute_iou_at_k(token_scores, true_masks, k=3):
    ious=[]
    for scores,mask in zip(token_scores,true_masks):
        scores=np.array(scores); mask=np.array(mask); true=set(np.where(mask==1)[0])
        if not true: continue
        pred=set(np.argsort(scores)[-k:]); ious.append(len(true & pred)/len(true | pred))
    return float(np.mean(ious)) if ious else 0.0

def compute_localization_metrics(token_scores, true_masks, top_k_values=(1,3,5)):
    m={}
    for k in top_k_values:
        m[f'top_{k}_hit']=compute_topk_hit(token_scores,true_masks,k)
        m[f'iou_at_{k}']=compute_iou_at_k(token_scores,true_masks,k)
    m['mean_rank']=compute_mean_rank(token_scores,true_masks)
    return m
