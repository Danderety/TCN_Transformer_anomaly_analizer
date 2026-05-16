import numpy as np

def normalize_scores(scores):
    a=np.array(scores,dtype=float)
    return np.zeros_like(a) if len(a)==0 or a.max()-a.min()<1e-8 else (a-a.min())/(a.max()-a.min())

def top_positions(token_errors, event_ids=None, pad_id=0, top_k=5):
    s=np.array(token_errors,dtype=float).copy()
    if event_ids is not None:
        e=np.array(event_ids); s[e==pad_id] = -np.inf
    return [int(i) for i in np.argsort(s)[-top_k:][::-1] if np.isfinite(s[i])]

def build_explanation(event_ids, token_errors, vocab=None, pad_id=0, top_k=5):
    pos=top_positions(token_errors,event_ids,pad_id,top_k); events=[]
    for p in pos:
        eid=int(event_ids[p]); events.append({'position':p,'event_id':eid,'event_template':vocab.decode(eid) if vocab else str(eid),'score':float(token_errors[p])})
    return {'top_positions':pos,'top_events':events,'text':f'Наибольший вклад в anomaly score внесли позиции {pos}.'}
