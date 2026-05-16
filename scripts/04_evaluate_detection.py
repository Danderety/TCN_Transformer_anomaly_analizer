import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.io import load_json, save_json, ensure_dir
from src.data.dataset import LogSequenceDataset
from src.models.factory import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.scoring import score_dataset
from src.evaluation.thresholds import compute_fixed_threshold, apply_threshold
from src.evaluation.detection_metrics import compute_detection_metrics

def loader(data,c):
    ds=LogSequenceDataset(data['sequences'],data.get('labels'),max_len=c['data']['max_seq_len'],pad_id=c['data']['pad_id'],session_ids=data.get('session_ids'))
    return DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0))

def eval_one(c, model_name):
    ck=Path(f'outputs/models/{model_name}_best.pt')
    if not ck.exists(): print(f'Skip {model_name}: missing {ck}'); return None
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    va=load_json(f"{c['data']['splits_dir']}/val.json"); te=load_json(f"{c['data']['splits_dir']}/test.json")
    model=build_model(model_name,c).to(device); model,_=load_checkpoint(model,ck,device)
    va_r=score_dataset(model,loader(va,c),device,c['data']['pad_id']); te_r=score_dataset(model,loader(te,c),device,c['data']['pad_id'])
    normal=[s for s,y in zip(va_r['scores'],va_r['labels']) if int(y)==0] or va_r['scores']
    thr=compute_fixed_threshold(normal,c['threshold']['fixed_percentile']); pred=apply_threshold(te_r['scores'],thr)
    m=compute_detection_metrics(te_r['labels'],te_r['scores'],pred); m.update({'model':model_name,'threshold_type':'fixed','threshold':thr})
    save_json({'scores':te_r['scores'],'labels':te_r['labels'],'predictions':pred,'threshold':thr,'session_ids':te_r.get('session_ids',[])}, f'outputs/predictions/{model_name}_test_predictions.json')
    return m

def main(config_path):
    c=load_config(config_path); ensure_dir('outputs/metrics'); ensure_dir('outputs/predictions'); rows=[]
    for mn in c['evaluation']['models']:
        m=eval_one(c,mn)
        if m: rows.append(m); print(m)
    pd.DataFrame(rows).to_csv('outputs/metrics/detection_metrics.csv',index=False)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
