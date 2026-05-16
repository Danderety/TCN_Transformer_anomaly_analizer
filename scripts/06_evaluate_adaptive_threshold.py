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
from src.evaluation.thresholds import AdaptiveThreshold
from src.evaluation.detection_metrics import compute_detection_metrics

def make_loader(data,c):
    ds=LogSequenceDataset(data['sequences'],data.get('labels'),max_len=c['data']['max_seq_len'],pad_id=c['data']['pad_id'],session_ids=data.get('session_ids'))
    return DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0))

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/metrics'); ensure_dir(f'{output_dir}/predictions'); mn='tcn_transformer_ae'; ck=Path(output_dir)/'models'/f'{mn}_best.pt'
    if not ck.exists(): raise FileNotFoundError(ck)
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    va=load_json(f"{c['data']['splits_dir']}/val.json"); te=load_json(f"{c['data']['splits_dir']}/test.json")
    model=build_model(mn,c).to(device); model,_=load_checkpoint(model,ck,device)
    va_r=score_dataset(model,make_loader(va,c),device,c['data']['pad_id']); te_r=score_dataset(model,make_loader(te,c),device,c['data']['pad_id'])
    ad=AdaptiveThreshold(c['threshold']['adaptive_window_size'],c['threshold']['adaptive_k'],c['threshold']['adaptive_min_window'],c['threshold']['update_only_normal'])
    ad.warmup([s for s,y in zip(va_r['scores'],va_r['labels']) if int(y)==0]); pred,thr=ad.predict_many(te_r['scores'])
    m=compute_detection_metrics(te_r['labels'],te_r['scores'],pred); m.update({'model':mn,'threshold_type':'adaptive','window_size':c['threshold']['adaptive_window_size'],'k':c['threshold']['adaptive_k']})
    pd.DataFrame([m]).to_csv(f'{output_dir}/metrics/adaptive_threshold_metrics.csv',index=False)
    save_json({'scores':te_r['scores'],'labels':te_r['labels'],'predictions':pred,'thresholds':thr,'session_ids':te_r.get('session_ids',[])}, f'{output_dir}/predictions/adaptive_predictions.json')
    print(m)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
