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
from src.evaluation.localization_metrics import compute_localization_metrics

def main(config_path):
    c=load_config(config_path); ensure_dir('outputs/metrics'); ensure_dir('outputs/predictions'); device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    sy=load_json(f"{c['data']['splits_dir']}/test_synthetic.json")
    ds=LogSequenceDataset(sy['sequences'],sy.get('labels'),sy.get('localization_masks'),c['data']['max_seq_len'],c['data']['pad_id'],sy.get('session_ids'))
    ld=DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0)); rows=[]
    for mn in c['evaluation']['models']:
        ck=Path(f'outputs/models/{mn}_best.pt')
        if not ck.exists(): print(f'Skip {mn}: missing {ck}'); continue
        model=build_model(mn,c).to(device); model,_=load_checkpoint(model,ck,device); r=score_dataset(model,ld,device,c['data']['pad_id'])
        m=compute_localization_metrics(r['token_errors'],r['localization_masks'],c['evaluation']['top_k']); m.update({'model':mn,'xai_method':'reconstruction_error_heatmap'}); rows.append(m); print(m)
        save_json({'inputs':r['inputs'],'token_errors':r['token_errors'],'localization_masks':r['localization_masks'],'labels':r['labels'],'session_ids':r.get('session_ids',[])}, f'outputs/predictions/{mn}_localization_predictions.json')
    pd.DataFrame(rows).to_csv('outputs/metrics/localization_metrics.csv',index=False)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
