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
from src.xai.attention import average_attention_results, capture_attention_dataset

def _limit_payload(payload, max_samples):
    max_samples = int(max_samples or 0)
    if max_samples <= 0:
        return payload
    limited = {}
    for key, value in payload.items():
        if isinstance(value, list):
            limited[key] = value[:max_samples]
        else:
            limited[key] = value
    return limited

def _save_attention_for_model(model, loader, device, config, output_dir, model_name):
    max_matrices = int(config.get('xai', {}).get('save_attention_matrices', 0) or 0)
    r = capture_attention_dataset(model, loader, device, config['data']['pad_id'], max_matrices=max_matrices)
    m = compute_localization_metrics(r['attention_scores'], r['localization_masks'], config['evaluation']['top_k'])
    m.update({'model':model_name,'xai_method':'transformer_attention_received'})
    save_json(r, f'{output_dir}/predictions/{model_name}_attention_predictions.json')
    return m

def _save_attention_for_ensemble(config, loader, device, output_dir):
    seeds = config.get('ensemble', {}).get('member_seeds', [])
    results = []
    used = []
    for seed in seeds:
        ck = Path(output_dir)/'models'/f'tcn_transformer_ae_seed{int(seed)}_best.pt'
        if not ck.exists():
            print(f'Skip attention ensemble member seed={seed}: missing {ck}')
            continue
        model = build_model('tcn_transformer_ae', config).to(device)
        model, _ = load_checkpoint(model, ck, device)
        try:
            results.append(capture_attention_dataset(model, loader, device, config['data']['pad_id']))
            used.append(int(seed))
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not results:
        print('Skip attention for tcn_transformer_ae_ensemble: no trained members found')
        return None
    r = average_attention_results(results)
    m = compute_localization_metrics(r['attention_scores'], r['localization_masks'], config['evaluation']['top_k'])
    m.update({
        'model':'tcn_transformer_ae_ensemble',
        'xai_method':'ensemble_transformer_attention_received',
        'ensemble_members':len(used),
        'ensemble_seeds':' '.join(map(str, used)),
    })
    r['member_seeds'] = used
    save_json(r, f'{output_dir}/predictions/tcn_transformer_ae_ensemble_attention_predictions.json')
    return m

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/metrics'); ensure_dir(f'{output_dir}/predictions'); device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    sy=load_json(f"{c['data']['splits_dir']}/test_synthetic.json")
    sy=_limit_payload(sy, c.get('evaluation', {}).get('max_localization_samples', 0))
    ds=LogSequenceDataset(sy['sequences'],sy.get('labels'),sy.get('localization_masks'),c['data']['max_seq_len'],c['data']['pad_id'],sy.get('session_ids'))
    ld=DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0)); rows=[]
    for mn in c['evaluation']['models']:
        if mn == 'tcn_transformer_ae_ensemble':
            attention_metrics = _save_attention_for_ensemble(c, ld, device, output_dir)
            if attention_metrics:
                rows.append(attention_metrics); print(attention_metrics)
            continue
        ck=Path(output_dir)/'models'/f'{mn}_best.pt'
        if not ck.exists(): print(f'Skip {mn}: missing {ck}'); continue
        model=build_model(mn,c).to(device); model,_=load_checkpoint(model,ck,device); r=score_dataset(model,ld,device,c['data']['pad_id'])
        m=compute_localization_metrics(r['token_errors'],r['localization_masks'],c['evaluation']['top_k']); m.update({'model':mn,'xai_method':'reconstruction_error_heatmap'}); rows.append(m); print(m)
        save_json({'inputs':r['inputs'],'token_errors':r['token_errors'],'localization_masks':r['localization_masks'],'labels':r['labels'],'session_ids':r.get('session_ids',[])}, f'{output_dir}/predictions/{mn}_localization_predictions.json')
        if mn == 'tcn_transformer_ae':
            attention_metrics = _save_attention_for_model(model, ld, device, c, output_dir, mn)
            rows.append(attention_metrics); print(attention_metrics)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(f'{output_dir}/metrics/localization_metrics.csv',index=False)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
