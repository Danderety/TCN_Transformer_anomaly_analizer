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
from src.data.validation_split import combine_validation_with_synthetic_if_needed
from src.models.factory import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.scoring import score_dataset
from src.evaluation.multiscale import score_payload_multiscale
from src.evaluation.thresholds import AdaptiveThreshold
from src.evaluation.detection_metrics import compute_detection_metrics

def _adaptive_k_grid(value):
    if value is None:
        value = [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0]
    if isinstance(value, str):
        value = [x.strip() for x in value.split(',') if x.strip()]
    return [float(x) for x in value]

def _fbeta_score(precision, recall, beta=3.0):
    precision=float(precision or 0.0); recall=float(recall or 0.0); beta2=float(beta)**2
    denom=beta2*precision+recall
    return 0.0 if denom <= 0 else (1.0+beta2)*precision*recall/denom

def _make_adaptive(threshold_cfg, window_size, min_window, k):
    return AdaptiveThreshold(
        window_size=window_size,
        k=k,
        min_window=min_window,
        update_only_normal=threshold_cfg['update_only_normal'],
        method=threshold_cfg.get('adaptive_method', 'mean_std'),
        quantile=threshold_cfg.get('adaptive_quantile', threshold_cfg.get('fixed_percentile', 95)),
        min_scale=threshold_cfg.get('adaptive_min_scale', 1e-6),
        warmup_trim_percentile=threshold_cfg.get('adaptive_warmup_trim_percentile'),
        threshold_floor=threshold_cfg.get('adaptive_threshold_floor'),
        threshold_ceiling=threshold_cfg.get('adaptive_threshold_ceiling'),
    )

def _calibrate_adaptive_k(val_scores, val_labels, warmup_scores, threshold_cfg, window_size, min_window):
    rows=[]
    for k in _adaptive_k_grid(threshold_cfg.get('adaptive_k_grid')):
        ad=_make_adaptive(threshold_cfg, window_size, min_window, k)
        ad.warmup(warmup_scores)
        pred,_thr=ad.predict_many(val_scores)
        metrics=compute_detection_metrics(val_labels,val_scores,pred)
        metrics['_k']=k
        metrics['_fbeta']=_fbeta_score(metrics['precision'],metrics['recall'],threshold_cfg.get('fbeta_beta',3.0))
        rows.append(metrics)
    target_recall=float(threshold_cfg.get('target_recall',0.99))
    max_fpr=float(threshold_cfg.get('max_validation_fpr',0.25))
    has_positive=sum(int(y)==1 for y in val_labels) > 0
    tiers=[
        [m for m in rows if m['recall']+1e-12 >= target_recall and m['false_positive_rate'] <= max_fpr+1e-12],
        [m for m in rows if m['recall']+1e-12 >= target_recall],
        rows,
    ]
    if not has_positive:
        tiers.insert(0,[m for m in rows if m['false_positive_rate'] <= max_fpr+1e-12])
    for tier in tiers:
        if tier:
            best=max(tier,key=lambda m:(m['_fbeta'],m['recall'],-m['false_positive_rate'],-m['_k']))
            return float(best['_k']), best
    return float(threshold_cfg['adaptive_k']), {}

def make_loader(data,c):
    ds=LogSequenceDataset(data['sequences'],data.get('labels'),max_len=c['data']['max_seq_len'],pad_id=c['data']['pad_id'],session_ids=data.get('session_ids'))
    return DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0))

def _adaptive_model_name(c):
    explicit = c.get('threshold', {}).get('adaptive_model')
    if explicit:
        return str(explicit)
    models = [str(x) for x in c.get('evaluation', {}).get('models', []) if str(x) != 'tcn_transformer_ae_ensemble']
    if 'tcn_transformer_ae' in models:
        return 'tcn_transformer_ae'
    if models:
        return models[0]
    return 'tcn_transformer_ae'

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/metrics'); ensure_dir(f'{output_dir}/predictions'); mn=_adaptive_model_name(c); ck=Path(output_dir)/'models'/f'{mn}_best.pt'
    if not ck.exists():
        print(f'Skip adaptive threshold for {mn}: missing {ck}')
        pd.DataFrame([]).to_csv(f'{output_dir}/metrics/adaptive_threshold_metrics.csv',index=False)
        return None
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{c['data']['splits_dir']}/train.json"); va=load_json(f"{c['data']['splits_dir']}/val.json"); va, val_source=combine_validation_with_synthetic_if_needed(c, va); te=load_json(f"{c['data']['splits_dir']}/test.json")
    model=build_model(mn,c).to(device); model,_=load_checkpoint(model,ck,device)
    if c.get('multi_scale', {}).get('enabled', False):
        va_r=score_payload_multiscale(model,va,tr,c,device); te_r=score_payload_multiscale(model,te,tr,c,device)
    else:
        va_r=score_dataset(model,make_loader(va,c),device,c['data']['pad_id'],include_details=False); te_r=score_dataset(model,make_loader(te,c),device,c['data']['pad_id'],include_details=False)
    raw_window=c['threshold']['adaptive_window_size']
    if str(raw_window).lower() == 'auto':
        window_size=min(
            int(c['threshold'].get('adaptive_window_max', 200)),
            max(int(c['threshold'].get('adaptive_window_min', 30)), int(float(c['threshold'].get('adaptive_window_fraction', 0.10)) * len(te_r['scores']))),
        )
    else:
        window_size=int(raw_window)
    threshold_cfg=c['threshold']
    min_window=min(int(threshold_cfg['adaptive_min_window']), window_size)
    warmup_scores=[s for s,y in zip(va_r['scores'],va_r['labels']) if int(y)==0] or va_r['scores']
    adaptive_k=float(threshold_cfg['adaptive_k'])
    adaptive_k_metrics={}
    if threshold_cfg.get('adaptive_calibrate_k', False):
        adaptive_k,adaptive_k_metrics=_calibrate_adaptive_k(va_r['scores'],va_r['labels'],warmup_scores,threshold_cfg,window_size,min_window)
    ad=_make_adaptive(threshold_cfg,window_size,min_window,adaptive_k)
    ad.warmup(warmup_scores); pred,thr=ad.predict_many(te_r['scores'])
    m=compute_detection_metrics(te_r['labels'],te_r['scores'],pred); m.update({
        'model':mn,
        'threshold_type':'adaptive',
        'window_size':window_size,
        'adaptive_window_size_config':raw_window,
        'adaptive_method':threshold_cfg.get('adaptive_method', 'mean_std'),
        'adaptive_quantile':threshold_cfg.get('adaptive_quantile', threshold_cfg.get('fixed_percentile', 95)),
        'adaptive_warmup_trim_percentile':threshold_cfg.get('adaptive_warmup_trim_percentile'),
        'k':adaptive_k,
        'configured_k':threshold_cfg['adaptive_k'],
        'adaptive_calibrate_k':threshold_cfg.get('adaptive_calibrate_k', False),
        'adaptive_validation_f1':adaptive_k_metrics.get('f1'),
        'adaptive_validation_recall':adaptive_k_metrics.get('recall'),
        'adaptive_validation_fpr':adaptive_k_metrics.get('false_positive_rate'),
        'validation_monitor_source':val_source,
    })
    pd.DataFrame([m]).to_csv(f'{output_dir}/metrics/adaptive_threshold_metrics.csv',index=False)
    save_json({'scores':te_r['scores'],'labels':te_r['labels'],'predictions':pred,'thresholds':thr,'session_ids':te_r.get('session_ids',[]),'adaptive_config':{
        'window_size':window_size,
        'min_window':min_window,
        'method':threshold_cfg.get('adaptive_method', 'mean_std'),
        'quantile':threshold_cfg.get('adaptive_quantile', threshold_cfg.get('fixed_percentile', 95)),
        'k':adaptive_k,
        'configured_k':threshold_cfg['adaptive_k'],
        'calibrate_k':threshold_cfg.get('adaptive_calibrate_k', False),
        'calibration_metrics':{k:v for k,v in adaptive_k_metrics.items() if not k.startswith('_')},
        'warmup_trim_percentile':threshold_cfg.get('adaptive_warmup_trim_percentile'),
        'update_only_normal':threshold_cfg['update_only_normal'],
    }}, f'{output_dir}/predictions/adaptive_predictions.json')
    print(m)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
