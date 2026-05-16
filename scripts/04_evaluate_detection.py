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
from src.evaluation.ensemble import combine_ensemble_scores
from src.evaluation.thresholds import apply_threshold, select_validation_safe_threshold
from src.evaluation.detection_metrics import compute_detection_metrics

def loader(data,c):
    ds=LogSequenceDataset(data['sequences'],data.get('labels'),max_len=c['data']['max_seq_len'],pad_id=c['data']['pad_id'],session_ids=data.get('session_ids'))
    return DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0))

def _score_single(c, model_name, checkpoint_path, train, val, test, device):
    model=build_model(model_name,c).to(device); model,_=load_checkpoint(model,checkpoint_path,device)
    return (
        score_dataset(model,loader(train,c),device,c['data']['pad_id']),
        score_dataset(model,loader(val,c),device,c['data']['pad_id']),
        score_dataset(model,loader(test,c),device,c['data']['pad_id']),
    )

def _raw_method_for(c, model_name):
    spot_models = set(c.get('threshold', {}).get('spot_for_models', []))
    return 'spot' if model_name in spot_models else c.get('threshold', {}).get('method', 'quantile')

def _metrics_row(model_name, train_scores, val_scores, val_labels, test_scores, test_labels, c):
    threshold_info = select_validation_safe_threshold(
        train_scores,
        val_scores,
        val_labels,
        c.get('threshold', {}),
        raw_method=_raw_method_for(c, model_name),
    )
    pred=apply_threshold(test_scores,threshold_info['threshold'])
    metrics=compute_detection_metrics(test_labels,test_scores,pred)
    metrics.update({
        'model':model_name,
        'threshold_type':threshold_info['threshold_type'],
        'threshold':threshold_info['threshold'],
        'raw_threshold':threshold_info['raw_threshold'],
        'raw_threshold_note':threshold_info['raw_note'],
        'safe_threshold':threshold_info['safe_threshold'],
        'safe_threshold_note':threshold_info['safe_note'],
        'was_safety_applied':threshold_info['was_safety_applied'],
        'val_precision':threshold_info['metrics']['precision'],
        'val_recall':threshold_info['metrics']['recall'],
        'val_f1':threshold_info['metrics']['f1'],
        'val_fp':threshold_info['metrics']['fp'],
        'val_fn':threshold_info['metrics']['fn'],
        'raw_val_fp':threshold_info['raw_metrics']['fp'],
        'raw_val_fn':threshold_info['raw_metrics']['fn'],
        'safe_val_fp':threshold_info['safe_metrics']['fp'],
        'safe_val_fn':threshold_info['safe_metrics']['fn'],
    })
    return metrics, pred, threshold_info

def eval_ensemble(c, train, val, test, device):
    ensemble_config=c.get('ensemble', {})
    seeds=ensemble_config.get('member_seeds', [])
    if not seeds:
        print('Skip tcn_transformer_ae_ensemble: no ensemble.member_seeds')
        return None
    output_dir=c.get('project', {}).get('output_dir', 'outputs')
    member_train=[]; member_val=[]; member_test=[]; used=[]
    for seed in seeds:
        ck=Path(output_dir)/'models'/f'tcn_transformer_ae_seed{int(seed)}_best.pt'
        if not ck.exists():
            print(f'Skip ensemble member seed={seed}: missing {ck}')
            continue
        tr_r, va_r, te_r=_score_single(c,'tcn_transformer_ae',ck,train,val,test,device)
        member_train.append(tr_r['scores']); member_val.append(va_r['scores']); member_test.append(te_r['scores']); used.append(int(seed))
    if not member_test:
        print('Skip tcn_transformer_ae_ensemble: no trained members found')
        return None
    norm=ensemble_config.get('score_norm','robust_z_train')
    train_scores, train_stats=combine_ensemble_scores(member_train,member_train,norm)
    val_scores, val_stats=combine_ensemble_scores(member_val,member_train,norm)
    test_scores, test_stats=combine_ensemble_scores(member_test,member_train,norm)
    metrics,pred,threshold_info=_metrics_row('tcn_transformer_ae_ensemble',train_scores,val_scores,val['labels'],test_scores,test['labels'],c)
    metrics.update({'ensemble_members':len(used),'ensemble_seeds':' '.join(map(str,used))})
    save_json({
        'scores':test_scores.tolist(),
        'labels':test['labels'],
        'predictions':pred,
        'threshold':threshold_info['threshold'],
        'raw_threshold':threshold_info['raw_threshold'],
        'safe_threshold':threshold_info['safe_threshold'],
        'was_safety_applied':threshold_info['was_safety_applied'],
        'member_seeds':used,
        'train_norm_stats':train_stats,
        'val_norm_stats':val_stats,
        'test_norm_stats':test_stats,
        'session_ids':test.get('session_ids',[]),
    }, f"{c.get('project', {}).get('output_dir', 'outputs')}/predictions/tcn_transformer_ae_ensemble_test_predictions.json")
    return metrics

def eval_one(c, model_name):
    if model_name == 'tcn_transformer_ae_ensemble':
        return None
    output_dir=c.get('project', {}).get('output_dir', 'outputs')
    ck=Path(output_dir)/'models'/f'{model_name}_best.pt'
    if not ck.exists(): print(f'Skip {model_name}: missing {ck}'); return None
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{c['data']['splits_dir']}/train.json"); va=load_json(f"{c['data']['splits_dir']}/val.json"); te=load_json(f"{c['data']['splits_dir']}/test.json")
    tr_r,va_r,te_r=_score_single(c,model_name,ck,tr,va,te,device)
    m,pred,threshold_info=_metrics_row(model_name,tr_r['scores'],va_r['scores'],va_r['labels'],te_r['scores'],te_r['labels'],c)
    save_json({
        'scores':te_r['scores'],
        'labels':te_r['labels'],
        'predictions':pred,
        'threshold':threshold_info['threshold'],
        'raw_threshold':threshold_info['raw_threshold'],
        'safe_threshold':threshold_info['safe_threshold'],
        'was_safety_applied':threshold_info['was_safety_applied'],
        'session_ids':te_r.get('session_ids',[])
    }, f"{output_dir}/predictions/{model_name}_test_predictions.json")
    return m

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/metrics'); ensure_dir(f'{output_dir}/predictions'); rows=[]
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{c['data']['splits_dir']}/train.json"); va=load_json(f"{c['data']['splits_dir']}/val.json"); te=load_json(f"{c['data']['splits_dir']}/test.json")
    for mn in c['evaluation']['models']:
        m=eval_ensemble(c,tr,va,te,device) if mn == 'tcn_transformer_ae_ensemble' else eval_one(c,mn)
        if m: rows.append(m); print(m)
    pd.DataFrame(rows).to_csv(f'{output_dir}/metrics/detection_metrics.csv',index=False)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
