import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import numpy as np
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
from src.evaluation.ensemble import combine_ensemble_scores
from src.evaluation.thresholds import apply_threshold, select_validation_safe_threshold, threshold_for_exact_recall
from src.evaluation.detection_metrics import compute_detection_metrics

def loader(data,c):
    ds=LogSequenceDataset(data['sequences'],data.get('labels'),max_len=c['data']['max_seq_len'],pad_id=c['data']['pad_id'],session_ids=data.get('session_ids'))
    return DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=False,num_workers=c['training'].get('num_workers',0))

def _score_single(c, model_name, checkpoint_path, train, val, test, device):
    model=build_model(model_name,c).to(device); model,_=load_checkpoint(model,checkpoint_path,device)
    try:
        if c.get('multi_scale', {}).get('enabled', False):
            return (
                score_payload_multiscale(model,train,train,c,device),
                score_payload_multiscale(model,val,train,c,device),
                score_payload_multiscale(model,test,train,c,device),
            )
        return (
            score_dataset(model,loader(train,c),device,c['data']['pad_id'],include_details=False),
            score_dataset(model,loader(val,c),device,c['data']['pad_id'],include_details=False),
            score_dataset(model,loader(test,c),device,c['data']['pad_id'],include_details=False),
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    metrics.update(_oracle_threshold_diagnostics(test_labels, test_scores))
    exact_info = None
    if c.get('threshold', {}).get('exact_recall_diagnostics', True) and sum(int(y) == 1 for y in test_labels) > 0:
        exact_threshold, exact_metrics, exact_note = threshold_for_exact_recall(test_labels, test_scores)
        exact_info = {'threshold': exact_threshold, 'metrics': exact_metrics, 'note': exact_note}
        metrics.update({
            'test_exact_recall_threshold': exact_threshold,
            'test_exact_recall_note': exact_note,
            'test_exact_precision': exact_metrics['precision'],
            'test_exact_recall': exact_metrics['recall'],
            'test_exact_f1': exact_metrics['f1'],
            'test_exact_fp': exact_metrics['fp'],
            'test_exact_fn': exact_metrics['fn'],
            'test_fp_cost_to_recall_1': exact_metrics['fp'] - metrics['fp'],
            'test_recovered_fn_to_recall_1': metrics['fn'] - exact_metrics['fn'],
        })
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
    return metrics, pred, threshold_info, exact_info

def _oracle_threshold_diagnostics(labels, scores):
    y=np.asarray(labels,dtype=int)
    s=np.asarray(scores,dtype=float)
    s=np.nan_to_num(s,nan=np.inf,posinf=np.inf,neginf=-np.inf)
    normal=s[y==0]
    anomaly=s[y==1]
    out={
        'oracle_perfect_possible': None,
        'oracle_perfect_threshold': None,
        'oracle_best_f1': None,
        'oracle_best_recall': None,
        'oracle_best_specificity': None,
        'oracle_best_fp': None,
        'oracle_best_fn': None,
        'oracle_best_threshold': None,
    }
    if len(normal)==0 or len(anomaly)==0 or len(s)==0:
        return out
    max_normal=float(np.max(normal))
    min_anomaly=float(np.min(anomaly))
    perfect=max_normal < min_anomaly
    out['oracle_perfect_possible']=bool(perfect)
    if perfect:
        out['oracle_perfect_threshold']=float((max_normal + min_anomaly) / 2.0)
    candidates=np.unique(s)
    candidates=np.concatenate([[np.nextafter(float(candidates.min()), -np.inf)], candidates, [np.nextafter(float(candidates.max()), np.inf)]])
    best=None
    for threshold in candidates:
        p=(s>=float(threshold)).astype(int).tolist()
        m=compute_detection_metrics(y,s,p)
        specificity=1.0 - m['false_positive_rate']
        key=(m['f1'], m['recall'], specificity, -m['fp'], -m['fn'])
        if best is None or key > best[0]:
            best=(key, threshold, m, specificity)
    if best:
        _key, threshold, m, specificity=best
        out.update({
            'oracle_best_f1':m['f1'],
            'oracle_best_recall':m['recall'],
            'oracle_best_specificity':specificity,
            'oracle_best_fp':m['fp'],
            'oracle_best_fn':m['fn'],
            'oracle_best_threshold':float(threshold),
        })
    return out

def _exact_info_if_needed(labels, scores, c):
    if c.get('threshold', {}).get('exact_recall_diagnostics', True) and sum(int(y) == 1 for y in labels) > 0:
        exact_threshold, exact_metrics, exact_note = threshold_for_exact_recall(labels, scores)
        return {'threshold': exact_threshold, 'metrics': exact_metrics, 'note': exact_note}
    return None

def _save_predictions(output_dir, model_name, split_name, scores, labels, pred, threshold_info, exact_info, session_ids=None, extra=None):
    suffix = 'test_predictions' if split_name == 'real' else f'{split_name}_test_predictions'
    payload = {
        'split': split_name,
        'scores': scores.tolist() if hasattr(scores, 'tolist') else scores,
        'labels': labels,
        'predictions': pred,
        'threshold': threshold_info['threshold'],
        'raw_threshold': threshold_info['raw_threshold'],
        'safe_threshold': threshold_info['safe_threshold'],
        'was_safety_applied': threshold_info['was_safety_applied'],
        'test_exact_recall': exact_info,
        'session_ids': session_ids or [],
    }
    if extra:
        payload.update(extra)
    save_json(payload, f"{output_dir}/predictions/{model_name}_{suffix}.json")

def _score_synthetic_one(c, model_name, checkpoint_path, synthetic, device):
    model=build_model(model_name,c).to(device); model,_=load_checkpoint(model,checkpoint_path,device)
    try:
        if c.get('multi_scale', {}).get('enabled', False):
            train=load_json(f"{c['data']['splits_dir']}/train.json")
            return score_payload_multiscale(model,synthetic,train,c,device)
        return score_dataset(model,loader(synthetic,c),device,c['data']['pad_id'],include_details=False)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def eval_ensemble(c, train, val, test, synthetic, device):
    ensemble_config=c.get('ensemble', {})
    seeds=ensemble_config.get('member_seeds', [])
    if not seeds:
        print('Skip tcn_transformer_ae_ensemble: no ensemble.member_seeds')
        return None
    output_dir=c.get('project', {}).get('output_dir', 'outputs')
    member_train=[]; member_val=[]; member_test=[]; member_synthetic=[]; used=[]
    for seed in seeds:
        ck=Path(output_dir)/'models'/f'tcn_transformer_ae_seed{int(seed)}_best.pt'
        if not ck.exists():
            print(f'Skip ensemble member seed={seed}: missing {ck}')
            continue
        tr_r, va_r, te_r=_score_single(c,'tcn_transformer_ae',ck,train,val,test,device)
        member_train.append(tr_r['scores']); member_val.append(va_r['scores']); member_test.append(te_r['scores'])
        if synthetic:
            sy_r=_score_synthetic_one(c,'tcn_transformer_ae',ck,synthetic,device)
            member_synthetic.append(sy_r['scores'])
        used.append(int(seed))
    if not member_test:
        print('Skip tcn_transformer_ae_ensemble: no trained members found')
        return None
    norm=ensemble_config.get('score_norm','robust_z_train')
    train_scores, train_stats=combine_ensemble_scores(member_train,member_train,norm)
    val_scores, val_stats=combine_ensemble_scores(member_val,member_train,norm)
    test_scores, test_stats=combine_ensemble_scores(member_test,member_train,norm)
    metrics,pred,threshold_info,exact_info=_metrics_row('tcn_transformer_ae_ensemble',train_scores,val_scores,val['labels'],test_scores,test['labels'],c)
    metrics.update({'ensemble_members':len(used),'ensemble_seeds':' '.join(map(str,used))})
    extra = {
        'member_seeds':used,
        'train_norm_stats':train_stats,
        'val_norm_stats':val_stats,
        'test_norm_stats':test_stats,
    }
    _save_predictions(output_dir,'tcn_transformer_ae_ensemble','real',test_scores,test['labels'],pred,threshold_info,exact_info,test.get('session_ids',[]),extra)
    if synthetic and member_synthetic:
        synthetic_scores, synthetic_stats=combine_ensemble_scores(member_synthetic,member_train,norm)
        synthetic_pred=apply_threshold(synthetic_scores,threshold_info['threshold'])
        synthetic_exact=_exact_info_if_needed(synthetic['labels'],synthetic_scores,c)
        _save_predictions(output_dir,'tcn_transformer_ae_ensemble','synthetic',synthetic_scores,synthetic['labels'],synthetic_pred,threshold_info,synthetic_exact,synthetic.get('session_ids',[]),{'synthetic_norm_stats':synthetic_stats, **extra})
        combined_scores=np.concatenate([test_scores, synthetic_scores])
        combined_labels=test['labels'] + synthetic['labels']
        combined_pred=pred + synthetic_pred
        combined_exact=_exact_info_if_needed(combined_labels,combined_scores,c)
        _save_predictions(output_dir,'tcn_transformer_ae_ensemble','combined',combined_scores,combined_labels,combined_pred,threshold_info,combined_exact,(test.get('session_ids',[]) + synthetic.get('session_ids',[])),extra)
    return metrics

def eval_one(c, model_name):
    if model_name == 'tcn_transformer_ae_ensemble':
        return None
    output_dir=c.get('project', {}).get('output_dir', 'outputs')
    ck=Path(output_dir)/'models'/f'{model_name}_best.pt'
    if not ck.exists(): print(f'Skip {model_name}: missing {ck}'); return None
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{c['data']['splits_dir']}/train.json"); va=load_json(f"{c['data']['splits_dir']}/val.json"); va, val_source=combine_validation_with_synthetic_if_needed(c, va); te=load_json(f"{c['data']['splits_dir']}/test.json")
    synthetic_path=Path(c['data']['splits_dir'])/'test_synthetic.json'
    synthetic=load_json(synthetic_path) if synthetic_path.exists() else None
    tr_r,va_r,te_r=_score_single(c,model_name,ck,tr,va,te,device)
    m,pred,threshold_info,exact_info=_metrics_row(model_name,tr_r['scores'],va_r['scores'],va_r['labels'],te_r['scores'],te_r['labels'],c)
    m['validation_monitor_source'] = val_source
    _save_predictions(output_dir,model_name,'real',te_r['scores'],te_r['labels'],pred,threshold_info,exact_info,te_r.get('session_ids',[]),{'validation_monitor_source':val_source})
    if synthetic:
        sy_r=_score_synthetic_one(c,model_name,ck,synthetic,device)
        sy_pred=apply_threshold(sy_r['scores'],threshold_info['threshold'])
        sy_exact=_exact_info_if_needed(sy_r['labels'],sy_r['scores'],c)
        _save_predictions(output_dir,model_name,'synthetic',sy_r['scores'],sy_r['labels'],sy_pred,threshold_info,sy_exact,sy_r.get('session_ids',[]),{'validation_monitor_source':val_source})
        combined_scores=te_r['scores'] + sy_r['scores']
        combined_labels=te_r['labels'] + sy_r['labels']
        combined_pred=pred + sy_pred
        combined_exact=_exact_info_if_needed(combined_labels,combined_scores,c)
        _save_predictions(output_dir,model_name,'combined',combined_scores,combined_labels,combined_pred,threshold_info,combined_exact,(te_r.get('session_ids',[]) + sy_r.get('session_ids',[])),{'validation_monitor_source':val_source})
    return m

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/metrics'); ensure_dir(f'{output_dir}/predictions'); rows=[]
    device=torch.device(c['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{c['data']['splits_dir']}/train.json"); va=load_json(f"{c['data']['splits_dir']}/val.json"); va, val_source=combine_validation_with_synthetic_if_needed(c, va); te=load_json(f"{c['data']['splits_dir']}/test.json")
    synthetic_path=Path(c['data']['splits_dir'])/'test_synthetic.json'
    synthetic=load_json(synthetic_path) if synthetic_path.exists() else None
    for mn in c['evaluation']['models']:
        m=eval_ensemble(c,tr,va,te,synthetic,device) if mn == 'tcn_transformer_ae_ensemble' else eval_one(c,mn)
        if m:
            m.setdefault('validation_monitor_source', val_source)
            rows.append(m); print(m)
    pd.DataFrame(rows).to_csv(f'{output_dir}/metrics/detection_metrics.csv',index=False)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
