import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
from pathlib import Path
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
from src.utils.config import load_config
from src.utils.io import load_json, ensure_dir
from src.visualization.plots import (
    plot_aggregate_token_heatmap,
    plot_confusion_matrix_counts,
    plot_confusion_matrix_grid,
    plot_score_distribution,
    plot_threshold_dynamics,
    plot_heatmap,
    plot_model_comparison,
)

def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()

def _safe_name(value):
    return str(value).replace('/', '_').replace(' ', '_')

def _anomaly_rows(payload, score_key):
    labels=payload.get('labels', [])
    scores=payload.get(score_key, [])
    masks=payload.get('localization_masks', [])
    selected_scores=[]; selected_masks=[]
    for i,row in enumerate(scores):
        if labels and int(labels[i]) != 1:
            continue
        selected_scores.append(row)
        if masks:
            selected_masks.append(masks[i])
    return selected_scores, selected_masks

def _prediction_name_parts(pred_path):
    name = pred_path.name
    if name.endswith('_combined_test_predictions.json'):
        return name.replace('_combined_test_predictions.json', ''), 'combined'
    if name.endswith('_synthetic_test_predictions.json'):
        return name.replace('_synthetic_test_predictions.json', ''), 'synthetic'
    if name.endswith('_test_predictions.json'):
        return name.replace('_test_predictions.json', ''), 'real'
    return pred_path.stem, 'unknown'

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/figures'); ensure_dir(f'{output_dir}/reports'); ensure_dir(f'{output_dir}/metrics')
    det=read_csv(f'{output_dir}/metrics/detection_metrics.csv'); ad=read_csv(f'{output_dir}/metrics/adaptive_threshold_metrics.csv'); loc=read_csv(f'{output_dir}/metrics/localization_metrics.csv')
    final=pd.concat([x for x in [det,ad] if not x.empty], ignore_index=True) if (not det.empty or not ad.empty) else pd.DataFrame()
    if not final.empty:
        final.to_csv(f'{output_dir}/metrics/final_comparison.csv',index=False); plot_model_comparison(final,'f1',f'{output_dir}/figures/model_comparison_f1.png')
    confusion_items=[]
    for pred_path in sorted((Path(output_dir)/'predictions').glob('*_test_predictions.json')):
        payload=load_json(pred_path)
        if 'labels' not in payload or 'predictions' not in payload:
            continue
        model_name, split_name = _prediction_name_parts(pred_path)
        title=f'{split_name}: {model_name}'
        confusion_items.append({'title': title, 'labels': payload['labels'], 'predictions': payload['predictions']})
        plot_confusion_matrix_counts(
            payload['labels'],
            payload['predictions'],
            f'Confusion matrix ({split_name}): {model_name}',
            f'{output_dir}/figures/confusion_matrix_{_safe_name(model_name)}_{split_name}.png'
        )
    adaptive_pred_path=Path(output_dir)/'predictions'/'adaptive_predictions.json'
    if adaptive_pred_path.exists():
        payload=load_json(adaptive_pred_path)
        if 'labels' in payload and 'predictions' in payload:
            title='real: adaptive_threshold'
            confusion_items.append({'title': title, 'labels': payload['labels'], 'predictions': payload['predictions']})
            plot_confusion_matrix_counts(
                payload['labels'],
                payload['predictions'],
                'Confusion matrix (real): adaptive_threshold',
                f'{output_dir}/figures/confusion_matrix_adaptive_threshold_real.png'
            )
    if confusion_items:
        plot_confusion_matrix_grid(
            confusion_items,
            'Confusion matrices overview',
            f'{output_dir}/figures/confusion_matrices_overview.png',
        )
    pred=Path(output_dir)/'predictions'/'tcn_transformer_ae_test_predictions.json'
    if pred.exists():
        pr=load_json(pred); ns=[s for s,y in zip(pr['scores'],pr['labels']) if int(y)==0]; ans=[s for s,y in zip(pr['scores'],pr['labels']) if int(y)==1]
        plot_score_distribution(ns,ans,f'{output_dir}/figures/anomaly_score_distribution.png')
    ap=Path(output_dir)/'predictions'/'adaptive_predictions.json'
    if ap.exists():
        a=load_json(ap)
        plot_threshold_dynamics(a['scores'],a['thresholds'],a['labels'],f'{output_dir}/figures/adaptive_threshold_dynamics.png',y_mode='robust')
        plot_threshold_dynamics(a['scores'],a['thresholds'],a['labels'],f'{output_dir}/figures/adaptive_threshold_dynamics_full_scale.png',y_mode='full')
        plot_threshold_dynamics(a['scores'],a['thresholds'],a['labels'],f'{output_dir}/figures/adaptive_threshold_dynamics_symlog.png',y_mode='symlog')
    lp=Path(output_dir)/'predictions'/'tcn_transformer_ae_localization_predictions.json'
    if lp.exists():
        l=load_json(lp); count=0
        for i in range(len(l['inputs'])):
            if int(l['labels'][i]) != 1: continue
            plot_heatmap(l['inputs'][i],l['token_errors'][i],l['localization_masks'][i],c['data']['pad_id'],f'Heatmap example {i}',f'{output_dir}/figures/heatmap_example_{i}.png')
            count += 1
            if count >= c['evaluation']['num_heatmap_examples']: break
    for loc_path in sorted((Path(output_dir)/'predictions').glob('*_localization_predictions.json')):
        payload=load_json(loc_path)
        if 'token_errors' not in payload:
            continue
        model_name=loc_path.name.replace('_localization_predictions.json','')
        scores,masks=_anomaly_rows(payload,'token_errors')
        plot_aggregate_token_heatmap(
            scores,
            masks,
            f'Aggregate reconstruction heatmap: {model_name}',
            f'{output_dir}/figures/aggregate_heatmap_{_safe_name(model_name)}_reconstruction.png'
        )
    for model_name in ['tcn_transformer_ae', 'tcn_transformer_ae_ensemble']:
        apath=Path(output_dir)/'predictions'/f'{model_name}_attention_predictions.json'
        if not apath.exists():
            continue
        a=load_json(apath); count=0
        for i in range(len(a['inputs'])):
            if a.get('labels') and int(a['labels'][i]) != 1:
                continue
            plot_heatmap(
                a['inputs'][i],
                a['attention_scores'][i],
                a.get('localization_masks', [None] * len(a['inputs']))[i],
                c['data']['pad_id'],
                f'Attention heatmap {model_name} example {i}',
                f'{output_dir}/figures/{model_name}_attention_heatmap_example_{i}.png'
            )
            count += 1
            if count >= c['evaluation']['num_heatmap_examples']: break
        scores,masks=_anomaly_rows(a,'attention_scores')
        plot_aggregate_token_heatmap(
            scores,
            masks,
            f'Aggregate attention heatmap: {model_name}',
            f'{output_dir}/figures/aggregate_heatmap_{_safe_name(model_name)}_attention.png'
        )
    lines=['# Experiment summary','']
    if not final.empty: lines += ['## Detection / adaptive', '', final.to_markdown(index=False), '']
    if not loc.empty: lines += ['## Localization', '', loc.to_markdown(index=False), '']
    Path(output_dir,'reports','summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Report assets generated: {output_dir}/')
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
