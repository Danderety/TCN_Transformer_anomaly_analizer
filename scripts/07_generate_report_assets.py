import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd
from src.utils.config import load_config
from src.utils.io import load_json, ensure_dir
from src.visualization.plots import plot_score_distribution, plot_threshold_dynamics, plot_heatmap, plot_model_comparison

def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()

def main(config_path):
    c=load_config(config_path); output_dir=c.get('project', {}).get('output_dir', 'outputs'); ensure_dir(f'{output_dir}/figures'); ensure_dir(f'{output_dir}/reports'); ensure_dir(f'{output_dir}/metrics')
    det=read_csv(f'{output_dir}/metrics/detection_metrics.csv'); ad=read_csv(f'{output_dir}/metrics/adaptive_threshold_metrics.csv'); loc=read_csv(f'{output_dir}/metrics/localization_metrics.csv')
    final=pd.concat([x for x in [det,ad] if not x.empty], ignore_index=True) if (not det.empty or not ad.empty) else pd.DataFrame()
    if not final.empty:
        final.to_csv(f'{output_dir}/metrics/final_comparison.csv',index=False); plot_model_comparison(final,'f1',f'{output_dir}/figures/model_comparison_f1.png')
    pred=Path(output_dir)/'predictions'/'tcn_transformer_ae_test_predictions.json'
    if pred.exists():
        pr=load_json(pred); ns=[s for s,y in zip(pr['scores'],pr['labels']) if int(y)==0]; ans=[s for s,y in zip(pr['scores'],pr['labels']) if int(y)==1]
        plot_score_distribution(ns,ans,f'{output_dir}/figures/anomaly_score_distribution.png')
    ap=Path(output_dir)/'predictions'/'adaptive_predictions.json'
    if ap.exists():
        a=load_json(ap); plot_threshold_dynamics(a['scores'],a['thresholds'],a['labels'],f'{output_dir}/figures/adaptive_threshold_dynamics.png')
    lp=Path(output_dir)/'predictions'/'tcn_transformer_ae_localization_predictions.json'
    if lp.exists():
        l=load_json(lp); count=0
        for i in range(len(l['inputs'])):
            if int(l['labels'][i]) != 1: continue
            plot_heatmap(l['inputs'][i],l['token_errors'][i],l['localization_masks'][i],c['data']['pad_id'],f'Heatmap example {i}',f'{output_dir}/figures/heatmap_example_{i}.png')
            count += 1
            if count >= c['evaluation']['num_heatmap_examples']: break
    lines=['# Experiment summary','']
    if not final.empty: lines += ['## Detection / adaptive', '', final.to_markdown(index=False), '']
    if not loc.empty: lines += ['## Localization', '', loc.to_markdown(index=False), '']
    Path(output_dir,'reports','summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Report assets generated: {output_dir}/')
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
