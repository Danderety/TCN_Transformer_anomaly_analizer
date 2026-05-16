import argparse
import subprocess
from pathlib import Path

PIPELINE = [
    'scripts/01_prepare_data.py',
    'scripts/02_train_baseline_tcn_ae.py',
    'scripts/03_train_tcn_transformer_ae.py',
    'scripts/08_train_tcn_transformer_ensemble.py',
    'scripts/04_evaluate_detection.py',
    'scripts/05_evaluate_localization.py',
    'scripts/06_evaluate_adaptive_threshold.py',
    'scripts/07_generate_report_assets.py',
]

def main(config, generate_demo=False):
    if generate_demo:
        subprocess.run(['python', 'scripts/00_generate_demo_data.py', '--output_dir', 'data/raw/demo'], check=True)
    for script in PIPELINE:
        print(f'\n=== {script} ===')
        subprocess.run(['python', script, '--config', config], check=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--generate_demo', action='store_true')
    a = p.parse_args()
    if not Path(a.config).exists():
        raise FileNotFoundError(a.config)
    main(a.config, a.generate_demo)
