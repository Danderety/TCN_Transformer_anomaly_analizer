import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
from src.utils.config import load_config
from src.utils.seed import set_seed
from scripts.train_common import train_model_from_config


def main(config_path):
    config = load_config(config_path)
    ensemble_config = config.get('ensemble', {})
    if not ensemble_config.get('enabled', False):
        print('Skip ensemble training: ensemble.enabled=false')
        return
    seeds = ensemble_config.get('member_seeds', [config['project']['seed']])
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    for seed in seeds:
        member_config = copy.deepcopy(config)
        member_config['project']['seed'] = int(seed)
        set_seed(int(seed))
        checkpoint_path = Path(output_dir) / 'models' / f'tcn_transformer_ae_seed{int(seed)}_best.pt'
        print(f'Training ensemble member seed={seed} -> {checkpoint_path}')
        train_model_from_config(member_config, 'tcn_transformer_ae', str(checkpoint_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)
