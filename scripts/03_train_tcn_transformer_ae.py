import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.utils.config import load_config
from src.utils.seed import set_seed
from scripts.train_common import train_model_from_config

def main(config_path):
    c=load_config(config_path); set_seed(c['project']['seed']); train_model_from_config(c, 'tcn_transformer_ae')
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
