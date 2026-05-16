import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.adapters import get_adapter
from src.data.preprocessing import clean_log_message, simple_template_parser
from src.data.vocab import EventVocab
from src.data.sequence_builder import build_event_sequences, window_sequences
from src.data.synthetic_anomalies import create_synthetic_dataset

def split(seqs, labels, ids, test_size, seed):
    strat = labels if len(set(labels)) > 1 and min(labels.count(0), labels.count(1)) >= 2 else None
    return train_test_split(seqs, labels, ids, test_size=test_size, random_state=seed, stratify=strat)

def main(config_path):
    c=load_config(config_path); set_seed(c['project']['seed'])
    ensure_dir(c['data']['processed_dir']); ensure_dir(c['data']['splits_dir']); ensure_dir(c['project']['output_dir'])
    df=get_adapter(c['data']['dataset_name'], c['data']['raw_dir']).load()
    df['clean_message']=df['raw_message'].apply(lambda x: clean_log_message(x, c['preprocessing']['lowercase'], c['preprocessing']['remove_timestamps'], c['preprocessing']['remove_numbers']))
    df['template']=df['clean_message'].apply(simple_template_parser)
    vocab=EventVocab(); vocab.build(df['template'].tolist()); df['event_id']=df['template'].apply(vocab.encode)
    seqs, labels, ids = build_event_sequences(df, c['data']['min_seq_len'])
    seqs, labels, ids = window_sequences(seqs, labels, ids, c['data']['max_seq_len'])
    if len(seqs) < 10: raise ValueError('Too few sequences. Check adapter/session_id/min_seq_len.')
    tr_s, te_s, tr_y, te_y, tr_id, te_id = split(seqs, labels, ids, c['data']['test_size'], c['project']['seed'])
    tr_s, va_s, tr_y, va_y, tr_id, va_id = split(tr_s, tr_y, tr_id, c['data']['val_size'], c['project']['seed'])
    normal_tr_s=[s for s,y in zip(tr_s,tr_y) if int(y)==0]; normal_tr_id=[sid for sid,y in zip(tr_id,tr_y) if int(y)==0]
    if not normal_tr_s: raise ValueError('No normal train sequences. Check label inference in adapter.')
    normal_for_synth=[s for s,y in zip(va_s,va_y) if int(y)==0] or normal_tr_s[:500]
    sy_s, sy_y, sy_m, sy_t = create_synthetic_dataset(normal_for_synth, len(vocab), c['synthetic_anomalies']['anomaly_ratio'])
    save_json({'sequences':normal_tr_s,'labels':[0]*len(normal_tr_s),'session_ids':normal_tr_id}, f"{c['data']['splits_dir']}/train.json")
    save_json({'sequences':va_s,'labels':va_y,'session_ids':va_id}, f"{c['data']['splits_dir']}/val.json")
    save_json({'sequences':te_s,'labels':te_y,'session_ids':te_id}, f"{c['data']['splits_dir']}/test.json")
    save_json({'sequences':sy_s,'labels':sy_y,'localization_masks':sy_m,'anomaly_types':sy_t,'session_ids':[f'synthetic_{i}' for i in range(len(sy_s))]}, f"{c['data']['splits_dir']}/test_synthetic.json")
    vocab.save(f"{c['data']['processed_dir']}/event_vocab.json"); df.to_csv(f"{c['data']['processed_dir']}/parsed_logs.csv", index=False)
    output_dir=c.get('project', {}).get('output_dir', 'outputs')
    c['model']['vocab_size']=len(vocab); used=f"{c['data']['processed_dir']}/used_config.yaml"; save_config(c, used); ensure_dir(f'{output_dir}/configs'); save_config(c, f"{output_dir}/configs/{Path(config_path).stem}_used.yaml")
    print(f"Prepared. vocab={len(vocab)} train_normal={len(normal_tr_s)} val={len(va_s)} test={len(te_s)} synthetic={len(sy_s)}")
    print(f"Use config: {used}")

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
