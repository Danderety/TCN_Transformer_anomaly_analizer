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

def split(seqs, labels, ids, masks, test_size, seed):
    strat = labels if len(set(labels)) > 1 and min(labels.count(0), labels.count(1)) >= 2 else None
    return train_test_split(seqs, labels, ids, masks, test_size=test_size, random_state=seed, stratify=strat)

def main(config_path):
    c=load_config(config_path); set_seed(c['project']['seed'])
    ensure_dir(c['data']['processed_dir']); ensure_dir(c['data']['splits_dir']); ensure_dir(c['project']['output_dir'])
    print(f"Loading raw dataset={c['data']['dataset_name']} from {c['data']['raw_dir']}", flush=True)
    df=get_adapter(
        c['data']['dataset_name'],
        c['data']['raw_dir'],
        max_files=c['data'].get('max_files'),
        max_lines_per_file=c['data'].get('max_lines_per_file') or c['data'].get('max_raw_rows'),
        max_runs_per_scenario=c['data'].get('max_runs_per_scenario'),
        session_granularity=c['data'].get('session_granularity'),
    ).load()
    if df.empty:
        raise ValueError('Adapter returned an empty dataframe. Check raw_dir and adapter settings.')
    print(f"Loaded rows={len(df):,} sessions={df['session_id'].nunique():,} labels={df['label'].value_counts().to_dict()}", flush=True)
    print('Cleaning messages and building event templates...', flush=True)
    df['clean_message']=df['raw_message'].apply(lambda x: clean_log_message(x, c['preprocessing']['lowercase'], c['preprocessing']['remove_timestamps'], c['preprocessing']['remove_numbers']))
    df['template']=df['clean_message'].apply(simple_template_parser)
    vocab=EventVocab(); vocab.build(df['template'].tolist()); df['event_id']=df['template'].apply(vocab.encode)
    print(f"Built vocab_size={len(vocab):,}; building sequences...", flush=True)
    seqs, labels, ids, masks = build_event_sequences(df, c['data']['min_seq_len'], return_masks=True)
    scale_lengths = c.get('multi_scale', {}).get('lengths', []) if c.get('multi_scale', {}).get('enabled', False) else []
    sequence_window_len = max([int(c['data']['max_seq_len'])] + [int(x) for x in scale_lengths])
    seqs, labels, ids, masks = window_sequences(seqs, labels, ids, sequence_window_len, c['data'].get('window_stride'), masks=masks)
    if len(seqs) < 10: raise ValueError('Too few sequences. Check adapter/session_id/min_seq_len.')
    print(f"Sequences={len(seqs):,} label_counts={{0: {labels.count(0)}, 1: {labels.count(1)}}}", flush=True)
    no_split = bool(c['data'].get('no_split', False) or c.get('evaluation', {}).get('full_real_evaluation', False))
    if no_split:
        tr_s, tr_y, tr_id, tr_m = list(seqs), list(labels), list(ids), list(masks)
        va_s, va_y, va_id, va_m = [], [], [], []
        te_s, te_y, te_id, te_m = list(seqs), list(labels), list(ids), list(masks)
    else:
        tr_s, te_s, tr_y, te_y, tr_id, te_id, tr_m, te_m = split(seqs, labels, ids, masks, c['data']['test_size'], c['project']['seed'])
        tr_s, va_s, tr_y, va_y, tr_id, va_id, tr_m, va_m = split(tr_s, tr_y, tr_id, tr_m, c['data']['val_size'], c['project']['seed'])
    train_before_filter_y = list(tr_y)
    normal_tr_s=[s for s,y in zip(tr_s,tr_y) if int(y)==0]; normal_tr_id=[sid for sid,y in zip(tr_id,tr_y) if int(y)==0]; normal_tr_m=[m for m,y in zip(tr_m,tr_y) if int(y)==0]
    if not normal_tr_s: raise ValueError('No normal train sequences. Check label inference in adapter.')
    if no_split:
        va_s, va_y, va_id, va_m = list(normal_tr_s), [0]*len(normal_tr_s), list(normal_tr_id), list(normal_tr_m)
    normal_va_s=[s for s,y in zip(va_s,va_y) if int(y)==0] or normal_tr_s[:500]
    normal_te_s=[s for s,y in zip(te_s,te_y) if int(y)==0] or normal_tr_s[:500]
    vsy_s, vsy_y, vsy_m, vsy_t = create_synthetic_dataset(normal_va_s, len(vocab), c['synthetic_anomalies']['anomaly_ratio'])
    sy_s, sy_y, sy_m, sy_t = create_synthetic_dataset(normal_te_s, len(vocab), c['synthetic_anomalies']['anomaly_ratio'])
    save_json({'sequences':normal_tr_s,'labels':[0]*len(normal_tr_s),'localization_masks':normal_tr_m,'session_ids':normal_tr_id}, f"{c['data']['splits_dir']}/train.json")
    save_json({'sequences':va_s,'labels':va_y,'localization_masks':va_m,'session_ids':va_id}, f"{c['data']['splits_dir']}/val.json")
    save_json({'sequences':te_s,'labels':te_y,'localization_masks':te_m,'session_ids':te_id}, f"{c['data']['splits_dir']}/test.json")
    save_json({'sequences':seqs,'labels':labels,'localization_masks':masks,'session_ids':ids}, f"{c['data']['splits_dir']}/full_real.json")
    save_json({'sequences':vsy_s,'labels':vsy_y,'localization_masks':vsy_m,'anomaly_types':vsy_t,'session_ids':[f'val_synthetic_{i}' for i in range(len(vsy_s))]}, f"{c['data']['splits_dir']}/val_synthetic.json")
    save_json({'sequences':sy_s,'labels':sy_y,'localization_masks':sy_m,'anomaly_types':sy_t,'session_ids':[f'synthetic_{i}' for i in range(len(sy_s))]}, f"{c['data']['splits_dir']}/test_synthetic.json")
    prepare_summary = {
        'dataset': c['data']['dataset_name'],
        'levels': [
            {
                'level': 'raw_rows',
                'description': 'original log/event rows before session aggregation',
                'normal': int((df['label'] == 0).sum()),
                'anomaly': int((df['label'] == 1).sum()),
                'total': int(len(df)),
            },
            {
                'level': 'raw_sessions',
                'description': 'unique session_id values in raw dataframe',
                'normal': None,
                'anomaly': None,
                'total': int(df['session_id'].nunique()),
            },
            {
                'level': 'windowed_sequences_before_split',
                'description': 'model samples after session aggregation/windowing, before train anomaly filtering',
                'normal': int(sum(int(y) == 0 for y in labels)),
                'anomaly': int(sum(int(y) == 1 for y in labels)),
                'total': int(len(labels)),
            },
            {
                'level': 'train_before_filter',
                'description': 'train split before keeping only normal samples for AE training',
                'normal': int(sum(int(y) == 0 for y in train_before_filter_y)),
                'anomaly': int(sum(int(y) == 1 for y in train_before_filter_y)),
                'total': int(len(train_before_filter_y)),
            },
            {
                'level': 'saved_train_for_ae',
                'description': 'saved train.json; anomalies are intentionally excluded',
                'normal': int(len(normal_tr_s)),
                'anomaly': 0,
                'total': int(len(normal_tr_s)),
            },
            {
                'level': 'saved_validation_for_loss',
                'description': 'validation loss split; in no_split mode this is normal-only',
                'normal': int(sum(int(y) == 0 for y in va_y)),
                'anomaly': int(sum(int(y) == 1 for y in va_y)),
                'total': int(len(va_y)),
            },
            {
                'level': 'saved_test_real',
                'description': 'test.json; in no_split mode this is the full real windowed dataset',
                'normal': int(sum(int(y) == 0 for y in te_y)),
                'anomaly': int(sum(int(y) == 1 for y in te_y)),
                'total': int(len(te_y)),
            },
            {
                'level': 'saved_real_detection_val_test',
                'description': 'real samples used to measure detection; in no_split mode this equals full real test.json',
                'normal': int(sum(int(y) == 0 for y in te_y) if no_split else sum(int(y) == 0 for y in va_y) + sum(int(y) == 0 for y in te_y)),
                'anomaly': int(sum(int(y) == 1 for y in te_y) if no_split else sum(int(y) == 1 for y in va_y) + sum(int(y) == 1 for y in te_y)),
                'total': int(len(te_y) if no_split else len(va_y) + len(te_y)),
            },
            {
                'level': 'saved_synthetic_val_test',
                'description': 'synthetic val+test samples used as auxiliary stress tests',
                'normal': int(sum(int(y) == 0 for y in vsy_y) + sum(int(y) == 0 for y in sy_y)),
                'anomaly': int(sum(int(y) == 1 for y in vsy_y) + sum(int(y) == 1 for y in sy_y)),
                'total': int(len(vsy_y) + len(sy_y)),
            },
        ],
        'no_split': no_split,
    }
    save_json(prepare_summary, f"{c['data']['processed_dir']}/prepare_summary.json")
    vocab.save(f"{c['data']['processed_dir']}/event_vocab.json"); df.to_csv(f"{c['data']['processed_dir']}/parsed_logs.csv", index=False)
    output_dir=c.get('project', {}).get('output_dir', 'outputs')
    c['model']['vocab_size']=len(vocab); used=f"{c['data']['processed_dir']}/used_config.yaml"; save_config(c, used); ensure_dir(f'{output_dir}/configs'); save_config(c, f"{output_dir}/configs/{Path(config_path).stem}_used.yaml")
    print(f"Prepared. mode={'no_split_full_real' if no_split else 'train_val_test_split'} vocab={len(vocab)} train_normal={len(normal_tr_s)} val={len(va_s)} test={len(te_s)} synthetic={len(sy_s)}")
    print(
        "Label summary: "
        f"train={{0: {len(normal_tr_s)}, 1: 0}} "
        f"val={{0: {sum(int(y)==0 for y in va_y)}, 1: {sum(int(y)==1 for y in va_y)}}} "
        f"test={{0: {sum(int(y)==0 for y in te_y)}, 1: {sum(int(y)==1 for y in te_y)}}} "
        f"val_synthetic={{0: {sum(int(y)==0 for y in vsy_y)}, 1: {sum(int(y)==1 for y in vsy_y)}}} "
        f"synthetic={{0: {sum(int(y)==0 for y in sy_y)}, 1: {sum(int(y)==1 for y in sy_y)}}}",
        flush=True,
    )
    print(f"Use config: {used}")

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); main(p.parse_args().config)
