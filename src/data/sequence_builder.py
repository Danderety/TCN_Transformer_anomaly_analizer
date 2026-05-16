def build_event_sequences(df, min_seq_len=5):
    sequences, labels, session_ids = [], [], []
    df = df.sort_values(['session_id', 'timestamp'])
    for sid, g in df.groupby('session_id', sort=False):
        seq = [int(x) for x in g['event_id'].tolist()]
        if len(seq) >= min_seq_len:
            sequences.append(seq)
            labels.append(int(g['label'].max()))
            session_ids.append(str(sid))
    return sequences, labels, session_ids

def window_sequences(sequences, labels, session_ids, max_len=128, stride=None):
    stride = stride or max_len // 2
    out_s, out_y, out_id = [], [], []
    for seq, y, sid in zip(sequences, labels, session_ids):
        if len(seq) <= max_len:
            out_s.append(seq); out_y.append(y); out_id.append(sid)
        else:
            for start in range(0, len(seq), stride):
                chunk = seq[start:start+max_len]
                if len(chunk) > 1:
                    out_s.append(chunk); out_y.append(y); out_id.append(f'{sid}__w{start}')
    return out_s, out_y, out_id
