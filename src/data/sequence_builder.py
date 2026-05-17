def build_event_sequences(df, min_seq_len=5, return_masks=False):
    sequences, labels, session_ids, masks = [], [], [], []
    df = df.sort_values(['session_id', 'timestamp'])
    for sid, g in df.groupby('session_id', sort=False):
        seq = [int(x) for x in g['event_id'].tolist()]
        row_mask = [int(x) for x in g['label'].tolist()]
        if len(seq) >= min_seq_len:
            sequences.append(seq)
            labels.append(int(max(row_mask)))
            session_ids.append(str(sid))
            masks.append(row_mask)
    if return_masks:
        return sequences, labels, session_ids, masks
    return sequences, labels, session_ids

def window_sequences(sequences, labels, session_ids, max_len=128, stride=None, masks=None):
    stride = stride or max_len // 2
    out_s, out_y, out_id, out_m = [], [], [], []
    if masks is None:
        masks = [[int(y)] * len(seq) for seq, y in zip(sequences, labels)]
        return_masks = False
    else:
        return_masks = True
    for seq, y, sid, mask in zip(sequences, labels, session_ids, masks):
        if len(seq) <= max_len:
            out_s.append(seq); out_y.append(int(max(mask) if mask else y)); out_id.append(sid); out_m.append(mask)
        else:
            for start in range(0, len(seq), stride):
                chunk = seq[start:start+max_len]
                chunk_mask = mask[start:start+max_len]
                if len(chunk) > 1:
                    out_s.append(chunk); out_y.append(int(max(chunk_mask) if chunk_mask else y)); out_id.append(f'{sid}__w{start}'); out_m.append(chunk_mask)
    if return_masks:
        return out_s, out_y, out_id, out_m
    return out_s, out_y, out_id
