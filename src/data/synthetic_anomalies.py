import random, copy

def inject_synthetic_anomaly(seq, vocab_size, anomaly_type=None, pad_id=0):
    seq = [int(x) for x in seq if int(x) != pad_id]
    if len(seq) < 4:
        return seq, [0] * len(seq), 'none'
    anomaly_type = anomaly_type or random.choice(['insert', 'delete', 'replace', 'swap', 'repeat'])
    new_seq = copy.deepcopy(seq); mask = [0] * len(new_seq)
    if anomaly_type == 'insert':
        pos = random.randint(1, len(new_seq)-1); ev = random.randint(2, max(2, vocab_size-1))
        new_seq.insert(pos, ev); mask.insert(pos, 1)
    elif anomaly_type == 'delete':
        pos = random.randint(1, len(new_seq)-2); del new_seq[pos]
        mask = [0] * len(new_seq); mask[min(pos, len(mask)-1)] = 1
    elif anomaly_type == 'replace':
        pos = random.randint(1, len(new_seq)-2); new_seq[pos] = random.randint(2, max(2, vocab_size-1)); mask[pos] = 1
    elif anomaly_type == 'swap':
        pos = random.randint(1, len(new_seq)-3); new_seq[pos], new_seq[pos+1] = new_seq[pos+1], new_seq[pos]; mask[pos] = mask[pos+1] = 1
    elif anomaly_type == 'repeat':
        pos = random.randint(1, len(new_seq)-2); new_seq.insert(pos, new_seq[pos]); mask.insert(pos, 1)
    return new_seq, mask, anomaly_type

def create_synthetic_dataset(normal_sequences, vocab_size, anomaly_ratio=0.5):
    seqs, labels, masks, types = [], [], [], []
    for seq in normal_sequences:
        if random.random() < anomaly_ratio:
            s, m, t = inject_synthetic_anomaly(seq, vocab_size)
            seqs.append(s); labels.append(1); masks.append(m); types.append(t)
        else:
            seqs.append(seq); labels.append(0); masks.append([0]*len(seq)); types.append('normal')
    return seqs, labels, masks, types
