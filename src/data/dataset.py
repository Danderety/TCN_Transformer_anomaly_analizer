import torch
from torch.utils.data import Dataset

def pad_sequence(seq, max_len, pad_id=0):
    seq = [int(x) for x in seq[:max_len]]
    return seq + [pad_id] * (max_len - len(seq))

def pad_mask(mask, max_len):
    mask = [int(x) for x in mask[:max_len]]
    return mask + [0] * (max_len - len(mask))

class LogSequenceDataset(Dataset):
    def __init__(self, sequences, labels=None, localization_masks=None, max_len=128, pad_id=0, session_ids=None):
        self.sequences = sequences; self.labels = labels; self.localization_masks = localization_masks
        self.max_len = max_len; self.pad_id = pad_id; self.session_ids = session_ids
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        item = {'x': torch.tensor(pad_sequence(self.sequences[idx], self.max_len, self.pad_id), dtype=torch.long)}
        if self.labels is not None: item['y'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.localization_masks is not None: item['loc_mask'] = torch.tensor(pad_mask(self.localization_masks[idx], self.max_len), dtype=torch.long)
        if self.session_ids is not None: item['session_id'] = self.session_ids[idx]
        return item
