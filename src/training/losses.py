import torch.nn.functional as F

def masked_cross_entropy_loss(logits, target, pad_id=0):
    b, l, v = logits.shape
    return F.cross_entropy(logits.reshape(b*l, v), target.reshape(b*l), ignore_index=pad_id, reduction='mean')
