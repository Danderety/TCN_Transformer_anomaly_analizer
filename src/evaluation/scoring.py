import torch
import torch.nn.functional as F

@torch.no_grad()
def reconstruction_error_per_position(model, x, pad_id=0):
    model.eval(); logits=model(x); b,l,v=logits.shape
    loss = F.cross_entropy(logits.reshape(b*l,v), x.reshape(b*l), ignore_index=pad_id, reduction='none').reshape(b,l)
    mask=x.ne(pad_id).float(); token_errors=loss*mask; scores=token_errors.sum(1)/mask.sum(1).clamp(min=1)
    return token_errors, scores

@torch.no_grad()
def score_dataset(model, loader, device, pad_id=0, include_details=True):
    out={'scores': [], 'token_errors': [], 'inputs': [], 'labels': [], 'localization_masks': [], 'session_ids': []}
    model.eval()
    for batch in loader:
        x=batch['x'].to(device); te, sc = reconstruction_error_per_position(model, x, pad_id)
        out['scores'] += sc.cpu().numpy().tolist()
        if include_details:
            out['token_errors'] += te.cpu().numpy().tolist()
            out['inputs'] += x.cpu().numpy().tolist()
        if 'y' in batch: out['labels'] += batch['y'].cpu().numpy().tolist()
        if include_details and 'loc_mask' in batch:
            out['localization_masks'] += batch['loc_mask'].cpu().numpy().tolist()
        if 'session_id' in batch: out['session_ids'] += list(batch['session_id'])
        del x, te, sc
    if not include_details:
        out.pop('token_errors')
        out.pop('inputs')
        out.pop('localization_masks')
    return out
