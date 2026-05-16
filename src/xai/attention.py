import numpy as np
import torch


def attention_token_scores(attention_layers, input_ids=None, pad_id=0):
    """Aggregate captured self-attention into one score per source token.

    Each layer tensor has shape [batch, heads, query_pos, key_pos]. Averaging
    across layers, heads and query positions gives how much every token was
    attended to by the sequence.
    """
    if not attention_layers:
        raise ValueError('No attention layers captured. Call model(..., return_attention=True).')
    stacked = torch.stack(attention_layers, dim=0).float()
    scores = stacked.mean(dim=(0, 2, 3))
    if input_ids is not None:
        mask = input_ids.ne(pad_id)
        scores = scores * mask.float()
    return scores.detach().cpu().numpy()


@torch.no_grad()
def capture_attention_dataset(model, loader, device, pad_id=0, max_matrices=0):
    out = {
        'attention_scores': [],
        'inputs': [],
        'labels': [],
        'localization_masks': [],
        'session_ids': [],
        'attention_matrices': [],
    }
    model.eval()
    saved_matrices = 0
    for batch in loader:
        x = batch['x'].to(device)
        _logits, attention_layers = model(x, return_attention=True)
        scores = attention_token_scores(attention_layers, x, pad_id)
        out['attention_scores'] += scores.tolist()
        out['inputs'] += x.cpu().numpy().tolist()
        if 'y' in batch:
            out['labels'] += batch['y'].cpu().numpy().tolist()
        if 'loc_mask' in batch:
            out['localization_masks'] += batch['loc_mask'].cpu().numpy().tolist()
        if 'session_id' in batch:
            out['session_ids'] += list(batch['session_id'])
        if max_matrices and saved_matrices < max_matrices:
            for i in range(x.shape[0]):
                if saved_matrices >= max_matrices:
                    break
                out['attention_matrices'].append({
                    'sample_index': len(out['inputs']) - x.shape[0] + i,
                    'layers': [layer[i].detach().cpu().numpy().tolist() for layer in attention_layers],
                })
                saved_matrices += 1
    if not max_matrices:
        out.pop('attention_matrices')
    return out


def average_attention_results(results):
    if not results:
        raise ValueError('results is empty')
    scores = np.mean([np.asarray(r['attention_scores'], dtype=np.float32) for r in results], axis=0)
    first = results[0]
    return {
        'attention_scores': scores.tolist(),
        'inputs': first['inputs'],
        'labels': first.get('labels', []),
        'localization_masks': first.get('localization_masks', []),
        'session_ids': first.get('session_ids', []),
    }
