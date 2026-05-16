import torch.nn.functional as F

def gradient_saliency(model, x, pad_id=0):
    model.eval(); logits, emb = model(x, return_embeddings=True); emb.retain_grad(); b,l,v=logits.shape
    loss=F.cross_entropy(logits.reshape(b*l,v),x.reshape(b*l),ignore_index=pad_id,reduction='none').reshape(b,l)
    mask=x.ne(pad_id).float(); score=(loss*mask).sum()/mask.sum().clamp(min=1)
    model.zero_grad(); score.backward(); return emb.grad.abs().sum(-1).detach().cpu().numpy()
