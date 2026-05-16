import numpy as np
import torch
from src.training.losses import masked_cross_entropy_loss

class Trainer:
    def __init__(self, model, optimizer, device, pad_id=0, gradient_clip=1.0):
        self.model=model; self.optimizer=optimizer; self.device=device; self.pad_id=pad_id; self.gradient_clip=gradient_clip
    def train_epoch(self, loader):
        self.model.train(); losses=[]
        for batch in loader:
            x=batch['x'].to(self.device); self.optimizer.zero_grad(set_to_none=True)
            loss=masked_cross_entropy_loss(self.model(x), x, self.pad_id); loss.backward()
            if self.gradient_clip: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step(); losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0
    @torch.no_grad()
    def validate_epoch(self, loader):
        self.model.eval(); losses=[]
        for batch in loader:
            x=batch['x'].to(self.device); losses.append(masked_cross_entropy_loss(self.model(x), x, self.pad_id).item())
        return float(np.mean(losses)) if losses else 0.0
