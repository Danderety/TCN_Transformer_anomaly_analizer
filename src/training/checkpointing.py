from pathlib import Path
import torch

def save_checkpoint(model, optimizer, epoch, metrics, config, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'epoch': epoch, 'metrics': metrics, 'config': config}, path)

def load_checkpoint(model, path, device):
    ckpt=torch.load(path, map_location=device); model.load_state_dict(ckpt['model_state_dict']); return model, ckpt
