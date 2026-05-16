import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.utils.io import load_json
from src.data.dataset import LogSequenceDataset
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.training.checkpointing import save_checkpoint

def train_model_from_config(config, model_name):
    device=torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{config['data']['splits_dir']}/train.json"); va=load_json(f"{config['data']['splits_dir']}/val.json")
    tr_ds=LogSequenceDataset(tr['sequences'],tr.get('labels'),max_len=config['data']['max_seq_len'],pad_id=config['data']['pad_id'],session_ids=tr.get('session_ids'))
    va_ds=LogSequenceDataset(va['sequences'],va.get('labels'),max_len=config['data']['max_seq_len'],pad_id=config['data']['pad_id'],session_ids=va.get('session_ids'))
    tr_ld=DataLoader(tr_ds,batch_size=config['training']['batch_size'],shuffle=True,num_workers=config['training'].get('num_workers',0))
    va_ld=DataLoader(va_ds,batch_size=config['training']['batch_size'],shuffle=False,num_workers=config['training'].get('num_workers',0))
    model=build_model(model_name,config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    trainer=Trainer(model,opt,device,config['data']['pad_id'],config['training']['gradient_clip'])
    best=float('inf'); patience=0; ckpt=f'outputs/models/{model_name}_best.pt'
    for epoch in range(1,config['training']['epochs']+1):
        tl=trainer.train_epoch(tr_ld); vl=trainer.validate_epoch(va_ld); print(f'{model_name} epoch={epoch:03d} train_loss={tl:.4f} val_loss={vl:.4f}')
        if vl < best:
            best=vl; patience=0; save_checkpoint(model,opt,epoch,{'train_loss':tl,'val_loss':vl},config,ckpt)
        else:
            patience += 1
        if patience >= config['training']['patience']:
            print('Early stopping'); break
