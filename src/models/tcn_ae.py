import torch.nn as nn
from src.models.tcn import TCNEncoder

class TCNAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, tcn_layers=4, dropout=0.1, pad_id=0):
        super().__init__(); self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.encoder = TCNEncoder(d_model, tcn_layers, dropout=dropout)
        self.decoder = nn.Linear(d_model, vocab_size)
    def forward(self, x, return_embeddings=False):
        emb = self.embedding(x); z = self.encoder(emb); logits = self.decoder(z)
        return (logits, emb) if return_embeddings else logits
