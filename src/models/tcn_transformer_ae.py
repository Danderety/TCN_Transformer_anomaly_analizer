import torch.nn as nn
from src.models.tcn import TCNEncoder

class TCNTransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, tcn_layers=4, transformer_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, pad_id=0):
        super().__init__(); self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tcn = TCNEncoder(d_model, tcn_layers, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
    def forward(self, x, return_embeddings=False):
        mask = x.eq(self.pad_id); emb = self.embedding(x); z = self.tcn(emb)
        z = self.transformer(z, src_key_padding_mask=mask); logits = self.decoder(z)
        return (logits, emb) if return_embeddings else logits
