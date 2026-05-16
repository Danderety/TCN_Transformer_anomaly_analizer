import torch.nn as nn
from src.models.tcn import TCNEncoder

class AttentionCaptureEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_attention = False
        self.last_attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Explicit forward avoids PyTorch's eval/no_grad fastpath, which can
        # bypass _sa_block and therefore skip attention capture.
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn_weights.detach() if attn_weights is not None else None
        return self.dropout1(x)

class TCNTransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, tcn_layers=4, transformer_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, pad_id=0):
        super().__init__(); self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tcn = TCNEncoder(d_model, tcn_layers, dropout=dropout)
        layer = AttentionCaptureEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, vocab_size)

    def _set_attention_capture(self, enabled):
        for layer in self.transformer.layers:
            if hasattr(layer, 'capture_attention'):
                layer.capture_attention = bool(enabled)
                layer.last_attn_weights = None

    def get_last_attention(self):
        weights = []
        for layer in self.transformer.layers:
            attn = getattr(layer, 'last_attn_weights', None)
            if attn is not None:
                weights.append(attn)
        return weights

    def forward(self, x, return_embeddings=False, return_attention=False):
        mask = x.eq(self.pad_id); emb = self.embedding(x); z = self.tcn(emb)
        self._set_attention_capture(return_attention)
        z = self.transformer(z, src_key_padding_mask=mask); logits = self.decoder(z)
        if return_attention and return_embeddings:
            return logits, emb, self.get_last_attention()
        if return_attention:
            return logits, self.get_last_attention()
        return (logits, emb) if return_embeddings else logits
