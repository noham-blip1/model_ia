# -*- coding: utf-8 -*-
from pathlib import Path
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model doit etre divisible par heads.")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.projection(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalAttention(d_model, heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class IANC003(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, heads, layers, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        # ModuleList conserve les noms de poids blocks.0..., identiques a Sequential.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.block_size = int(block_size)

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError("Les entrees doivent avoir la forme [batch, sequence].")
        batch, seq_len = idx.shape
        if seq_len < 1 or seq_len > self.block_size:
            raise ValueError(f"Longueur de sequence invalide : {seq_len} (maximum {self.block_size}).")
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets et idx doivent avoir la meme forme.")
            if not torch.any(targets != -100):
                raise ValueError("Batch sans cible valide : perte non calculable.")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


def build_model_from_checkpoint(checkpoint, tokenizer_vocab_size):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint invalide : dictionnaire attendu.")
    config = dict(checkpoint.get("config") or {})
    required = ("block_size", "d_model", "heads", "layers")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError("Configuration du checkpoint incomplete : " + ", ".join(missing))
    checkpoint_vocab = int(checkpoint.get("vocab_size", -1))
    if checkpoint_vocab != int(tokenizer_vocab_size):
        raise ValueError(
            f"Vocabulaire incompatible : checkpoint={checkpoint_vocab}, tokenizer={tokenizer_vocab_size}."
        )
    model = IANC003(
        vocab_size=checkpoint_vocab,
        block_size=int(config["block_size"]),
        d_model=int(config["d_model"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
        dropout=float(config.get("dropout", 0.1)),
    )
    state = checkpoint.get("model_state")
    if state is None:
        state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Aucun poids model_state/model_state_dict dans le checkpoint.")
    model.load_state_dict(state, strict=True)
    return model, config
