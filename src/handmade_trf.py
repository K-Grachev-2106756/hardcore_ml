from dataclasses import dataclass

import torch 
from torch import nn
import torch.nn.functional as F



@dataclass
class TrfConfig:
    emb_dim: int
    context_size: int
    head_size: int
    n_heads: int
    n_blocks: int
    vocab_size: int
    dropout: float = 0.0


class SelfAttentionHead(nn.Module):

    def __init__(self, cfg: TrfConfig):
        super().__init__()
        self.queries = nn.Linear(cfg.emb_dim, cfg.head_size, bias=False)
        self.keys = nn.Linear(cfg.emb_dim, cfg.head_size, bias=False)
        self.values = nn.Linear(cfg.emb_dim, cfg.head_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)  # Dropout: a simple way to prevent nn from overfitting (2014)

        self.register_buffer("tril", torch.tril(torch.ones(cfg.context_size, cfg.context_size)))

    
    def forward(self, idx):
        b, t, c = idx.shape
        
        wei = self.keys(idx) @ self.queries(idx).transpose(-2, -1) * c ** (-0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # Диагональ применяется для блока декодера, 
                                                                      # в энкодер блоках допускается взаимосвязь между предыдущими и последующими токенами
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ self.values(idx)


class FlashSelfAttention(nn.Module):

    def __init__(self, cfg: TrfConfig):
        super().__init__()
        
        assert cfg.emb_dim % cfg.n_heads == 0

        self.attn = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim)  # self.attn(x) выполняет параллельное перемножение с входными данными
                                                             # ключей, запросов и значений, тк они сразу в одном месте
        self.proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)

        self.n_heads = cfg.n_heads
        self.emb_dim = cfg.emb_dim

    
    def forward(self, idx: torch.Tensor):
        B, T, C = idx.size()
        
        qkv = self.attn(idx)
        q, k, v = qkv.split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(k, q, v, is_causal=True)  # Flash-attn
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class MultiHeadAttention(nn.Module):

    def __init__(self, cfg: TrfConfig):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(cfg) for _ in range(cfg.n_heads)])
        self.proj = nn.Linear(cfg.n_heads * cfg.head_size, cfg.n_heads * cfg.head_size)
        self.dropout = nn.Dropout(cfg.dropout)

    
    def forward(self, idx):
        out = torch.cat([h(idx) for h in self.heads], dim=-1)
        out = self.proj(out)  # Линейное преобразование
        
        return self.dropout(out)
    

class FeedForward(nn.Module):

    def __init__(self, cfg: TrfConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),  # Линейное преобразование
            nn.Dropout(cfg.dropout),
        )

    
    def forward(self, idx):
        return self.net(idx)


class TrfBlock(nn.Module):
    
    def __init__(self, cfg: TrfConfig):
        super().__init__()
        self.sa_heads = MultiHeadAttention(cfg)
        self.ffwd_net = FeedForward(cfg)
        self.layer_norm1 = nn.LayerNorm(cfg.emb_dim)  # Layer Normalization (2016)
        self.layer_norm2 = nn.LayerNorm(cfg.emb_dim)
    
    
    def forward(self, idx):
        x = idx + self.sa_heads(self.layer_norm1(idx))  # Deep Residual Learning for Image Recognition (2015)
        x = x + self.ffwd_net(self.layer_norm2(x))
        return x