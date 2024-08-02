import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import torch.nn.functional as F


class NMA(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, topk_ratios=None):
        super(NMA, self).__init__()
        self.scale = scale    # None
        self.output_attention = output_attention    # True
        self.dropout = nn.Dropout(attention_dropout)

        # Assuming topk_ratios is a list of ratios for top-k attention
        self.topk_ratios = topk_ratios if topk_ratios else [0.67, 0.75, 0.8]
        self.attn_weights = nn.Parameter(torch.ones(len(self.topk_ratios)), requires_grad=True)
        # self.attn_weights = nn.Parameter(torch.full((len(self.topk_ratios),), 0.2), requires_grad=True)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # 256  100  8  64
        _, S, _, D = values.shape  # 100  64
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) *scale  # [batch_size, head, window_size, window_size]

        out = 0
        for i, ratio in enumerate(self.topk_ratios):
            topk_val, topk_idx = torch.topk(scores, k=int(S * ratio), dim=-1, sorted=False)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, value=1)
            topk_attn = scores.masked_fill(mask == 0, float('-inf'))
            topk_attn = F.softmax(topk_attn, dim=-1)
            topk_attn = self.dropout(topk_attn)
            out += torch.einsum("bhls,bshd->blhd", topk_attn, values) * self.attn_weights[i]

        if self.output_attention:
            return out, topk_attn
        else:
            return out, None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # queries, keys, values维度均为：[batch_size, window_size, 512]    经过embedding之后的噪声序列
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)   # [batch_size, window_size, 8, 64]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [batch_size, window_size, 8, 64]
        values = self.value_projection(values).view(B, S, H, -1)   # [batch_size, window_size, 8, 64]

        output = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out, A = output
        out = out.reshape(B, L, -1)
        out = self.out_projection(out)

        return out
