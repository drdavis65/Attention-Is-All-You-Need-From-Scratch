import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, T_q, _ = q.shape
        _, T_k, _ = kv.shape

        # Project separately
        Q = self.q_proj(q)  # [B, T_q, d_model]
        K = self.k_proj(kv) # [B, T_k, d_model]
        V = self.v_proj(kv) # [B, T_k, d_model]

        # Reshape for multi-head
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T_q, d_h]
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T_k, d_h]
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T_k, d_h]

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, h, T_q, T_k]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, h, T_q, T_k]

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)  # [B, h, T_q, d_h]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.head_dim)  # [B, T_q, d_model]
        return self.out_proj(attn_output)

class MaskedMultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Combine QKV into one projection
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                      # [B, T, 3 * d_model]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)  # [B, T, 3, h, d_h]
        qkv = qkv.permute(2, 0, 3, 1, 4)            # [3, B, h, T, d_h]
        Q, K, V = qkv[0], qkv[1], qkv[2]            # [B, h, T, d_h] each

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, h, T, T]

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()  # [T, T]
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, h, T, T]
        attn_output = torch.matmul(attn_weights, V)    # [B, h, T, d_h]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T, h, d_h]
        attn_output = attn_output.view(B, T, self.num_heads * self.head_dim)  # [B, T, d_model]

        return self.out_proj(attn_output)

        
class FeedForward(nn.Module):
    def __init__(self, d_model : int, d_ff : int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x) -> TensorType[float]:
        return self.net(x)

