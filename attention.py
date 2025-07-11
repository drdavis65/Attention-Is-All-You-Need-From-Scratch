import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

N = 6 # number of layers
h = 8 # number of heads
d_model = 512 # embedding dimensions

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim : int, head_size : int):
        super().__init__()
        self.get_keys = nn.Linear(embedding_dim, head_size, bias = False)
        self.get_queries = nn.Linear(embedding_dim, head_size, bias = False)
        self.get_values = nn.Linear(embedding_dim, head_size, bias = False)

    def forward(self, q, kv, mask=None) -> TensorType[float]:
        # Batch Size x Context (Sequence Length) x Embedding Dimensions
        K = self.get_keys(kv)
        Q = self.get_queries(q)
        V = self.get_values(kv)

        scores = (Q @ K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        output = attn_weights @ V

        return output
        
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int):
        super().__init__()
        self.attn_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attn_heads.append(SingleHeadSelfAttention(embedding_dim, embedding_dim // num_heads))
    
    def forward(self, q, kv) -> TensorType[float]:
        head_outputs = []
        for head in self.attn_heads:
            head_outputs.append(head(q, kv))
        concatentated = torch.cat(head_outputs, dim=-1)
        return concatentated

class MaskedMultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int):
        super().__init__()
        self.attn_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attn_heads.append(SingleHeadSelfAttention(embedding_dim, embedding_dim // num_heads))
    
    def forward(self, q, kv) -> TensorType[float]:
        T = q.shape[1]
        mask = torch.tril(torch.ones(T, T, device = q.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        head_outputs = []
        for head in self.attn_heads:
            head_outputs.append(head(q, kv, mask))
        concatenated = torch.cat(head_outputs, dim=-1)
        return concatenated

        
          






    
