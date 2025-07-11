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
            self.get_querys = nn.Linear(embedding_dim, head_size, bias = False)
            self.get_values = nn.Linear(embedding_dim, head_size, bias = False)

        def forward(self, q, k_v, mask=None) -> TensorType[float]:
            # Batch Size x Context (Sequence Length) x Embedding Dimensions
            K = self.get_keys(k_v)
            Q = self.get_querys(q)
            V = self.get_values(k_v)

            scores = (Q @ K.transpose(-2, -1)) / (K.shape[2] ** 0.5)

            scores = scores.masked_fill(mask, float('inf'))
            attn_weights = F.softmax(dim=-1)

            output = attn_weights @ V

            return output






    
