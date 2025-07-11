import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadSelfAttention(nn.Module):
        def __init__(self, embedding_dim : int, head_size : int):
            super().__init__()
            self.get_keys = nn.Linear(embedding_dim, head_size, bias = False)
            self.get_querys = nn.Linear(embedding_dim, head_size, bias = False)
            self.get_values = nn.Linear(embedding_dim, head_size, bias = False)

        def forward(self, q, k, v, head_size, mask=None) -> TensorType[float]:
              
    
