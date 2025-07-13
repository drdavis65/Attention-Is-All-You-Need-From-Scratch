import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
import math

p_drop = 0.1

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size : int, d_model : int, context_window : int):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p_drop)
        pe = torch.zeros(context_window, d_model)
        pos = torch.arange(0, context_window).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('positional_encoding', pe)
    
    def forward(self, token_ids : TensorType[int]) -> TensorType[float]:
        seq_length = token_ids.size(1)
        tok_embeddings = self.tok_embeddings(token_ids)
        pos_encoding = self.positional_encoding[:seq_length,:].unsqueeze(0)
        return self.dropout(tok_embeddings + pos_encoding)
