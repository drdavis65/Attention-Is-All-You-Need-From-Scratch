import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from attention import MultiHeadedSelfAttention
from feedforward import FeedForward

h = 8 # number of heads
p_drop = 0.1 # probability of dropout
d_ff = 2048 # feedforward hidden layer neurons

class Encoder(nn.Module):
    def __init__(self, d_model : int):
        super().__init__()
        self.MHSA = MultiHeadedSelfAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.FF = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)
    
    def forward(self, in_embed) -> TensorType[float]:
        attn_out = self.norm1(in_embed + self.dropout1(self.MHSA(in_embed, in_embed)))
        encoder_out = self.norm2(attn_out + self.dropout2(self.FF(attn_out)))
        return encoder_out


