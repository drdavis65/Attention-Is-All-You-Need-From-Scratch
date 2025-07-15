import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from attention import MaskedMultiHeadedSelfAttention, MultiHeadedSelfAttention
from feedforward import FeedForward

h = 8 # number of heads
p_drop = 0.1 # probability of dropout
d_ff = 2048 # feedforward hidden layer neurons

class Decoder(nn.Module):
    def __init__(self, d_model : int):
        super().__init__()
        self.MMHSA = MaskedMultiHeadedSelfAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.MHSA = MultiHeadedSelfAttention(d_model, h)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)
        self.FF = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p_drop)
    
    def forward(self, out_embed, encoder_out) -> TensorType[float]:
        masked_attn_out = self.norm1(out_embed + self.dropout1(self.MMHSA(out_embed, out_embed)))
        cross_attn_out = self.norm2(masked_attn_out + self.dropout2(self.MHSA(masked_attn_out, encoder_out)))
        decoder_out = self.norm3(cross_attn_out + self.dropout3(self.FF(cross_attn_out)))
        return decoder_out
