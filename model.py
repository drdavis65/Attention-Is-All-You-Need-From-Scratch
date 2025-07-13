import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

import decoder_block, encoder_block


N = 6 # number of layers
h = 8 # number of heads
d_model = 512 # embedding dimensions
d_ff = 2048 # feedforward hidden layer

class Transformer(nn.Module):
    def __init__(self, d_model : int):
        super().__init__()
        self.encoder_blocks = nn.Sequential()
        self.decoder_blocks = nn.Sequential()
        for _ in range(N):
            self.encoder_blocks.append(encoder_block.Encoder(d_model))
            self.decoder_blocks.append(decoder_block.Decoder(d_model))
        